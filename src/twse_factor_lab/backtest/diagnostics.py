"""Turnover diagnostics, engine cross-check, and backtest realism report."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest

logger = logging.getLogger(__name__)


def compute_turnover_diagnostics(
    portfolio_weights: pd.DataFrame,
    *,
    cost_model: CostModel | None = None,
) -> pd.DataFrame:
    """Compute portfolio-level turnover diagnostics from weight DataFrame."""
    if portfolio_weights.empty:
        return _empty_diagnostics()

    # Pivot to wide: rows=execution_date, cols=ticker
    pivot = portfolio_weights.pivot_table(
        index="execution_date",
        columns="ticker",
        values="target_weight",
        aggfunc="last",
    ).fillna(0.0)
    pivot = pivot.sort_index()

    # Daily turnover = sum of absolute weight changes / 2
    weight_diff = pivot.diff().fillna(pivot)
    buy_chg = weight_diff.clip(lower=0.0).sum(axis=1)
    sell_chg = (-weight_diff.clip(upper=0.0)).sum(axis=1)
    daily_turnover = buy_chg + sell_chg

    avg_daily = float(daily_turnover.mean())
    median_daily = float(daily_turnover.median())
    max_daily = float(daily_turnover.max())
    annualized = avg_daily * 252

    # Holdings count
    holdings = (pivot > 0).sum(axis=1)
    avg_holdings = float(holdings.mean())

    # Buys and sells per rebalance date (dates where turnover > 0)
    active = daily_turnover[daily_turnover > 0]
    buys_per = float(buy_chg[active.index].mean()) if not active.empty else 0.0
    sells_per = float(sell_chg[active.index].mean()) if not active.empty else 0.0

    # Estimated cost drag
    if cost_model is not None:
        one_way_cost = (cost_model.buy_cost_rate + cost_model.sell_cost_rate) / 2
        estimated_cost_drag = avg_daily * one_way_cost * 252
    else:
        estimated_cost_drag = avg_daily * (0.001425 + 0.001425 + 0.003 + 0.001) * 252

    rows = [
        ("average_daily_turnover", avg_daily, "sum of |Δweight| per day"),
        ("median_daily_turnover", median_daily, "median daily"),
        ("max_daily_turnover", max_daily, "single-day peak"),
        ("annualized_turnover_estimate", annualized, "avg_daily × 252"),
        ("avg_holdings", avg_holdings, "mean non-zero tickers per date"),
        (
            "avg_buys_per_rebalance",
            buys_per,
            "mean positive weight increase on active dates",
        ),
        (
            "avg_sells_per_rebalance",
            sells_per,
            "mean positive weight decrease on active dates",
        ),
        ("estimated_cost_drag", estimated_cost_drag, "avg_daily × one_way_cost × 252"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "notes"])


def _empty_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=["metric", "value", "notes"])


def run_engine_comparison(
    close_matrix: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compare vectorbt vs fallback engine; gracefully handle missing vectorbt."""
    backtest_cfg = config.get("backtest", {}) or {}
    initial_cash = float(backtest_cfg.get("initial_cash", 1_000_000))
    top_n = int(backtest_cfg.get("top_n", 20))
    fees = backtest_cfg.get("fees", {}) or {}
    cost_model = CostModel(
        buy_fee_rate=float(fees.get("buy_fee_rate", 0.001425)),
        sell_fee_rate=float(fees.get("sell_fee_rate", 0.001425)),
        transaction_tax_rate=float(fees.get("transaction_tax_rate", 0.003)),
        slippage_rate=float(fees.get("slippage_rate", 0.001)),
    )

    rows = []

    # Always run fallback
    _, fb_metrics = run_weight_backtest(
        close_matrix=close_matrix,
        portfolio_weights=portfolio_weights,
        cost_model=cost_model,
        initial_cash=initial_cash,
        top_n=top_n,
        use_vectorbt=False,
        allow_fallback=True,
    )
    fb = fb_metrics.iloc[0]
    rows.append(
        {
            "engine": "fallback_weight_engine",
            "total_return": float(fb["total_return"]),
            "sharpe": float(fb["sharpe"]),
            "max_drawdown": float(fb["max_drawdown"]),
            "turnover": float(fb["turnover"]),
            "notes": "deterministic fallback engine",
        }
    )

    # Try vectorbt
    try:
        __import__("vectorbt")
        _, vbt_metrics = run_weight_backtest(
            close_matrix=close_matrix,
            portfolio_weights=portfolio_weights,
            cost_model=cost_model,
            initial_cash=initial_cash,
            top_n=top_n,
            use_vectorbt=True,
            allow_fallback=False,
        )
        vbt = vbt_metrics.iloc[0]
        rows.append(
            {
                "engine": "vectorbt",
                "total_return": float(vbt["total_return"]),
                "sharpe": float(vbt["sharpe"]),
                "max_drawdown": float(vbt["max_drawdown"]),
                "turnover": float(vbt["turnover"]),
                "notes": "vectorbt engine",
            }
        )
    except Exception as exc:
        logger.info("vectorbt unavailable: %s", exc)
        rows.append(
            {
                "engine": "vectorbt",
                "total_return": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "turnover": np.nan,
                "notes": "vectorbt unavailable",
            }
        )

    return pd.DataFrame(rows)


def build_backtest_realism_report(
    *,
    config_path: str | Path,
    config: dict[str, Any],
    close_matrix: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    topn_df: pd.DataFrame,
    rebalance_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    engine_df: pd.DataFrame,
    output_paths: dict[str, Path],
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    backtest_cfg = config.get("backtest", {}) or {}
    factor_name = str(backtest_cfg.get("factor_name", "historical_price_volume"))
    top_n = int(backtest_cfg.get("top_n", 20))
    ticker_count = int(close_matrix.shape[1])

    def _fmt(v: Any, decimals: int = 4) -> str:
        if isinstance(v, float) and np.isnan(v):
            return "NaN"
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)

    def _diag(metric: str) -> str:
        if diagnostics_df.empty:
            return "N/A"
        row = diagnostics_df[diagnostics_df["metric"] == metric]
        if row.empty:
            return "N/A"
        return _fmt(float(row.iloc[0]["value"]))

    # --- Interpretation helpers ---
    no_cost_row = (
        scenarios_df[scenarios_df["scenario"] == "no_cost"]
        if not scenarios_df.empty
        else pd.DataFrame()
    )
    base_cost_row = (
        scenarios_df[scenarios_df["scenario"] == "base_cost"]
        if not scenarios_df.empty
        else pd.DataFrame()
    )

    no_cost_return = (
        float(no_cost_row.iloc[0]["total_return"]) if not no_cost_row.empty else None
    )
    base_cost_return = (
        float(base_cost_row.iloc[0]["total_return"])
        if not base_cost_row.empty
        else None
    )

    cost_improved = (
        no_cost_return is not None
        and base_cost_return is not None
        and no_cost_return > base_cost_return + 0.05
    )
    cost_analysis = (
        f"no_cost={_fmt(no_cost_return)}, base_cost={_fmt(base_cost_return)} → "
        + (
            "yes, cost removal materially improved return"
            if cost_improved
            else "minimal cost impact or factor itself is weak"
        )
    )

    # Rebalance turnover trend
    rebalance_turnover = {}
    if not rebalance_df.empty:
        for _, row in rebalance_df.iterrows():
            rebalance_turnover[row["rebalance_frequency"]] = float(row["turnover"])
    rebalance_summary_lines = [
        f"  - {freq}: turnover={_fmt(rebalance_turnover.get(freq, float('nan')))}"
        for freq in ["daily", "weekly", "monthly"]
    ]

    # Rebalance sensitivity: did lower freq improve sharpe?
    rebalance_sharpe = {}
    if not rebalance_df.empty:
        for _, row in rebalance_df.iterrows():
            rebalance_sharpe[row["rebalance_frequency"]] = float(row["sharpe"])

    # Buffer rule: is it enabled?
    rules = backtest_cfg.get("rules", {}) or {}
    buffer_enabled = bool(rules.get("hold_until_drop", False))
    drop_rank_buffer = int(rules.get("drop_rank_buffer", 30))

    # vectorbt status
    vbt_row = (
        engine_df[engine_df["engine"] == "vectorbt"]
        if not engine_df.empty
        else pd.DataFrame()
    )
    vbt_status = "unavailable"
    if not vbt_row.empty:
        notes = str(vbt_row.iloc[0].get("notes", ""))
        if "unavailable" not in notes.lower():
            vbt_status = "available"

    # Failure diagnosis
    if no_cost_return is not None and no_cost_return > -0.1:
        failure_diagnosis = (
            "Cost and turnover are the dominant contributors to the negative result. "
            "The gross alpha signal may exist but is overwhelmed by transaction costs "
            "at daily rebalance frequency."
        )
    elif no_cost_return is not None and no_cost_return < -0.5:
        failure_diagnosis = (
            "Even without transaction costs the strategy underperforms significantly. "
            "The historical_price_volume factor appears to lack positive alpha "
            "in this universe/period. Factor weakness is the primary cause of failure."
        )
    else:
        failure_diagnosis = (
            "Both factor weakness and transaction costs contribute to the poor result. "
            "Further investigation with different factors or time periods is needed."
        )

    proceed_to_week4 = (
        "Recommend proceeding to Week 4 if no_cost Sharpe > 0 or if weekly/monthly "
        "rebalance shows materially improved performance. Otherwise, consider "
        "alternative factor selection first."
    )

    lines = [
        "# Backtest Realism Report",
        "",
        "---",
        "",
        "> **研究聲明 Research Disclaimer**",
        "> - research backtest only · not investment advice · not production-ready",
        f"> - OHLCV coverage: {ticker_count} / 1090 tickers (yfinance fallback)",
        "> - valuation snapshot factors excluded (pb_inverse / pe_inverse / dividend_yield / latest_snapshot_mixed)",  # noqa: E501
        "",
        "---",
        "",
        "## 1. Run Metadata",
        "",
        f"- generated_at: {generated_at}",
        f"- config_path: {config_path}",
        "- pipeline: run_backtest_diagnostics.py",
        f"- factor_name: {factor_name}",
        f"- top_n: {top_n}",
        f"- ohlcv_ticker_count: {ticker_count}",
        "",
        "## 2. Purpose",
        "",
        "Week 3 baseline backtest produced total_return = -0.979297, Sharpe = -3.065453.",  # noqa: E501
        "This report diagnoses whether the poor result is caused by:",
        "1. Factor alpha weakness (signal itself has no predictive power)",
        "2. Transaction cost drag (daily rebalance × high costs)",
        "3. Excessive turnover overwhelming any gross return",
        "4. Combination of the above",
        "",
        "## 3. Baseline Strategy Recap",
        "",
        "- factor: historical_price_volume (equal-weight composite of momentum_60d, low_volatility_20d, volume_ratio_5d_60d)",  # noqa: E501
        "- selection: Top 20 by composite score",
        "- weighting: equal weight",
        "- execution: T+1 (signal at date T, trade at T+1)",
        "- baseline result: total_return=-0.979297, annualized_return=-0.395379, sharpe=-3.065453, max_drawdown=-0.979297, turnover=0.596807",  # noqa: E501
        "",
        "## 4. Universe Coverage",
        "",
        f"- actual OHLCV tickers: {ticker_count} / 1090 TWSE listed",
        "- data source: yfinance fallback (TWSE OpenAPI OHLCV not fully available)",
        "- survivorship bias caveat: delisted tickers may be underrepresented",
        "",
        "## 5. Selected Factor",
        "",
        f"- factor_name: {factor_name}",
        "- eligible: yes (historical price-volume, no snapshot valuation component)",
        "- excluded factors: pb_inverse, pe_inverse, dividend_yield, latest_snapshot_mixed",  # noqa: E501
        "",
        "## 6. Rebalance Frequency Results",
        "",
    ]

    if not rebalance_df.empty:
        lines.append(
            "| frequency | rebalance_count | total_return | annualized_return | sharpe | max_drawdown | turnover |"  # noqa: E501
        )
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in rebalance_df.iterrows():
            lines.append(
                f"| {row['rebalance_frequency']} | {int(row['rebalance_count'])} "
                f"| {_fmt(row['total_return'])} | {_fmt(row['annualized_return'])} "
                f"| {_fmt(row['sharpe'])} | {_fmt(row['max_drawdown'])} | {_fmt(row['turnover'])} |"  # noqa: E501
            )
    else:
        lines.append("*(no rebalance sensitivity data)*")

    lines += (
        [
            "",
            "**Turnover by frequency:**",
        ]
        + rebalance_summary_lines
        + [
            "",
            "## 7. Cost Sensitivity Results",
            "",
        ]
    )

    if not scenarios_df.empty:
        lines.append(
            "| scenario | total_return | annualized_return | sharpe | max_drawdown | turnover | cost_drag |"  # noqa: E501
        )
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in scenarios_df.iterrows():
            lines.append(
                f"| {row['scenario']} | {_fmt(row['total_return'])} "
                f"| {_fmt(row['annualized_return'])} | {_fmt(row['sharpe'])} "
                f"| {_fmt(row['max_drawdown'])} | {_fmt(row['turnover'])} "
                f"| {_fmt(row['cost_drag'])} |"
            )
        lines += ["", f"**Analysis**: {cost_analysis}"]
    else:
        lines.append("*(no cost sensitivity data)*")

    lines += [
        "",
        "## 8. Top N Sensitivity Results",
        "",
    ]

    if not topn_df.empty:
        lines.append(
            "| top_n | total_return | annualized_return | sharpe | max_drawdown | turnover | notes |"  # noqa: E501
        )
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in topn_df.iterrows():
            lines.append(
                f"| {int(row['top_n'])} | {_fmt(row['total_return'])} "
                f"| {_fmt(row['annualized_return'])} | {_fmt(row['sharpe'])} "
                f"| {_fmt(row['max_drawdown'])} | {_fmt(row['turnover'])} "
                f"| {row.get('notes', '')} |"
            )
    else:
        lines.append("*(no top N sensitivity data)*")

    lines += [
        "",
        "## 9. Turnover Diagnostics",
        "",
        f"- average_daily_turnover: {_diag('average_daily_turnover')}",
        f"- median_daily_turnover: {_diag('median_daily_turnover')}",
        f"- max_daily_turnover: {_diag('max_daily_turnover')}",
        f"- annualized_turnover_estimate: {_diag('annualized_turnover_estimate')}",
        f"- avg_holdings: {_diag('avg_holdings')}",
        f"- avg_buys_per_rebalance: {_diag('avg_buys_per_rebalance')}",
        f"- avg_sells_per_rebalance: {_diag('avg_sells_per_rebalance')}",
        f"- estimated_cost_drag: {_diag('estimated_cost_drag')}",
        "",
        "## 10. Buffer Rule Impact",
        "",
        f"- hold_until_drop: {buffer_enabled}",
        f"- drop_rank_buffer: {drop_rank_buffer}",
        "- Status: buffer rule enabled in config. Compare turnover vs non-buffer run for full impact analysis.",  # noqa: E501
        "",
        "## 11. Engine Status / Cross-Check",
        "",
        f"- vectorbt_status: {vbt_status}",
    ]

    if not engine_df.empty:
        lines.append("")
        lines.append(
            "| engine | total_return | sharpe | max_drawdown | turnover | notes |"
        )
        lines.append("|---|---|---|---|---|---|")
        for _, row in engine_df.iterrows():
            lines.append(
                f"| {row['engine']} | {_fmt(row['total_return'])} "
                f"| {_fmt(row['sharpe'])} | {_fmt(row['max_drawdown'])} "
                f"| {_fmt(row['turnover'])} | {row.get('notes', '')} |"
            )

    lines += [
        "",
        "## 12. Interpretation",
        "",
        "### Q1: Did no-cost performance improve materially?",
        f"- {cost_analysis}",
        "",
        "### Q2: Did weekly/monthly rebalance reduce turnover?",
    ]

    if rebalance_turnover:
        daily_t = rebalance_turnover.get("daily", float("nan"))
        weekly_t = rebalance_turnover.get("weekly", float("nan"))
        monthly_t = rebalance_turnover.get("monthly", float("nan"))
        lines += [
            f"- daily={_fmt(daily_t)}, weekly={_fmt(weekly_t)}, monthly={_fmt(monthly_t)}",  # noqa: E501
            "- "
            + (
                "Yes, lower frequency significantly reduces turnover."
                if not np.isnan(weekly_t)
                and not np.isnan(daily_t)
                and weekly_t < daily_t * 0.5
                else "Turnover reduction is present but modest, or data insufficient."
            ),
        ]
    else:
        lines.append("- *(insufficient data)*")

    lines += [
        "",
        "### Q3: Did lower turnover improve Sharpe or drawdown?",
    ]
    if rebalance_sharpe:
        lines += [
            "- Sharpe by frequency: "
            + ", ".join(
                f"{f}={_fmt(rebalance_sharpe.get(f, float('nan')))}"
                for f in ["daily", "weekly", "monthly"]
            ),
        ]
    else:
        lines.append("- *(insufficient data)*")

    lines += [
        "",
        "### Q4: Is the strategy failing due to factor weakness or cost/turnover?",
        f"- {failure_diagnosis}",
        "",
        "### Q5: Should this factor proceed to Week 4 research?",
        f"- {proceed_to_week4}",
        "",
        "## 13. Limitations",
        "",
        "- OHLCV coverage is 100 / 1090 (9.2%) — small universe introduces concentration risk",  # noqa: E501
        "- yfinance fallback data may differ from official TWSE closing prices",
        "- Valuation snapshot factors (pb_inverse, pe_inverse, dividend_yield) excluded — composite is price-volume only",  # noqa: E501
        "- Backtest period and survivorship bias not fully controlled",
        "- Buffer rule reduces turnover but introduces path-dependency (results depend on history)",  # noqa: E501
        "- All cost assumptions are simplified (no market impact model)",
        "- vectorbt engine status: " + vbt_status,
        "",
        "## 14. Recommended Next Step",
        "",
        "1. If no_cost Sharpe > 0: proceed to Week 4 with weekly/monthly rebalance and buffer rule",  # noqa: E501
        "2. If no_cost Sharpe < 0: investigate alternative factor construction or factor selection criteria",  # noqa: E501
        "3. Consider expanding OHLCV universe beyond 100 tickers for more robust results",  # noqa: E501
        "4. Add sub-period analysis (2018–2020 vs 2021–2024) to test time-period stability",  # noqa: E501
        "",
        "## 15. Generated Artifacts",
        "",
    ]

    for key in [
        "rebalance_calendar",
        "backtest_scenarios",
        "topn_sensitivity",
        "rebalance_sensitivity",
        "backtest_turnover_diagnostics",
        "backtest_engine_comparison",
        "backtest_realism_report",
    ]:
        path = output_paths.get(key, "N/A")
        lines.append(f"- {key}: {path}")

    return "\n".join(lines)

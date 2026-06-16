from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from run_data_pipeline import resolve_path
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.data.manifest import append_manifest_entries, build_manifest_entry
from twse_factor_lab.data.parquet_store import ParquetStore
from twse_factor_lab.portfolio.selection import build_topn_positions
from twse_factor_lab.portfolio.weights import build_equal_weight_portfolio
from twse_factor_lab.selection.scoreboard import (
    build_factor_scoreboard,
    select_backtest_factor,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _backtest_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("backtest", {}) or {}


def _path_map(config_path: str | Path, paths: dict[str, str]) -> dict[str, Path]:
    defaults = {
        "factor_scoreboard": "data/processed/factor_scoreboard.parquet",
        "selected_factor_scores": "data/processed/selected_factor_scores.parquet",
        "topn_positions": "data/processed/topn_positions.parquet",
        "portfolio_weights": "data/processed/portfolio_weights.parquet",
        "backtest_results": "data/processed/backtest_results.parquet",
        "backtest_metrics": "data/processed/backtest_metrics.parquet",
        "composite_factor_report": "reports/composite_factor_report.md",
        "backtest_report": "reports/backtest_report.md",
    }
    merged = dict(defaults | paths)
    return {name: resolve_path(config_path, value) for name, value in merged.items()}


def check_backtest_readiness(
    close_matrix: pd.DataFrame,
    *,
    min_ticker_count: int,
) -> None:
    ticker_count = int(close_matrix.shape[1])
    if ticker_count < min_ticker_count:
        raise RuntimeError(
            "OHLCV coverage is not suitable for baseline Top N backtest: "
            f"actual_ohlcv_ticker_count={ticker_count}, "
            f"min_ticker_count={min_ticker_count}"
        )


def _cost_model_from_config(backtest_config: dict[str, Any]) -> CostModel:
    fees = backtest_config.get("fees", {}) or {}
    return CostModel(
        buy_fee_rate=float(fees.get("buy_fee_rate", 0.001425)),
        sell_fee_rate=float(fees.get("sell_fee_rate", 0.001425)),
        transaction_tax_rate=float(fees.get("transaction_tax_rate", 0.003)),
        slippage_rate=float(fees.get("slippage_rate", 0.001)),
    )


def build_composite_factor_report(
    *,
    config_path: str | Path,
    selected_factor: str,
    scoreboard: pd.DataFrame,
    close_matrix: pd.DataFrame,
    output_paths: dict[str, Path],
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    ticker_count = int(close_matrix.shape[1])
    selected = scoreboard[scoreboard["factor"] == selected_factor]
    selected_row = selected.iloc[0] if not selected.empty else None
    top_bottom_spread = (
        selected_row.top_bottom_spread if selected_row is not None else "N/A"
    )
    avg_turnover = selected_row.avg_turnover if selected_row is not None else "N/A"
    monotonicity_pass = (
        selected_row.monotonicity_pass if selected_row is not None else "N/A"
    )
    lines = [
        "# Composite Factor Report",
        "",
        "## Run Metadata",
        "",
        f"- generated_at: {generated_at}",
        f"- config_path: {Path(config_path)}",
        "- pipeline_name: run_backtest.py",
        "",
        "## Factor Scoreboard Summary",
        "",
    ]
    for row in scoreboard.itertuples(index=False):
        lines.append(
            "- "
            f"{row.factor}: horizon={row.best_horizon}, "
            f"ir={row.ir}, candidate={row.is_backtest_candidate}, notes={row.notes}"
        )
    lines.extend(
        [
            "",
            "## Selected Factor",
            "",
            f"- selected_factor: {selected_factor}",
            "",
            "## Selection Rationale",
            "",
            (
                "- historical_price_volume is the Week 3 default because it is an "
                "eligible historical composite and the strongest current baseline "
                "candidate from Week 2.5 analysis."
            ),
            "",
            "## Snapshot Factor Exclusion",
            "",
            (
                "- pb_inverse, pe_inverse, dividend_yield, and latest_snapshot_mixed "
                "are excluded because valuation data is latest snapshot only."
            ),
            "",
            "## IC / IR Summary For Selected Factor",
            "",
        ]
    )
    if selected_row is not None:
        lines.append(
            f"- IC mean={selected_row.ic_mean}, IC std={selected_row.ic_std}, "
            f"IR={selected_row.ir}, best_horizon={selected_row.best_horizon}"
        )
    lines.extend(
        [
            "",
            "## Quantile Return Summary",
            "",
            f"- top_bottom_spread: {top_bottom_spread}",
            "",
            "## Turnover Summary",
            "",
            f"- avg_turnover: {avg_turnover}",
            "",
            "## Monotonicity Warning",
            "",
            f"- monotonicity_pass: {monotonicity_pass}",
            "",
            "## Universe / OHLCV Coverage",
            "",
            f"- actual_ohlcv_ticker_count: {ticker_count}",
            "",
            "## Limitations",
            "",
            "- This is a research factor selection report, not production readiness.",
            "- OHLCV coverage may be below the full TWSE universe.",
            "- Factor monotonicity checks are imperfect.",
            "",
            "## Generated Artifacts",
            "",
            f"- factor_scoreboard: {output_paths['factor_scoreboard']}",
            f"- selected_factor_scores: {output_paths['selected_factor_scores']}",
        ]
    )
    return "\n".join(lines)


def build_backtest_report(
    *,
    config_path: str | Path,
    selected_factor: str,
    close_matrix: pd.DataFrame,
    topn_positions: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    metrics: pd.DataFrame,
    output_paths: dict[str, Path],
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    metric = metrics.iloc[0]
    lines = [
        "# Backtest Report",
        "",
        "## Run Metadata",
        "",
        f"- generated_at: {generated_at}",
        f"- config_path: {Path(config_path)}",
        "- pipeline_name: run_backtest.py",
        "",
        "## Scope",
        "",
        "- research backtest only",
        "- not live trading",
        "- not investment advice",
        "",
        "## Backtest Readiness",
        "",
        f"- actual_ohlcv_ticker_count: {close_matrix.shape[1]}",
        "- readiness_status: passed",
        "",
        "## Inputs",
        "",
        f"- close_matrix: {output_paths['close_matrix']}",
        f"- factors_composite: {output_paths['factors_composite']}",
        f"- topn_positions: {output_paths['topn_positions']}",
        f"- portfolio_weights: {output_paths['portfolio_weights']}",
        "",
        "## Selected Factor",
        "",
        f"- selected_factor: {selected_factor}",
        "",
        "## Portfolio Construction",
        "",
        f"- selected_rows: {int(topn_positions['selected'].sum())}",
        f"- top_n: {metric.top_n}",
        "- weighting_rule: equal weight",
        "",
        "## Execution Lag",
        "",
        f"- execution_lag_days: {portfolio_weights['execution_lag_days'].max()}",
        "- factor at date T trades on execution_date T+1",
        "",
        "## Cost Model",
        "",
        f"- cost_model_summary: {metric.cost_model_summary}",
        "",
        "## Backtest Engine",
        "",
        f"- engine: {metric.engine}",
        "",
        "## Backtest Metrics",
        "",
    ]
    for column in metrics.columns:
        lines.append(f"- {column}: {metric[column]}")
    lines.extend(
        [
            "",
            "## Equity Curve / Drawdown Summary",
            "",
            f"- total_return: {metric.total_return}",
            f"- max_drawdown: {metric.max_drawdown}",
            "",
            "## Limitations",
            "",
            "- OHLCV ticker coverage and coverage ratio remain research limitations.",
            "- yfinance fallback data may differ from official TWSE data.",
            "- valuation snapshot factors excluded.",
            "- latest_snapshot_mixed excluded.",
            "- factor monotonicity not fully passed.",
            "- transaction costs are simplified assumptions.",
            "",
            "## Generated Artifacts",
            "",
        ]
    )
    for name in [
        "factor_scoreboard",
        "selected_factor_scores",
        "topn_positions",
        "portfolio_weights",
        "backtest_results",
        "backtest_metrics",
        "composite_factor_report",
        "backtest_report",
    ]:
        lines.append(f"- {name}: {output_paths[name]}")
    return "\n".join(lines)


def run_backtest(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    paths = _path_map(config_path, config["paths"])
    backtest_config = _backtest_config(config)
    store = ParquetStore()

    close_matrix = store.load(paths["close_matrix"])
    check_backtest_readiness(
        close_matrix,
        min_ticker_count=int(backtest_config.get("min_ticker_count", 100)),
    )
    factors_composite = store.load(paths["factors_composite"])
    ic_summary = store.load(paths["factor_ic_summary"])
    quantile_returns = store.load(paths["factor_quantile_returns"])
    turnover = store.load(paths["factor_turnover"])
    monotonicity = store.load(paths["factor_monotonicity"])

    scoreboard = build_factor_scoreboard(
        ic_summary=ic_summary,
        quantile_returns=quantile_returns,
        turnover=turnover,
        monotonicity=monotonicity,
        ohlcv_ticker_count=int(close_matrix.shape[1]),
    )
    selected_factor = select_backtest_factor(
        scoreboard,
        requested_factor=backtest_config.get("factor_name", "historical_price_volume"),
    )
    selected_scores = factors_composite[
        (factors_composite["composite_type"] == selected_factor)
        & (~factors_composite["is_snapshot_component_used"].astype(bool))
    ].copy()
    topn_positions = build_topn_positions(
        factors_composite,
        top_n=int(backtest_config.get("top_n", 20)),
        factor_name=selected_factor,
    )
    portfolio_weights = build_equal_weight_portfolio(
        topn_positions,
        execution_lag_days=int(backtest_config.get("execution_lag_days", 1)),
    )
    cost_model = _cost_model_from_config(backtest_config)
    vectorbt_config = backtest_config.get("vectorbt", {}) or {}
    backtest_results, backtest_metrics = run_weight_backtest(
        close_matrix=close_matrix,
        portfolio_weights=portfolio_weights,
        cost_model=cost_model,
        initial_cash=float(backtest_config.get("initial_cash", 1_000_000)),
        top_n=int(backtest_config.get("top_n", 20)),
        use_vectorbt=bool(vectorbt_config.get("use_vectorbt", True)),
        allow_fallback=bool(vectorbt_config.get("allow_fallback", True)),
    )

    store.save(scoreboard, paths["factor_scoreboard"])
    store.save(selected_scores, paths["selected_factor_scores"])
    store.save(topn_positions, paths["topn_positions"])
    store.save(portfolio_weights, paths["portfolio_weights"])
    store.save(backtest_results, paths["backtest_results"])
    store.save(backtest_metrics, paths["backtest_metrics"])

    composite_report = build_composite_factor_report(
        config_path=config_path,
        selected_factor=selected_factor,
        scoreboard=scoreboard,
        close_matrix=close_matrix,
        output_paths=paths,
    )
    paths["composite_factor_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["composite_factor_report"].write_text(composite_report, encoding="utf-8")

    backtest_report = build_backtest_report(
        config_path=config_path,
        selected_factor=selected_factor,
        close_matrix=close_matrix,
        topn_positions=topn_positions,
        portfolio_weights=portfolio_weights,
        metrics=backtest_metrics,
        output_paths=paths,
    )
    paths["backtest_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["backtest_report"].write_text(backtest_report, encoding="utf-8")

    created_at = datetime.now(UTC)
    schema_version = str(
        config.get("week2", {}).get("manifest_schema_version", "1.0.0")
    )
    entries = [
        build_manifest_entry(
            artifact_name=name,
            path=str(paths[name]),
            frame=frame,
            source_inputs=[str(paths["factors_composite"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="Week 3 backtest artifact",
        )
        for name, frame in {
            "factor_scoreboard": scoreboard,
            "selected_factor_scores": selected_scores,
            "topn_positions": topn_positions,
            "portfolio_weights": portfolio_weights,
            "backtest_results": backtest_results,
            "backtest_metrics": backtest_metrics,
        }.items()
    ]
    append_manifest_entries(entries, paths["manifest"])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Week 3 backtest pipeline.")
    parser.add_argument("--config", required=True, help="Path to strategy YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_backtest(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

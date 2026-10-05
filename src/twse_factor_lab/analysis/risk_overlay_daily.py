"""Historical MA60 breadth overlay with daily signal and T+1 execution."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from twse_factor_lab.analysis.risk_overlay import (
    BASE_MANIFEST,
    BREADTH_THRESHOLD,
    CLASSIFICATION_POLICY,
    EXPOSURE_HIGH,
    EXPOSURE_LOW,
    HISTORICAL_DIR,
    MA_WINDOW,
    NAMESPACE,
    PRICE_PATH,
    UNIVERSE_PATH,
    _file_sha256,
    _git_commit,
    _replay_events,
    _series_data,
    _write_json,
    calculate_ma60_breadth,
    classify_risk_overlay,
    discover_canonical_benchmark,
    exposure_for_breadth,
)
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.production.fundamental import (
    STRATEGY_FINGERPRINT,
    STRATEGY_ID,
    sha256_payload,
)
from twse_factor_lab.reporting.performance_adapter import (
    FrozenStrategyChanged,
    load_frozen_identity,
    load_repository_performance_data,
)
from twse_factor_lab.reporting.performance_metrics import calculate_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DAILY_OUTPUT_DIR = NAMESPACE / "risk-overlay-daily"
PREVIOUS_OUTPUT_DIR = NAMESPACE / "risk-overlay"
PREVIOUS_VALIDATION_PATH = PREVIOUS_OUTPUT_DIR / "risk_overlay_validation.json"
PREVIOUS_COMPARISON_PATH = PREVIOUS_OUTPUT_DIR / "risk_overlay_comparison.csv"
BASE_REBALANCES_PATH = HISTORICAL_DIR / "canonical_backtest_rebalances.parquet"
DAILY_OVERLAY_ID = "market_breadth_ma60_040_half_exposure_daily_v1"
DAILY_COMBINED_ID = (
    "fundamental_g2g3_top5_reb60_score_weighted_plus_breadth040_daily_v1"
)
HISTORICAL_START = pd.Timestamp("2021-01-04")
HISTORICAL_END = pd.Timestamp("2025-12-31")
EXPECTED_SESSIONS = 1212
POLICY_VERSION = "risk-overlay-classification-policy-v1"


class DailyRiskOverlayError(ValueError):
    """Raised when the daily overlay contract cannot be satisfied."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise DailyRiskOverlayError(reason)


def _metric_value(metrics: Any, name: str) -> Any:
    return metrics.metrics[name].value


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    return {name: item.value for name, item in metrics.metrics.items()}


def _policy_payload(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not (root / PREVIOUS_VALIDATION_PATH).exists():
        raise DailyRiskOverlayError("IMMUTABLE_PREVIOUS_OVERLAY_MISSING")
    path = root / PREVIOUS_VALIDATION_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("overlay_id") == "market_breadth_ma60_040_half_exposure_v1",
        "IMMUTABLE_PREVIOUS_OVERLAY_CHANGED",
    )
    _require(
        payload.get("classification") == "ADVERSE",
        "IMMUTABLE_PREVIOUS_OVERLAY_NOT_ADVERSE",
    )
    policy = payload.get("classification_policy")
    _require(policy == CLASSIFICATION_POLICY, "CLASSIFICATION_POLICY_GAP")
    policy_hash = sha256_payload(policy)
    return (
        payload,
        {
            "source": str(PREVIOUS_VALIDATION_PATH).replace("\\", "/"),
            "version": POLICY_VERSION,
            "hash": policy_hash,
            "rules": policy,
        },
        _file_sha256(path),
    )


def _load_cost(manifest: dict[str, Any]) -> CostModel:
    try:
        payload = manifest["cost_model"]
        cost = CostModel(
            buy_fee_rate=float(payload["buy_fee_rate"]),
            sell_fee_rate=float(payload["sell_fee_rate"]),
            transaction_tax_rate=float(payload["transaction_tax_rate"]),
            slippage_rate=float(payload["slippage_rate"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DailyRiskOverlayError("COST_MODEL_UNAVAILABLE") from exc
    expected = manifest.get("cost_model_hash")
    if expected and expected != sha256_payload(cost.summary()):
        raise DailyRiskOverlayError("COST_MODEL_CHANGED")
    return cost


def _normalise_rebalances(events: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "execution_date", "ticker", "target_weight"}
    missing = required - set(events.columns)
    if missing:
        raise DailyRiskOverlayError(
            "REBALANCE_COLUMNS_MISSING:" + ",".join(sorted(missing))
        )
    result = events.loc[:, sorted(required)].copy()
    for column in ("signal_date", "execution_date"):
        result[column] = pd.to_datetime(result[column]).dt.normalize()
    result["ticker"] = result["ticker"].astype(str)
    result["target_weight"] = pd.to_numeric(result["target_weight"], errors="raise")
    return result.sort_values(["execution_date", "ticker"], kind="stable")


def _relative_weights_preserved(base: pd.Series, scaled: pd.Series) -> bool:
    base = base[base.ne(0.0)]
    scaled = scaled[scaled.ne(0.0)]
    if base.empty and scaled.empty:
        return True
    if set(base.index) != set(scaled.index) or base.sum() == 0 or scaled.sum() == 0:
        return False
    left = (base / base.sum()).sort_index()
    right = (scaled / scaled.sum()).sort_index()
    return bool(np.allclose(left.to_numpy(), right.to_numpy(), atol=1e-12))


def _daily_targets(
    rebalances: pd.DataFrame,
    breadth: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = _normalise_rebalances(rebalances)
    sessions = pd.DatetimeIndex(sessions).normalize().sort_values()
    columns = pd.Index(sorted(events["ticker"].unique()))
    base_targets = (
        events.pivot_table(
            index="execution_date",
            columns="ticker",
            values="target_weight",
            aggfunc="last",
        )
        .reindex(columns=columns)
        .fillna(0.0)
        .reindex(sessions)
        .ffill()
        .fillna(0.0)
    )
    breadth_frame = breadth.copy()
    breadth_frame["trade_date"] = pd.to_datetime(
        breadth_frame["trade_date"]
    ).dt.normalize()
    breadth_frame = breadth_frame.set_index("trade_date").reindex(sessions)
    desired = breadth_frame["breadth"].map(exposure_for_breadth).astype(float)
    desired.index = sessions
    executed: list[float] = []
    previous: list[float] = []
    signal_dates: list[pd.Timestamp | pd.NaT] = []
    signal_values: list[float] = []
    signal_valid: list[bool] = []
    current = EXPOSURE_LOW
    for position, _date in enumerate(sessions):
        previous.append(float(current))
        if position == 0:
            signal_dates.append(pd.NaT)
            signal_values.append(np.nan)
            signal_valid.append(False)
        else:
            signal_date = sessions[position - 1]
            signal_dates.append(signal_date)
            value = breadth_frame.loc[signal_date, "breadth"]
            signal_values.append(float(value) if pd.notna(value) else np.nan)
            signal_valid.append(bool(pd.notna(value)))
            desired_state = float(desired.loc[signal_date])
            if desired_state != current:
                current = desired_state
        executed.append(float(current))
    exposure = pd.Series(executed, index=sessions, dtype="float64")
    weights = base_targets.mul(exposure, axis=0)
    desired_at_execution = [EXPOSURE_LOW] + desired.iloc[:-1].tolist()
    rebalance_dates = set(events["execution_date"])
    base_previous = base_targets.shift(1).fillna(0.0)
    audit = pd.DataFrame(
        {
            "trade_date": sessions,
            "is_rebalance": [date in rebalance_dates for date in sessions],
            "breadth_signal_date": signal_dates,
            "breadth": signal_values,
            "breadth_signal": [
                bool(value > BREADTH_THRESHOLD) if math.isfinite(value) else False
                for value in signal_values
            ],
            "signal_valid": signal_valid,
            "desired_exposure": desired_at_execution,
            "previous_exposure": previous,
            "new_exposure": executed,
            "exposure_transition": [
                (
                    "NO_CHANGE"
                    if previous[index] == executed[index]
                    else f"{previous[index]:.1f}_TO_{executed[index]:.1f}"
                )
                for index in range(len(sessions))
            ],
        }
    )
    audit["selected_tickers_changed"] = [
        bool(
            set(base_targets.iloc[index][lambda row: row.ne(0.0)].index)
            != set(base_previous.iloc[index][lambda row: row.ne(0.0)].index)
        )
        if audit.iloc[index]["is_rebalance"]
        else False
        for index in range(len(sessions))
    ]
    audit["relative_weights_preserved"] = [
        _relative_weights_preserved(base_targets.iloc[index], weights.iloc[index])
        for index in range(len(sessions))
    ]
    history = breadth.copy()
    history["trade_date"] = pd.to_datetime(history["trade_date"]).dt.normalize()
    history = history.sort_values("trade_date", kind="stable").reset_index(drop=True)
    history["desired_exposure"] = history["breadth"].map(exposure_for_breadth)
    next_dates = list(sessions[1:]) + [pd.NaT]
    history["execution_date"] = history["trade_date"].map(
        dict(zip(sessions, next_dates, strict=True))
    )
    history["signal_valid"] = history["breadth"].notna()
    history["threshold"] = BREADTH_THRESHOLD
    return weights, history, audit


def _daily_events(weights: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    frame = weights.rename_axis("execution_date").stack().rename("target_weight")
    result = frame.reset_index().rename(columns={"level_1": "ticker"})
    signal_dates = {
        date: (sessions[index - 1] if index else pd.NaT)
        for index, date in enumerate(sessions)
    }
    result["signal_date"] = result["execution_date"].map(signal_dates)
    return result.loc[:, ["signal_date", "execution_date", "ticker", "target_weight"]]


def _transaction_costs(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    return frame.groupby(pd.to_datetime(frame["date"]).dt.normalize())[
        "total_cost"
    ].sum()


def _add_trade_audit(
    audit: pd.DataFrame,
    base_replay: dict[str, Any],
    daily_replay: dict[str, Any],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    result = audit.copy()
    base_returns = base_replay["returns"].set_index("date")
    daily_returns = daily_replay["returns"].set_index("date")
    base_costs = _transaction_costs(base_replay["transactions"])
    daily_costs = _transaction_costs(daily_replay["transactions"])
    result["turnover"] = (
        daily_returns["turnover"].reindex(result["trade_date"]).fillna(0.0).to_numpy()
    )
    result["base_turnover"] = (
        base_returns["turnover"].reindex(result["trade_date"]).fillna(0.0).to_numpy()
    )
    result["transaction_cost"] = (
        daily_costs.reindex(result["trade_date"]).fillna(0.0).to_numpy()
    )
    result["base_transaction_cost"] = (
        base_costs.reindex(result["trade_date"]).fillna(0.0).to_numpy()
    )
    result["overlay_transaction_cost"] = (
        result["transaction_cost"] - result["base_transaction_cost"]
    )
    result["overlay_turnover"] = [
        float(
            abs(weights.iloc[index] * (row.new_exposure - row.previous_exposure)).sum()
        )
        if row.exposure_transition != "NO_CHANGE"
        else 0.0
        for index, row in enumerate(result.itertuples(index=False))
    ]
    for row in result.loc[~result["is_rebalance"]].itertuples():
        _require(not row.selected_tickers_changed, "NON_REBALANCE_SELECTION_CHANGE")
        _require(row.relative_weights_preserved, "NON_REBALANCE_RELATIVE_WEIGHT_CHANGE")
        if row.turnover > 1e-12:
            _require(
                row.exposure_transition != "NO_CHANGE",
                "NON_REBALANCE_TURNOVER_NOT_EXPOSURE",
            )
    return result


def _risk_off_durations(exposure: pd.Series) -> list[int]:
    values = exposure.to_numpy()
    durations: list[int] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value == EXPOSURE_LOW and start is None:
            start = index
        if start is not None and (value != EXPOSURE_LOW or index == len(values) - 1):
            end = index if value == EXPOSURE_LOW else index - 1
            durations.append(end - start + 1)
            start = None
    return durations


def _annual_returns(returns: pd.Series) -> dict[str, float]:
    frame = returns.to_frame("return")
    frame["year"] = frame.index.year
    return {
        f"{int(year)}_return": float((1.0 + group["return"]).prod() - 1.0)
        for year, group in frame.groupby("year")
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _drawdown(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def _metric_comparison(
    base_metrics: Any,
    previous: dict[str, Any],
    daily_metrics: Any,
    previous_comparison: pd.DataFrame,
    base_returns: pd.Series,
    daily_returns: pd.Series,
) -> pd.DataFrame:
    names = (
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "max_drawdown_start",
        "max_drawdown_trough",
        "max_drawdown_recovery",
        "max_drawdown_duration",
        "annualized_volatility",
        "turnover",
        "transaction_cost",
    )
    rows: list[dict[str, Any]] = []
    previous_metrics = previous["overlay"]["metrics"]
    for name in names:
        base = _metric_value(base_metrics, name)
        reb60 = previous_metrics.get(name)
        daily = _metric_value(daily_metrics, name)
        rows.append(
            {
                "metric": name,
                "base": base,
                "reb60": reb60,
                "daily": daily,
                "daily_minus_base": (
                    float(daily) - float(base)
                    if isinstance(base, (int, float))
                    and isinstance(daily, (int, float))
                    else np.nan
                ),
                "daily_minus_reb60": (
                    float(daily) - float(reb60)
                    if isinstance(reb60, (int, float))
                    and isinstance(daily, (int, float))
                    else np.nan
                ),
            }
        )
    previous_annual = (
        previous_comparison.set_index("metric")["overlay"].to_dict()
        if not previous_comparison.empty
        else {}
    )
    annual = sorted(
        set(_annual_returns(base_returns)) | set(_annual_returns(daily_returns))
    )
    for name in annual:
        base = _annual_returns(base_returns).get(name)
        daily = _annual_returns(daily_returns).get(name)
        reb60 = _float_or_none(previous_annual.get(name))
        rows.append(
            {
                "metric": name,
                "base": base,
                "reb60": reb60,
                "daily": daily,
                "daily_minus_base": daily - base
                if base is not None and daily is not None
                else np.nan,
                "daily_minus_reb60": daily - reb60
                if reb60 is not None and daily is not None
                else np.nan,
            }
        )
    for label, start, end in (("2024_2025_max_drawdown", "2024-01-01", "2025-12-31"),):
        base_dd = float(_drawdown(base_returns.loc[start:end]).min())
        daily_dd = float(_drawdown(daily_returns.loc[start:end]).min())
        reb60_dd = _float_or_none(previous_annual.get(label))
        rows.append(
            {
                "metric": label,
                "base": base_dd,
                "reb60": reb60_dd,
                "daily": daily_dd,
                "daily_minus_base": daily_dd - base_dd,
                "daily_minus_reb60": daily_dd - reb60_dd
                if reb60_dd is not None
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _drawdown_audit(
    base_replay: dict[str, Any],
    daily_replay: dict[str, Any],
    audit: pd.DataFrame,
) -> pd.DataFrame:
    base = base_replay["returns"].set_index("date")
    daily = daily_replay["returns"].set_index("date")
    result = audit.loc[audit["trade_date"].between("2024-01-01", "2025-12-31")].copy()
    result["base_equity"] = (
        base["portfolio_value"].reindex(result["trade_date"]).to_numpy()
    )
    result["overlay_equity"] = (
        daily["portfolio_value"].reindex(result["trade_date"]).to_numpy()
    )
    result["base_drawdown"] = (
        _drawdown(base["daily_return"]).reindex(result["trade_date"]).to_numpy()
    )
    result["overlay_drawdown"] = (
        _drawdown(daily["daily_return"]).reindex(result["trade_date"]).to_numpy()
    )
    return result.loc[
        :,
        [
            "trade_date",
            "base_equity",
            "overlay_equity",
            "base_drawdown",
            "overlay_drawdown",
            "breadth",
            "breadth_signal",
            "breadth_signal_date",
            "new_exposure",
            "exposure_transition",
        ],
    ].rename(columns={"trade_date": "date", "new_exposure": "executed_exposure"})


def _write_charts(
    output: Path,
    base_replay: dict[str, Any],
    daily_replay: dict[str, Any],
    audit: pd.DataFrame,
    previous_comparison: pd.DataFrame,
) -> list[str]:
    chart_dir = output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = base_replay["returns"].set_index("date")
    daily = daily_replay["returns"].set_index("date")
    base_return = base["daily_return"]
    daily_return = daily["daily_return"]
    paths = [
        chart_dir / "01_base_vs_daily_overlay_cumulative_return.png",
        chart_dir / "02_base_vs_daily_overlay_drawdown.png",
        chart_dir / "03_market_breadth_and_daily_exposure.png",
        chart_dir / "04_annual_return_comparison.png",
        chart_dir / "05_2024_2025_drawdown_exposure_detail.png",
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(base.index, (1 + base_return).cumprod() - 1, label="BASE")
    plt.plot(daily.index, (1 + daily_return).cumprod() - 1, label="DAILY/T+1")
    plt.title("BASE vs DAILY/T+1 Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths[0], dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(base.index, _drawdown(base_return), label="BASE")
    plt.plot(daily.index, _drawdown(daily_return), label="DAILY/T+1")
    plt.title("BASE vs DAILY/T+1 Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths[1], dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    axis = plt.gca()
    axis.plot(audit["trade_date"], audit["breadth"], label="Breadth")
    axis.axhline(BREADTH_THRESHOLD, color="black", linestyle="--", label="0.40")
    second = axis.twinx()
    second.step(
        audit["trade_date"],
        audit["new_exposure"],
        where="post",
        color="tab:red",
        label="Exposure",
    )
    axis.set_title("Market Breadth and Daily/T+1 Exposure")
    axis.set_ylabel("Breadth")
    second.set_ylabel("Exposure")
    plt.tight_layout()
    plt.savefig(paths[2], dpi=140)
    plt.close()

    years = sorted(set(base.index.year) | set(daily.index.year))
    base_annual = _annual_returns(base_return)
    daily_annual = _annual_returns(daily_return)
    reb60_map = previous_comparison.set_index("metric")["reb60"].to_dict()
    x = np.arange(len(years))
    width = 0.27
    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width,
        [base_annual.get(f"{year}_return", np.nan) for year in years],
        width,
        label="BASE",
    )
    plt.bar(
        x,
        [reb60_map.get(f"{year}_return", np.nan) for year in years],
        width,
        label="REB60",
    )
    plt.bar(
        x + width,
        [daily_annual.get(f"{year}_return", np.nan) for year in years],
        width,
        label="DAILY",
    )
    plt.xticks(x, years)
    plt.title("Annual Return Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths[3], dpi=140)
    plt.close()

    detail = audit.loc[audit["trade_date"].between("2024-01-01", "2025-12-31")].copy()
    detail["date"] = detail["trade_date"]
    detail["base_drawdown"] = (
        _drawdown(base_return).reindex(detail["trade_date"]).to_numpy()
    )
    detail["overlay_drawdown"] = (
        _drawdown(daily_return).reindex(detail["trade_date"]).to_numpy()
    )
    detail["executed_exposure"] = detail["new_exposure"]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(detail["date"], detail["overlay_drawdown"], label="DAILY Drawdown")
    axis.plot(detail["date"], detail["base_drawdown"], label="BASE Drawdown")
    axis.set_ylabel("Drawdown")
    second = axis.twinx()
    second.plot(detail["date"], detail["breadth"], color="tab:green", label="Breadth")
    second.step(
        detail["date"],
        detail["executed_exposure"],
        where="post",
        color="tab:red",
        label="Exposure",
    )
    axis.set_title("2024-2025 Drawdown and Exposure Detail")
    fig.tight_layout()
    fig.savefig(paths[4], dpi=140)
    plt.close(fig)
    return [
        str(path.relative_to(output.parent.parent)).replace("\\", "/") for path in paths
    ]


def _report(payload: dict[str, Any], comparison: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "UNAVAILABLE"
        return f"{value:.6f}" if isinstance(value, (int, float)) else str(value)

    table = [
        "| Metric | BASE | REB60 | DAILY | Daily-Base | Daily-REB60 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        table.append(
            f"| {row.metric} | {fmt(row.base)} | {fmt(row.reb60)} | "
            f"{fmt(row.daily)} | {fmt(row.daily_minus_base)} | "
            f"{fmt(row.daily_minus_reb60)} |"
        )
    drawdown = payload["drawdown_audit"]
    return "\n".join(
        [
            "# Daily Risk Overlay Validation Report",
            "",
            "## 1. Experiment Definition",
            "",
            f"Base Strategy = {STRATEGY_ID}",
            f"Daily Overlay = {DAILY_OVERLAY_ID}",
            "Only Changed Dimension = exposure execution frequency",
            "",
            "## 2. Integrity",
            "",
            f"Base Fingerprint = {payload['base_strategy_fingerprint']}",
            "Base Selection Changed = "
            f"{'YES' if not payload['selection_preserved'] else 'NO'}",
            "Base Ranking Changed = "
            f"{'YES' if not payload['selection_preserved'] else 'NO'}",
            "Relative Weights Changed = "
            f"{'YES' if not payload['relative_weights_preserved'] else 'NO'}",
            "",
            "## 3. Execution Semantics",
            "",
            "Breadth Frequency = DAILY",
            "Signal Timing = T close after legal close data is available",
            "Execution Timing = next legal trading session (T+1)",
            f"Threshold = {BREADTH_THRESHOLD:.2f}",
            f"Exposure = {EXPOSURE_HIGH:.1f} / {EXPOSURE_LOW:.1f}",
            "Stock Selection Frequency = REB60",
            "",
            "## 4. BASE vs DAILY",
            "",
            *table,
            "",
            "## 5. REB60 Overlay vs DAILY Overlay",
            "",
            "The immutable prior overlay remains `ADVERSE`; its artifacts "
            "were read-only inputs.",
            "",
            "## 6. Exposure Activity",
            "",
            f"Transition Count = {payload['exposure_activity']['transition_count']}",
            f"Risk-off Count = {payload['exposure_activity']['risk_off_count']}",
            f"Risk-on Count = {payload['exposure_activity']['risk_on_count']}",
            "Half Exposure Days = "
            f"{payload['exposure_activity']['half_exposure_days']}",
            "Average Exposure = "
            f"{payload['exposure_activity']['average_exposure']:.6f}",
            "Overlay Turnover = "
            f"{payload['exposure_activity']['overlay_turnover']:.6f}",
            "Overlay Transaction Cost = "
            f"{payload['exposure_activity']['overlay_transaction_cost']:.6f}",
            "",
            "## 7. 2024-2025 Drawdown",
            "",
            f"Base MDD = {payload['comparison']['base_2024_2025_mdd']:.6f}",
            f"REB60 Overlay MDD = {payload['comparison']['reb60_2024_2025_mdd']:.6f}",
            f"Daily Overlay MDD = {payload['comparison']['daily_2024_2025_mdd']:.6f}",
            f"First Risk-off Signal = {drawdown['first_risk_off_signal']}",
            f"First Risk-off Execution = {drawdown['first_risk_off_execution']}",
            f"MDD Improvement = {payload['comparison']['daily_mdd_improvement']:.6f}",
            "",
            "## 8. Return Sacrifice",
            "",
            f"CAGR Difference = {payload['comparison']['daily_cagr_difference']:.6f}",
            "Total Return Difference = "
            f"{payload['comparison']['daily_total_return_difference']:.6f}",
            "",
            "## 9. Classification",
            "",
            payload["classification"],
            "",
            "## 10. Final State",
            "",
            "Base Strategy = FROZEN / UNCHANGED",
            "Previous REB60 Overlay = REJECTED / ADVERSE",
            "Daily Overlay = RESEARCH_ONLY",
            f"Fresh OOS = {payload['fresh_oos_status']}",
            "Production Ready = NO",
            "",
            "Evidence Label = HISTORICAL_RISK_OVERLAY_EVALUATION",
            f"Benchmark = {payload['benchmark']['status']}",
        ]
    )


def evaluate_daily_risk_overlay(root: str | Path) -> dict[str, Any]:
    """Generate the additive daily/T+1 historical overlay evidence package."""
    root = Path(root).resolve()
    identity = load_frozen_identity(root)
    if (
        identity.strategy_id != STRATEGY_ID
        or identity.strategy_fingerprint != STRATEGY_FINGERPRINT
    ):
        raise FrozenStrategyChanged("FROZEN_STRATEGY_CHANGED")
    previous, policy, previous_hash = _policy_payload(root)
    source = load_repository_performance_data(root, identity)["BACKTEST"]
    if not source.available or source.returns is None:
        raise DailyRiskOverlayError("NO_IDENTITY_BOUND_CANONICAL_BACKTEST_SOURCE")
    manifest = json.loads((root / BASE_MANIFEST).read_text(encoding="utf-8"))
    _require(manifest.get("strategy_id") == STRATEGY_ID, "FROZEN_STRATEGY_CHANGED")
    _require(
        manifest.get("strategy_fingerprint") == STRATEGY_FINGERPRINT,
        "FROZEN_STRATEGY_CHANGED",
    )
    sessions = pd.DatetimeIndex(
        pd.to_datetime(source.returns.index).tz_localize(None)
    ).normalize()
    _require(sessions.min() == HISTORICAL_START, "HISTORICAL_WINDOW_CHANGED")
    _require(sessions.max() == HISTORICAL_END, "HISTORICAL_WINDOW_CHANGED")
    _require(len(sessions) == EXPECTED_SESSIONS, "HISTORICAL_SESSION_COUNT_CHANGED")
    close = pd.read_parquet(root / PRICE_PATH)
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    close.columns = close.columns.astype(str)
    universe = pd.read_parquet(root / UNIVERSE_PATH)
    rebalances = pd.read_parquet(root / BASE_REBALANCES_PATH)
    cost = _load_cost(manifest)
    breadth = calculate_ma60_breadth(
        close, universe, start=HISTORICAL_START, end=HISTORICAL_END
    )
    weights, breadth_history, audit = _daily_targets(rebalances, breadth, sessions)
    close_scope = close.loc[:, weights.columns].reindex(sessions)
    base_replay = _replay_events(
        close_scope, _normalise_rebalances(rebalances), cost, sessions
    )
    daily_events = _daily_events(weights, sessions)
    _require(
        bool(
            (
                daily_events["signal_date"].isna()
                | daily_events["signal_date"].lt(daily_events["execution_date"])
            ).all()
        ),
        "DAILY_SIGNAL_NOT_BEFORE_EXECUTION",
    )
    daily_replay = _replay_events(close_scope, daily_events, cost, sessions)
    audit = _add_trade_audit(audit, base_replay, daily_replay, weights)
    replay_returns = base_replay["returns"].set_index("date")["daily_return"]
    canonical_returns = source.returns.copy()
    canonical_returns.index = canonical_returns.index.tz_localize(None)
    base_replay_matches = bool(
        np.allclose(
            replay_returns.reindex(canonical_returns.index).to_numpy(),
            canonical_returns.to_numpy(),
            atol=1e-12,
            rtol=1e-12,
        )
    )
    _require(base_replay_matches, "BASE_REPLAY_MISMATCH")
    benchmark = discover_canonical_benchmark(root)
    benchmark_returns = (
        benchmark.get("returns") if benchmark["status"] == "AVAILABLE" else None
    )
    base_data = _series_data(
        "BACKTEST",
        base_replay,
        benchmark_returns,
    )
    daily_data = _series_data("BACKTEST", daily_replay, benchmark_returns)
    base_metrics = calculate_metrics(base_data)
    daily_metrics = calculate_metrics(daily_data)
    base_metrics_match = True
    for name in (
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "annualized_volatility",
        "turnover",
        "transaction_cost",
    ):
        expected = previous.get("base", {}).get("metrics", {}).get(name)
        actual = _metric_value(base_metrics, name)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            base_metrics_match = base_metrics_match and math.isclose(
                float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12
            )
    _require(base_metrics_match, "BASE_METRICS_MISMATCH")
    previous_comparison = (
        pd.read_csv(root / PREVIOUS_COMPARISON_PATH)
        if (root / PREVIOUS_COMPARISON_PATH).exists()
        else pd.DataFrame()
    )
    comparison = _metric_comparison(
        base_metrics,
        previous,
        daily_metrics,
        previous_comparison,
        replay_returns,
        daily_replay["returns"].set_index("date")["daily_return"],
    )
    policy_frame = comparison.loc[
        comparison["metric"].isin(
            ["cagr", "sharpe", "max_drawdown", "annualized_volatility"]
        ),
        ["metric", "base", "daily"],
    ].rename(columns={"daily": "overlay"})
    classification = classify_risk_overlay(policy_frame)
    audit_path = _drawdown_audit(base_replay, daily_replay, audit)
    drawdown_rows = audit_path
    risk_off_signals = drawdown_rows.loc[
        drawdown_rows["breadth"].le(BREADTH_THRESHOLD)
        & drawdown_rows["breadth"].notna()
    ]
    risk_off_exec = drawdown_rows.loc[
        drawdown_rows["executed_exposure"].eq(EXPOSURE_LOW)
    ]
    durations = _risk_off_durations(audit["new_exposure"])
    transitions = audit.loc[audit["exposure_transition"] != "NO_CHANGE"]
    activity = {
        "transition_count": int(len(transitions)),
        "risk_off_count": int(
            (transitions["exposure_transition"] == "1.0_TO_0.5").sum()
        ),
        "risk_on_count": int(
            (transitions["exposure_transition"] == "0.5_TO_1.0").sum()
        ),
        "full_exposure_days": int((audit["new_exposure"] == EXPOSURE_HIGH).sum()),
        "half_exposure_days": int((audit["new_exposure"] == EXPOSURE_LOW).sum()),
        "half_exposure_ratio": float((audit["new_exposure"] == EXPOSURE_LOW).mean()),
        "average_exposure": float(audit["new_exposure"].mean()),
        "overlay_turnover": float(audit["overlay_turnover"].sum()),
        "overlay_transaction_cost": float(audit["overlay_transaction_cost"].sum()),
        "average_risk_off_duration": float(np.mean(durations)) if durations else None,
        "minimum_risk_off_duration": min(durations) if durations else None,
        "maximum_risk_off_duration": max(durations) if durations else None,
    }
    daily_returns = daily_replay["returns"].set_index("date")["daily_return"]
    comparison_values = comparison.set_index("metric")
    base_2024_2025_mdd = float(
        _drawdown(replay_returns.loc["2024-01-01":"2025-12-31"]).min()
    )
    daily_2024_2025_mdd = float(
        _drawdown(daily_returns.loc["2024-01-01":"2025-12-31"]).min()
    )
    reb60_2024_2025_mdd = previous.get("comparison", {}).get("mdd_2024_2025_overlay")
    payload: dict[str, Any] = {
        "status": "AVAILABLE",
        "evidence_label": "HISTORICAL_RISK_OVERLAY_EVALUATION",
        "base_strategy_id": STRATEGY_ID,
        "base_strategy_fingerprint": STRATEGY_FINGERPRINT,
        "overlay_id": DAILY_OVERLAY_ID,
        "combined_candidate_id": DAILY_COMBINED_ID,
        "historical_window": {
            "start": HISTORICAL_START.date().isoformat(),
            "end": HISTORICAL_END.date().isoformat(),
            "sessions": len(sessions),
        },
        "rule": {
            "ma_window": MA_WINDOW,
            "threshold": BREADTH_THRESHOLD,
            "exposure_high": EXPOSURE_HIGH,
            "exposure_low": EXPOSURE_LOW,
            "signal_frequency": "DAILY",
            "execution": "NEXT_LEGAL_SESSION",
            "stock_selection_frequency": "REB60",
        },
        "base": {"metrics": _metrics_dict(base_metrics)},
        "previous_reb60_overlay": {
            "identity": previous.get("overlay_id"),
            "classification": previous.get("classification"),
            "metrics": previous.get("overlay", {}).get("metrics", {}),
        },
        "daily": {"metrics": _metrics_dict(daily_metrics)},
        "comparison": {
            "base_2024_2025_mdd": base_2024_2025_mdd,
            "reb60_2024_2025_mdd": reb60_2024_2025_mdd,
            "daily_2024_2025_mdd": daily_2024_2025_mdd,
            "daily_mdd_improvement": daily_2024_2025_mdd - base_2024_2025_mdd,
            "daily_cagr_difference": float(
                comparison_values.at["cagr", "daily"]
                - comparison_values.at["cagr", "base"]
            ),
            "daily_total_return_difference": float(
                comparison_values.at["total_return", "daily"]
                - comparison_values.at["total_return", "base"]
            ),
        },
        "exposure_activity": activity,
        "drawdown_audit": {
            "first_risk_off_signal": (
                risk_off_signals["breadth_signal_date"].min().date().isoformat()
                if not risk_off_signals.empty
                and pd.notna(risk_off_signals["breadth_signal_date"].min())
                else "NOT_OBSERVED"
            ),
            "first_risk_off_execution": (
                risk_off_exec["date"].min().date().isoformat()
                if not risk_off_exec.empty
                else "NOT_OBSERVED"
            ),
            "risk_off_days_in_2024_2025": int(
                (drawdown_rows["executed_exposure"] == EXPOSURE_LOW).sum()
            ),
        },
        "classification_policy": policy,
        "classification": classification,
        "benchmark": {
            key: value for key, value in benchmark.items() if key != "returns"
        },
        "base_replay_matches_canonical": base_replay_matches,
        "base_metrics_match_previous": base_metrics_match,
        "selection_preserved": bool(
            audit.loc[~audit["is_rebalance"], "selected_tickers_changed"]
            .eq(False)
            .all()
        ),
        "relative_weights_preserved": bool(audit["relative_weights_preserved"].all()),
        "fresh_oos_status": "UNCHANGED / INSUFFICIENT_DATA",
        "production_ready": False,
        "strategy_modified": False,
        "previous_overlay_artifact_sha256": previous_hash,
        "identity_binding": {
            "strategy_config_hash": manifest.get("strategy_config_hash"),
            "data_snapshot_hash": manifest.get("data_snapshot_hash"),
            "universe_hash": manifest.get("universe_hash"),
            "cost_model_hash": manifest.get("cost_model_hash"),
            "code_commit": manifest.get("code_commit"),
        },
    }
    output = root / DAILY_OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "risk_overlay_daily_validation.json"
    if validation_path.exists():
        persisted = json.loads(validation_path.read_text(encoding="utf-8"))
        payload["charts"] = persisted.get("charts", [])
        payload["generated_at"] = persisted.get("generated_at")
        payload["code_commit"] = persisted.get("code_commit")
    else:
        breadth_history.to_parquet(
            output / "market_breadth_daily_history.parquet", index=False
        )
        audit.loc[
            :,
            [
                "trade_date",
                "breadth_signal_date",
                "breadth",
                "desired_exposure",
                "previous_exposure",
                "new_exposure",
                "exposure_transition",
                "is_rebalance",
            ],
        ].to_parquet(output / "daily_exposure_history.parquet", index=False)
        audit.to_parquet(output / "daily_overlay_trade_audit.parquet", index=False)
        audit_path.to_csv(output / "2024_2025_drawdown_overlay_audit.csv", index=False)
        comparison.to_csv(output / "risk_overlay_daily_comparison.csv", index=False)
        payload["charts"] = _write_charts(
            output, base_replay, daily_replay, audit, comparison
        )
        payload["generated_at"] = datetime.now(UTC).isoformat()
        payload["code_commit"] = _git_commit(root)
        _write_json(validation_path, payload)
        (output / "risk_overlay_daily_validation_report.md").write_text(
            _report(payload, comparison), encoding="utf-8"
        )
    payload["artifact_sha256"] = {
        path.name: _file_sha256(path)
        for path in (
            output / "market_breadth_daily_history.parquet",
            output / "daily_exposure_history.parquet",
            output / "daily_overlay_trade_audit.parquet",
            output / "2024_2025_drawdown_overlay_audit.csv",
            output / "risk_overlay_daily_comparison.csv",
            output / "risk_overlay_daily_validation.json",
            output / "risk_overlay_daily_validation_report.md",
        )
    }
    _require(
        _file_sha256(root / PREVIOUS_VALIDATION_PATH) == previous_hash,
        "IMMUTABLE_PREVIOUS_OVERLAY_MUTATED",
    )
    return payload


__all__ = [
    "DAILY_COMBINED_ID",
    "DAILY_OVERLAY_ID",
    "DailyRiskOverlayError",
    "evaluate_daily_risk_overlay",
]

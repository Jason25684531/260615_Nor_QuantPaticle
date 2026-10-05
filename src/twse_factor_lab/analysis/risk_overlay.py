"""Minimal, historical-only MA60 breadth risk-overlay evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from twse_factor_lab.backtest.accounting import (
    canonical_replay,
    canonical_transaction_cost,
)
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.production.fundamental import (
    STRATEGY_FINGERPRINT,
    STRATEGY_ID,
    sha256_payload,
)
from twse_factor_lab.reporting.performance_adapter import (
    FrozenStrategyChanged,
    PerformanceData,
    load_frozen_identity,
    load_repository_performance_data,
)
from twse_factor_lab.reporting.performance_metrics import calculate_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMESPACE = Path("data/research/fundamental-production-final-v1")
HISTORICAL_DIR = NAMESPACE / "historical-evidence"
OUTPUT_DIR = NAMESPACE / "risk-overlay"
PRICE_PATH = Path("data/processed/close_matrix.parquet")
UNIVERSE_PATH = Path("data/processed/research_universe.parquet")
BASE_MANIFEST = HISTORICAL_DIR / "canonical_backtest_manifest.json"
BASE_REBALANCES = HISTORICAL_DIR / "canonical_backtest_rebalances.parquet"
BENCHMARK_PATH = HISTORICAL_DIR / "canonical_benchmark_returns.parquet"
OVERLAY_ID = "market_breadth_ma60_040_half_exposure_v1"
COMBINED_ID = "fundamental_g2g3_top5_reb60_score_weighted_plus_breadth040_v1"
BREADTH_THRESHOLD = 0.40
EXPOSURE_HIGH = 1.0
EXPOSURE_LOW = 0.5
MA_WINDOW = 60
INITIAL_CASH = 1_000_000.0
CLASSIFICATION_POLICY = {
    "mdd_improvement_min": 0.05,
    "volatility_reduction_min": 0.02,
    "max_cagr_sacrifice": 0.03,
    "adverse_cagr_sacrifice": 0.05,
}


class RiskOverlayError(ValueError):
    """Raised when the additive historical overlay contract cannot be met."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def exposure_for_breadth(
    breadth: float | None,
    *,
    threshold: float = BREADTH_THRESHOLD,
    exposure_high: float = EXPOSURE_HIGH,
    exposure_low: float = EXPOSURE_LOW,
) -> float:
    """Return the fixed step exposure; missing breadth is conservative."""
    if breadth is None or not math.isfinite(float(breadth)):
        return float(exposure_low)
    return float(exposure_high if float(breadth) > threshold else exposure_low)


def calculate_ma60_breadth(
    close: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    window: int = MA_WINDOW,
) -> pd.DataFrame:
    """Calculate daily PIT-safe breadth from trailing prices and eligibility."""
    if window <= 0:
        raise ValueError("window must be positive")
    prices = close.copy()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index)).normalize()
    prices.columns = prices.columns.astype(str)
    prices = prices.sort_index()
    ma = prices.rolling(window=window, min_periods=window).mean()
    required = {"date", "ticker", "is_eligible"}
    missing = required - set(universe.columns)
    if missing:
        raise RiskOverlayError(f"UNIVERSE_COLUMNS_MISSING:{','.join(sorted(missing))}")
    eligible = universe.loc[:, ["date", "ticker", "is_eligible"]].copy()
    eligible["date"] = pd.to_datetime(eligible["date"]).dt.normalize()
    eligible["ticker"] = eligible["ticker"].astype(str)
    eligible = eligible.drop_duplicates(["date", "ticker"])
    eligible_matrix = (
        eligible.pivot(index="date", columns="ticker", values="is_eligible")
        .reindex(index=prices.index, columns=prices.columns)
        .fillna(False)
        .astype(bool)
    )
    usable = eligible_matrix & prices.notna() & ma.notna()
    above = usable & prices.gt(ma)
    denominator = usable.sum(axis=1).astype(float)
    numerator = above.sum(axis=1).astype(float)
    breadth = numerator.div(denominator.replace(0.0, np.nan))
    result = pd.DataFrame(
        {
            "trade_date": prices.index,
            "eligible_stock_count": denominator.to_numpy(),
            "above_ma60_count": numerator.to_numpy(),
            "breadth": breadth.to_numpy(),
        }
    )
    if start is not None:
        result = result.loc[result["trade_date"] >= pd.Timestamp(start)]
    if end is not None:
        result = result.loc[result["trade_date"] <= pd.Timestamp(end)]
    return result.reset_index(drop=True)


def apply_rebalance_exposure(
    events: pd.DataFrame,
    breadth: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Scale targets at REB60 execution only and carry exposure between events."""
    required = {"signal_date", "execution_date", "ticker", "target_weight"}
    missing = required - set(events.columns)
    if missing:
        raise RiskOverlayError(f"REBALANCE_COLUMNS_MISSING:{','.join(sorted(missing))}")
    base = events.copy()
    base["signal_date"] = pd.to_datetime(base["signal_date"]).dt.normalize()
    base["execution_date"] = pd.to_datetime(base["execution_date"]).dt.normalize()
    base["ticker"] = base["ticker"].astype(str)
    base["target_weight"] = pd.to_numeric(base["target_weight"], errors="raise")
    signal_breadth = breadth.copy()
    signal_breadth["trade_date"] = pd.to_datetime(
        signal_breadth["trade_date"]
    ).dt.normalize()
    signal_breadth = signal_breadth.sort_values("trade_date").set_index("trade_date")
    valid = signal_breadth["breadth"].where(signal_breadth["breadth"].notna()).ffill()
    exposures = {
        date: exposure_for_breadth(valid.get(date))
        for date in pd.DatetimeIndex(sorted(base["signal_date"].unique()))
    }
    scaled = base.copy()
    scaled["breadth"] = scaled["signal_date"].map(signal_breadth["breadth"])
    scaled["exposure"] = scaled["signal_date"].map(exposures).fillna(EXPOSURE_LOW)
    scaled["target_weight"] = scaled["target_weight"] * scaled["exposure"]
    execution_exposure = scaled[["execution_date", "exposure"]].drop_duplicates(
        "execution_date"
    )
    day_index = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize().sort_values()
    state = pd.Series(EXPOSURE_LOW, index=day_index, dtype="float64")
    for row in execution_exposure.sort_values("execution_date").itertuples():
        state.loc[state.index >= row.execution_date] = float(row.exposure)
    audit = signal_breadth.reset_index().rename(columns={"index": "trade_date"})
    audit["exposure"] = state.reindex(audit["trade_date"]).to_numpy()
    audit["rebalance_date"] = audit["trade_date"].isin(
        pd.DatetimeIndex(sorted(base["signal_date"].unique()))
    )
    return scaled, state, audit


def _daily_weights(events: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(index=sessions)
    columns = pd.Index(sorted(events["ticker"].astype(str).unique()))
    targets = (
        events.pivot_table(
            index="execution_date",
            columns="ticker",
            values="target_weight",
            aggfunc="last",
        )
        .reindex(columns=columns)
        .fillna(0.0)
    )
    return targets.reindex(sessions).ffill().fillna(0.0)


def _replay_events(
    close: pd.DataFrame,
    events: pd.DataFrame,
    cost: CostModel,
    sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    weights = _daily_weights(events, sessions)
    close = close.reindex(index=sessions, columns=weights.columns)
    results, returns, turnover, orders = canonical_replay(
        close, weights, cost, INITIAL_CASH
    )
    results.index = sessions
    returns.index = sessions
    orders.index = sessions
    return_frame = pd.DataFrame(
        {
            "date": sessions,
            "daily_return": returns.to_numpy(),
            "gross_return": results["gross_returns"].to_numpy(),
            "cost_return": results["cost_returns"].to_numpy(),
            "portfolio_value": results["equity"].to_numpy(),
            "turnover": turnover.reindex(sessions).fillna(0.0).to_numpy(),
        }
    )
    position_rows: list[dict[str, Any]] = []
    for date, row in results.iterrows():
        equity = float(row["equity"])
        cash = float(row["cash"])
        for ticker in weights.columns:
            value = float(row[f"position:{ticker}"])
            position_rows.append(
                {
                    "date": date,
                    "ticker": str(ticker),
                    "market_value": value,
                    "cash": cash,
                    "portfolio_value": equity,
                    "weight": value / equity if equity else 0.0,
                }
            )
    transactions: list[dict[str, Any]] = []
    for date, row in orders.iterrows():
        for ticker, amount in row[row.ne(0)].items():
            price = float(close.at[date, ticker])
            value = float(amount * price)
            components = canonical_transaction_cost(
                abs(value) if amount > 0 else 0.0,
                abs(value) if amount < 0 else 0.0,
                cost,
            )
            transactions.append(
                {
                    "date": date,
                    "ticker": str(ticker),
                    "amount": float(amount),
                    "price": price,
                    "value": value,
                    "commission": components["buy_fee"] + components["sell_fee"],
                    "tax": components["sell_tax"],
                    "slippage": components["slippage_cost"],
                    "total_cost": components["total_cost"],
                }
            )
    return {
        "returns": return_frame,
        "positions": pd.DataFrame(position_rows),
        "transactions": pd.DataFrame(transactions),
        "weights": weights,
        "orders": orders,
    }


def discover_canonical_benchmark(root: str | Path) -> dict[str, Any]:
    """Return only an existing identity-valid benchmark; never create a proxy."""
    root = Path(root).resolve()
    manifest_path = root / BASE_MANIFEST
    if not manifest_path.exists():
        return {"status": "UNAVAILABLE", "reason": "NO_CANONICAL_BENCHMARK_MANIFEST"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_status") != "AVAILABLE" or not manifest.get(
        "benchmark_id"
    ):
        return {
            "status": "UNAVAILABLE",
            "reason": manifest.get(
                "benchmark_reason", "NO_CANONICAL_MARKET_BENCHMARK_IN_REPOSITORY"
            ),
        }
    path = root / BENCHMARK_PATH
    if not path.exists():
        return {"status": "UNAVAILABLE", "reason": "BENCHMARK_ARTIFACT_MISSING"}
    data = pd.read_parquet(path)
    if data.empty or not {"date", "daily_return"}.issubset(data.columns):
        return {"status": "UNAVAILABLE", "reason": "BENCHMARK_ARTIFACT_EMPTY"}
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = data.sort_values("date").drop_duplicates("date")
    return {
        "status": "AVAILABLE",
        "benchmark_id": manifest["benchmark_id"],
        "source": manifest.get("benchmark_source"),
        "returns": data.set_index("date")["daily_return"].astype(float),
    }


def _metric_value(metrics: Any, name: str) -> Any:
    value = metrics.metrics[name].value
    return value


def _metric_frame(base: Any, overlay: Any) -> pd.DataFrame:
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
    rows = []
    for name in names:
        left = _metric_value(base, name)
        right = _metric_value(overlay, name)
        rows.append(
            {
                "metric": name,
                "base": left,
                "overlay": right,
                "difference": (
                    float(right) - float(left)
                    if isinstance(left, (int, float))
                    and isinstance(right, (int, float))
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _annual_rows(data: PerformanceData) -> pd.DataFrame:
    returns = data.returns
    if returns is None:
        return pd.DataFrame(columns=["year", "return"])
    rows = []
    frame = returns.to_frame("return")
    frame["year"] = frame.index.year
    for year, group in frame.groupby("year"):
        rows.append(
            {"year": int(year), "return": float((1 + group["return"]).prod() - 1)}
        )
    return pd.DataFrame(rows)


def _period_drawdown(data: PerformanceData, start: str, end: str) -> float | None:
    if data.returns is None:
        return None
    returns = data.returns.loc[start:end]
    if returns.empty:
        return None
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _add_annual_comparison(
    comparison: pd.DataFrame, base: PerformanceData, overlay: PerformanceData
) -> pd.DataFrame:
    left = _annual_rows(base).rename(columns={"return": "base"})
    right = _annual_rows(overlay).rename(columns={"return": "overlay"})
    merged = left.merge(right, on="year", how="outer").sort_values("year")
    annual = pd.DataFrame(
        {
            "metric": merged["year"].map(lambda year: f"{int(year)}_return"),
            "base": merged["base"],
            "overlay": merged["overlay"],
        }
    )
    annual["difference"] = annual["overlay"] - annual["base"]
    return pd.concat([comparison, annual], ignore_index=True)


def classify_risk_overlay(comparison: pd.DataFrame) -> str:
    """Apply the predeclared policy, independent of any observed result."""
    values = comparison.set_index("metric")
    required = {"cagr", "sharpe", "max_drawdown", "annualized_volatility"}
    if (
        not required.issubset(values.index)
        or values.loc[list(required)].isna().any().any()
    ):
        return "INSUFFICIENT_DATA"
    mdd_improvement = float(
        values.at["max_drawdown", "overlay"] - values.at["max_drawdown", "base"]
    )
    volatility_reduction = float(
        values.at["annualized_volatility", "base"]
        - values.at["annualized_volatility", "overlay"]
    )
    cagr_sacrifice = float(values.at["cagr", "base"] - values.at["cagr", "overlay"])
    sharpe_delta = float(values.at["sharpe", "overlay"] - values.at["sharpe", "base"])
    if (
        mdd_improvement >= CLASSIFICATION_POLICY["mdd_improvement_min"]
        and volatility_reduction >= CLASSIFICATION_POLICY["volatility_reduction_min"]
        and cagr_sacrifice <= CLASSIFICATION_POLICY["max_cagr_sacrifice"]
        and sharpe_delta >= 0
    ):
        return "RISK_OVERLAY_SUPPORTIVE"
    if cagr_sacrifice > CLASSIFICATION_POLICY["adverse_cagr_sacrifice"] or (
        mdd_improvement <= 0
        and (
            sharpe_delta < 0
            or cagr_sacrifice > CLASSIFICATION_POLICY["max_cagr_sacrifice"]
        )
    ):
        return "ADVERSE"
    return "MIXED"


def _series_data(
    period: str,
    frame: dict[str, Any],
    benchmark: pd.Series | None,
) -> PerformanceData:
    def utc_dates(values: Any) -> pd.DatetimeIndex:
        dates = pd.DatetimeIndex(pd.to_datetime(values))
        return (
            dates.tz_convert("UTC")
            if dates.tz is not None
            else dates.tz_localize("UTC")
        )

    returns = frame["returns"].set_index("date")["daily_return"]
    returns.index = utc_dates(returns.index)
    positions = frame["positions"].copy()
    transactions = frame["transactions"].copy()
    for item in (positions, transactions):
        item["date"] = utc_dates(item["date"])
    benchmark_series = None
    if benchmark is not None:
        benchmark_series = benchmark.reindex(returns.index.tz_localize(None)).copy()
        benchmark_series.index = returns.index
    return PerformanceData.from_frames(
        period,
        returns,
        benchmark_returns=benchmark_series,
        positions=positions,
        transactions=transactions,
    )


def _chart_paths(output: Path) -> list[Path]:
    return [
        output / "charts" / f"{index:02d}_{name}.png"
        for index, name in (
            (1, "base_vs_overlay_cumulative_return"),
            (2, "base_vs_overlay_drawdown"),
            (3, "market_breadth_and_exposure"),
            (4, "annual_return_comparison"),
        )
    ]


def _write_charts(
    output: Path,
    base: PerformanceData,
    overlay: PerformanceData,
    breadth: pd.DataFrame,
) -> list[str]:
    charts = _chart_paths(output)
    for path in charts:
        path.parent.mkdir(parents=True, exist_ok=True)
    base_returns = base.returns
    overlay_returns = overlay.returns
    if base_returns is None or overlay_returns is None:
        return []
    base_returns.index = pd.DatetimeIndex(base_returns.index)
    overlay_returns.index = pd.DatetimeIndex(overlay_returns.index)
    base_equity = (1 + base_returns).cumprod() - 1
    overlay_equity = (1 + overlay_returns).cumprod() - 1
    plt.figure(figsize=(10, 5))
    plt.plot(base_equity.index, base_equity, label="BASE")
    plt.plot(overlay_equity.index, overlay_equity, label="OVERLAY")
    plt.title("BASE vs OVERLAY Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts[0], dpi=140)
    plt.close()

    def drawdown(series: pd.Series) -> pd.Series:
        equity = (1 + series).cumprod()
        return equity / equity.cummax() - 1

    plt.figure(figsize=(10, 5))
    plt.plot(base_returns.index, drawdown(base_returns), label="BASE")
    plt.plot(overlay_returns.index, drawdown(overlay_returns), label="OVERLAY")
    plt.title("BASE vs OVERLAY Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts[1], dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    dates = pd.to_datetime(breadth["trade_date"])
    axis = plt.gca()
    axis.plot(dates, breadth["breadth"], label="Breadth")
    axis.axhline(BREADTH_THRESHOLD, color="black", linestyle="--", label="0.40")
    axis.set_ylabel("Breadth")
    second = axis.twinx()
    second.step(
        dates, breadth["exposure"], where="post", color="tab:red", label="Exposure"
    )
    second.set_ylabel("Exposure")
    axis.set_title("Market Breadth and REB60 Exposure")
    plt.tight_layout()
    plt.savefig(charts[2], dpi=140)
    plt.close()

    annual_base = _annual_rows(base).set_index("year")["return"]
    annual_overlay = _annual_rows(overlay).set_index("year")["return"]
    years = sorted(set(annual_base.index) | set(annual_overlay.index))
    x = np.arange(len(years))
    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2, [annual_base.get(y, np.nan) for y in years], width, label="BASE"
    )
    plt.bar(
        x + width / 2,
        [annual_overlay.get(y, np.nan) for y in years],
        width,
        label="OVERLAY",
    )
    plt.xticks(x, years)
    plt.title("Annual Return Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts[3], dpi=140)
    plt.close()
    return [
        str(path.relative_to(output.parent.parent)).replace("\\", "/")
        for path in charts
    ]


def _report(payload: dict[str, Any], comparison: pd.DataFrame) -> str:
    def value(name: str, scenario: str) -> str:
        item = payload[scenario]["metrics"].get(name)
        if item is None:
            return "UNAVAILABLE"
        return f"{item:.6f}" if isinstance(item, (int, float)) else str(item)

    def fmt(item: Any) -> str:
        if pd.isna(item):
            return "UNAVAILABLE"
        return f"{item:.6f}" if isinstance(item, (int, float)) else str(item)

    table = ["| Metric | Base | Overlay | Difference |", "|---|---:|---:|---:|"]
    for row in comparison.itertuples(index=False):
        table.append(
            f"| {row.metric} | {fmt(row.base)} | {fmt(row.overlay)} | "
            f"{fmt(row.difference)} |"
        )
    return "\n".join(
        [
            "# Risk Overlay Validation Report",
            "",
            "## 1. Base Strategy",
            "",
            f"Strategy = {STRATEGY_ID}",
            "G2 + G3 / Top5 / Score Weighted / REB60",
            "",
            "## 2. Risk Overlay",
            "",
            "Breadth = eligible stocks with Close > MA60 / eligible stocks",
            f"Threshold = {BREADTH_THRESHOLD:.2f}",
            f"Exposure = {EXPOSURE_HIGH:.1f} / {EXPOSURE_LOW:.1f} (REB60-only)",
            "",
            "## 3. Performance Comparison",
            "",
            *table,
            "",
            "## 4. Drawdown Comparison",
            "",
            f"Base MDD = {value('max_drawdown', 'base')}",
            f"Overlay MDD = {value('max_drawdown', 'overlay')}",
            f"Base Worst Drawdown Start = {value('max_drawdown_start', 'base')}",
            f"Overlay Worst Drawdown Start = {value('max_drawdown_start', 'overlay')}",
            f"Base Worst Drawdown Trough = {value('max_drawdown_trough', 'base')}",
            "Overlay Worst Drawdown Trough = "
            f"{value('max_drawdown_trough', 'overlay')}",
            f"Base Worst Drawdown Recovery = {value('max_drawdown_recovery', 'base')}",
            "Overlay Worst Drawdown Recovery = "
            f"{value('max_drawdown_recovery', 'overlay')}",
            f"Base Worst Drawdown Duration = {value('max_drawdown_duration', 'base')}",
            "Overlay Worst Drawdown Duration = "
            f"{value('max_drawdown_duration', 'overlay')}",
            f"2024-2025 Base MDD = {payload['comparison']['mdd_2024_2025_base']:.6f}",
            "2024-2025 Overlay MDD = "
            f"{payload['comparison']['mdd_2024_2025_overlay']:.6f}",
            "",
            "## 5. Return Cost",
            "",
            f"Base CAGR = {value('cagr', 'base')}",
            f"Overlay CAGR = {value('cagr', 'overlay')}",
            f"Return Sacrifice = {payload['comparison']['cagr_sacrifice']:.6f}",
            "",
            "## 6. Conclusion",
            "",
            payload["classification"],
            "",
            "## 7. Status",
            "",
            "Base Frozen Fingerprint = PASS",
            "Base Strategy Modified = NO",
            f"Fresh OOS = {payload['fresh_oos_status']}",
            "Production Ready = NO",
            "",
            "Evidence Label = HISTORICAL_RISK_OVERLAY_EVALUATION",
            f"Benchmark = {payload['benchmark']['status']}",
        ]
    )


def evaluate_risk_overlay(root: str | Path) -> dict[str, Any]:
    """Generate the additive BASE-vs-OVERLAY historical evidence package."""
    root = Path(root).resolve()
    identity = load_frozen_identity(root)
    if (
        identity.strategy_id != STRATEGY_ID
        or identity.strategy_fingerprint != STRATEGY_FINGERPRINT
    ):
        raise FrozenStrategyChanged("FROZEN_STRATEGY_CHANGED")
    source = load_repository_performance_data(root, identity)["BACKTEST"]
    if not source.available or source.returns is None:
        raise RiskOverlayError("NO_IDENTITY_BOUND_CANONICAL_BACKTEST_SOURCE")
    manifest_path = root / BASE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("strategy_fingerprint") != STRATEGY_FINGERPRINT:
        raise FrozenStrategyChanged("FROZEN_STRATEGY_CHANGED")
    close = pd.read_parquet(root / PRICE_PATH)
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    close.columns = close.columns.astype(str)
    universe = pd.read_parquet(root / UNIVERSE_PATH)
    sessions = pd.DatetimeIndex(
        pd.to_datetime(source.returns.index).tz_localize(None)
    ).sort_values()
    start, end = sessions.min(), sessions.max()
    breadth = calculate_ma60_breadth(close, universe, start=start, end=end)
    rebalances = pd.read_parquet(root / BASE_REBALANCES)
    scaled_events, _, breadth_audit = apply_rebalance_exposure(
        rebalances.loc[:, ["signal_date", "execution_date", "ticker", "target_weight"]],
        breadth,
        sessions,
    )
    cost_payload = manifest.get("cost_model", {})
    try:
        cost = CostModel(
            buy_fee_rate=float(cost_payload["buy_fee_rate"]),
            sell_fee_rate=float(cost_payload["sell_fee_rate"]),
            transaction_tax_rate=float(cost_payload["transaction_tax_rate"]),
            slippage_rate=float(cost_payload["slippage_rate"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RiskOverlayError("COST_MODEL_UNAVAILABLE") from exc
    expected_cost_hash = manifest.get("cost_model_hash")
    if expected_cost_hash and expected_cost_hash != sha256_payload(cost.summary()):
        raise RiskOverlayError("COST_MODEL_CHANGED")
    close_scope = close.loc[:, sorted(scaled_events["ticker"].astype(str).unique())]
    overlay_frame = _replay_events(close_scope, scaled_events, cost, sessions)
    benchmark = discover_canonical_benchmark(root)
    benchmark_series = (
        benchmark.get("returns") if benchmark["status"] == "AVAILABLE" else None
    )
    base_frame = {
        "returns": source.returns.reset_index().rename(
            columns={"index": "date", "returns": "daily_return"}
        ),
        "positions": source.positions
        if source.positions is not None
        else pd.DataFrame(),
        "transactions": source.transactions
        if source.transactions is not None
        else pd.DataFrame(),
    }
    base_data = _series_data("BACKTEST", base_frame, benchmark_series)
    overlay_data = _series_data("BACKTEST", overlay_frame, benchmark_series)
    base_metrics = calculate_metrics(base_data)
    overlay_metrics = calculate_metrics(overlay_data)
    comparison = _add_annual_comparison(
        _metric_frame(base_metrics, overlay_metrics), base_data, overlay_data
    )
    mdd_2024_2025_base = _period_drawdown(base_data, "2024-01-01", "2025-12-31")
    mdd_2024_2025_overlay = _period_drawdown(overlay_data, "2024-01-01", "2025-12-31")
    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {
                        "metric": "2024_2025_max_drawdown",
                        "base": mdd_2024_2025_base,
                        "overlay": mdd_2024_2025_overlay,
                        "difference": (
                            mdd_2024_2025_overlay - mdd_2024_2025_base
                            if mdd_2024_2025_base is not None
                            and mdd_2024_2025_overlay is not None
                            else np.nan
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    policy_values = comparison.set_index("metric")
    cagr_sacrifice = float(
        policy_values.at["cagr", "base"] - policy_values.at["cagr", "overlay"]
    )
    mdd_improvement = float(
        policy_values.at["max_drawdown", "overlay"]
        - policy_values.at["max_drawdown", "base"]
    )
    volatility_reduction = float(
        policy_values.at["annualized_volatility", "base"]
        - policy_values.at["annualized_volatility", "overlay"]
    )
    summary = {
        "full_exposure_days": int((breadth_audit["exposure"] == EXPOSURE_HIGH).sum()),
        "half_exposure_days": int((breadth_audit["exposure"] == EXPOSURE_LOW).sum()),
        "half_exposure_ratio": float(
            (breadth_audit["exposure"] == EXPOSURE_LOW).mean()
        ),
        "minimum_breadth": float(breadth["breadth"].min())
        if breadth["breadth"].notna().any()
        else None,
        "average_breadth": float(breadth["breadth"].mean())
        if breadth["breadth"].notna().any()
        else None,
        "rebalance_exposure_days": int(breadth_audit["rebalance_date"].sum()),
        "non_rebalance_turnover": 0.0,
    }
    base_keys = rebalances.loc[:, ["signal_date", "ticker"]].copy()
    overlay_keys = scaled_events.loc[:, ["signal_date", "ticker"]].copy()
    selection_preserved = base_keys.equals(overlay_keys)
    base_relative = rebalances.groupby("signal_date")["target_weight"].transform(
        lambda values: values / values.sum()
    )
    overlay_relative = scaled_events.groupby("signal_date")["target_weight"].transform(
        lambda values: values / values.sum()
    )
    relative_weights_preserved = bool(
        selection_preserved
        and np.allclose(base_relative.to_numpy(), overlay_relative.to_numpy())
    )

    def metrics_to_dict(metrics: Any) -> dict[str, Any]:
        return {name: item.value for name, item in metrics.metrics.items()}

    validation = {
        "status": "AVAILABLE",
        "evidence_label": "HISTORICAL_RISK_OVERLAY_EVALUATION",
        "base_strategy_id": STRATEGY_ID,
        "base_strategy_fingerprint": STRATEGY_FINGERPRINT,
        "overlay_id": OVERLAY_ID,
        "combined_candidate_id": COMBINED_ID,
        "historical_window": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "sessions": len(sessions),
        },
        "rule": {
            "ma_window": MA_WINDOW,
            "threshold": BREADTH_THRESHOLD,
            "exposure_high": EXPOSURE_HIGH,
            "exposure_low": EXPOSURE_LOW,
            "rebalance_only": True,
        },
        "base": {"metrics": metrics_to_dict(base_metrics)},
        "overlay": {"metrics": metrics_to_dict(overlay_metrics)},
        "breadth_summary": summary,
        "comparison": {
            "mdd_improvement": mdd_improvement,
            "volatility_reduction": volatility_reduction,
            "sharpe_delta": float(
                policy_values.at["sharpe", "overlay"]
                - policy_values.at["sharpe", "base"]
            ),
            "cagr_sacrifice": cagr_sacrifice,
            "total_return_difference": float(
                policy_values.at["total_return", "overlay"]
                - policy_values.at["total_return", "base"]
            ),
            "mdd_2024_2025_base": mdd_2024_2025_base,
            "mdd_2024_2025_overlay": mdd_2024_2025_overlay,
            "mdd_2024_2025_difference": (
                mdd_2024_2025_overlay - mdd_2024_2025_base
                if mdd_2024_2025_base is not None and mdd_2024_2025_overlay is not None
                else None
            ),
        },
        "classification_policy": CLASSIFICATION_POLICY,
        "classification": classify_risk_overlay(comparison),
        "benchmark": {
            key: value for key, value in benchmark.items() if key != "returns"
        },
        "fresh_oos_status": "UNCHANGED / INSUFFICIENT_DATA",
        "production_ready": False,
        "strategy_modified": False,
        "selection_preserved": selection_preserved,
        "relative_weights_preserved": relative_weights_preserved,
        "identity_binding": {
            "strategy_config_hash": manifest.get("strategy_config_hash"),
            "data_snapshot_hash": manifest.get("data_snapshot_hash"),
            "universe_hash": manifest.get("universe_hash"),
            "cost_model_hash": manifest.get("cost_model_hash"),
            "code_commit": manifest.get("code_commit"),
        },
    }
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "risk_overlay_validation.json"
    if validation_path.exists():
        persisted = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["charts"] = persisted.get("charts", [])
        validation["generated_at"] = persisted.get("generated_at")
        validation["code_commit"] = persisted.get("code_commit")
    else:
        breadth_audit.to_parquet(output / "market_breadth_history.parquet", index=False)
        _write_json(output / "breadth_overlay_summary.json", summary)
        comparison.to_csv(output / "risk_overlay_comparison.csv", index=False)
        validation["charts"] = _write_charts(
            output, base_data, overlay_data, breadth_audit
        )
        validation["generated_at"] = datetime.now(UTC).isoformat()
        validation["code_commit"] = _git_commit(root)
        _write_json(validation_path, validation)
        (output / "risk_overlay_validation_report.md").write_text(
            _report(
                {
                    **validation,
                    "base": {"metrics": metrics_to_dict(base_metrics)},
                    "overlay": {"metrics": metrics_to_dict(overlay_metrics)},
                },
                comparison,
            ),
            encoding="utf-8",
        )
    validation["artifact_sha256"] = {
        path.name: _file_sha256(path)
        for path in (
            output / "market_breadth_history.parquet",
            output / "breadth_overlay_summary.json",
            output / "risk_overlay_comparison.csv",
            output / "risk_overlay_validation.json",
            output / "risk_overlay_validation_report.md",
        )
    }
    return validation


__all__ = [
    "BREADTH_THRESHOLD",
    "CLASSIFICATION_POLICY",
    "COMBINED_ID",
    "EXPOSURE_HIGH",
    "EXPOSURE_LOW",
    "MA_WINDOW",
    "OVERLAY_ID",
    "RiskOverlayError",
    "apply_rebalance_exposure",
    "calculate_ma60_breadth",
    "classify_risk_overlay",
    "discover_canonical_benchmark",
    "evaluate_risk_overlay",
    "exposure_for_breadth",
]

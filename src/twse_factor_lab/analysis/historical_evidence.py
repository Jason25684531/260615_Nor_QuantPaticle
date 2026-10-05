"""PIT-safe historical evidence for the frozen Fundamental strategy."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
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
from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.production.final_runtime import (
    canonical_factor_rows,
    canonical_factor_values,
    fresh_oos_audit,
)
from twse_factor_lab.production.fundamental import (
    STRATEGY_FINGERPRINT,
    STRATEGY_ID,
    FundamentalStrategySpec,
    sha256_payload,
)
from twse_factor_lab.reporting.performance_adapter import PerformanceData
from twse_factor_lab.reporting.performance_metrics import calculate_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMESPACE = Path("data/research/fundamental-production-final-v1")
EVIDENCE_DIR = NAMESPACE / "historical-evidence"
POLICY_PATH = Path("config/historical_evidence.json")
PRICE_PATH = Path("data/processed/close_matrix.parquet")
OHLCV_PATH = Path("data/processed/ohlcv.parquet")
UNIVERSE_PATH = Path("data/processed/research_universe.parquet")
PIT_PATH = Path("data/processed/fundamental_pit_v2/fundamental_records.parquet")
PIT_MANIFEST_PATH = Path(
    "data/processed/fundamental_pit_v2/fundamental_manifest.json"
)
INITIAL_CASH = 1_000_000.0


class HistoricalEvidenceError(ValueError):
    """Raised when historical evidence cannot satisfy a hard contract."""


@dataclass(frozen=True)
class HistoricalInputs:
    records: pd.DataFrame
    universe: pd.DataFrame
    close: pd.DataFrame
    sessions: pd.DatetimeIndex
    start: pd.Timestamp
    end: pd.Timestamp


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_sha(path: Path) -> str:
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


def load_policy(root: str | Path) -> dict[str, Any]:
    path = Path(root).resolve() / POLICY_PATH
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "policy_version",
        "maximum_years",
        "minimum_sessions",
        "factor_sample_stride",
        "forward_horizons",
        "primary_horizon",
        "quantiles",
        "rolling_ic_observations",
        "bootstrap",
        "classification",
    }
    if not required.issubset(policy):
        raise HistoricalEvidenceError("HISTORICAL_POLICY_INVALID")
    if policy["primary_horizon"] not in policy["forward_horizons"]:
        raise HistoricalEvidenceError("HISTORICAL_POLICY_INVALID")
    return policy


def _load_inputs(root: Path, policy: dict[str, Any]) -> HistoricalInputs:
    required = (PIT_PATH, UNIVERSE_PATH, PRICE_PATH, OHLCV_PATH)
    missing = [str(path) for path in required if not (root / path).exists()]
    if missing:
        raise HistoricalEvidenceError("HISTORICAL_SOURCE_MISSING:" + ",".join(missing))
    records = pd.read_parquet(root / PIT_PATH)
    universe = pd.read_parquet(root / UNIVERSE_PATH)
    close = pd.read_parquet(root / PRICE_PATH)
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    close.columns = close.columns.astype(str)
    sessions = pd.DatetimeIndex(sorted(close.index.unique()))
    cutoff = pd.Timestamp(FundamentalStrategySpec().research_knowledge_cutoff)
    end = sessions[sessions < cutoff].max()
    earliest = end - pd.DateOffset(years=int(policy["maximum_years"]))
    window = sessions[(sessions > earliest) & (sessions <= end)]
    if window.empty:
        raise HistoricalEvidenceError("NO_LEGAL_HISTORICAL_WINDOW")
    by_date = {date: group for date, group in universe.groupby("date", sort=False)}
    start = None
    for date in window:
        day = by_date.get(date)
        if day is None:
            continue
        try:
            if canonical_factor_rows(records, day, date.date().isoformat()):
                start = date
                break
        except ValueError:
            continue
    if start is None:
        raise HistoricalEvidenceError("NO_LEGAL_G2_G3_WINDOW")
    sessions = window[window >= start]
    return HistoricalInputs(records, universe, close.loc[sessions], sessions, start, end)


def historical_inventory(root: str | Path, policy: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    inputs = _load_inputs(root, policy)
    records = inputs.records
    relevant = records.loc[records["metric"].isin(["operating_income", "eps"])]
    ohlcv = pd.read_parquet(root / OHLCV_PATH, columns=["date", "ticker"])
    pit_manifest = (
        json.loads((root / PIT_MANIFEST_PATH).read_text(encoding="utf-8"))
        if (root / PIT_MANIFEST_PATH).exists()
        else {}
    )
    current_pit_hash = _file_sha(root / PIT_PATH)
    recorded_pit_hash = pit_manifest.get("artifact_sha256", {}).get(
        "fundamental_records.parquet"
    )
    warnings = [
        "CURRENT_LISTED_ONLY_UNIVERSE_SURVIVORSHIP_LIMITATION",
        "NO_CANONICAL_MARKET_BENCHMARK_IN_REPOSITORY",
        "HISTORICAL_ROBUSTNESS_IS_NOT_HISTORICAL_OOS",
    ]
    if recorded_pit_hash and recorded_pit_hash != current_pit_hash:
        warnings.append("FUNDAMENTAL_MANIFEST_STALE_AFTER_OPERATING_INCOME_BACKFILL")
    available_years = (inputs.end - inputs.start).days / 365.2425
    old_backtests = sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in (root / "data").rglob("*backtest*.parquet")
        if EVIDENCE_DIR.as_posix() not in path.as_posix()
    )
    old_oos = sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in (root / "data").rglob("*oos*")
        if path.is_file() and EVIDENCE_DIR.as_posix() not in path.as_posix()
    )
    return {
        "price_start": pd.to_datetime(ohlcv["date"]).min().date().isoformat(),
        "price_end": pd.to_datetime(ohlcv["date"]).max().date().isoformat(),
        "fundamental_start": pd.to_datetime(relevant["available_date"])
        .min()
        .date()
        .isoformat(),
        "fundamental_end": pd.to_datetime(relevant["available_date"])
        .max()
        .date()
        .isoformat(),
        "usable_start": inputs.start.date().isoformat(),
        "usable_end": inputs.end.date().isoformat(),
        "historical_window_start": inputs.start.date().isoformat(),
        "historical_window_end": inputs.end.date().isoformat(),
        "available_years": available_years,
        "years_covered": available_years,
        "trading_sessions": len(inputs.sessions),
        "data_coverage_ratio": len(inputs.sessions)
        / len(inputs.close.loc[inputs.start : inputs.end]),
        "benchmark_available": False,
        "benchmark_status": "UNAVAILABLE",
        "benchmark_reason": "NO_CANONICAL_MARKET_BENCHMARK_IN_REPOSITORY",
        "pit_available": True,
        "pit_source": str(PIT_PATH).replace("\\", "/"),
        "price_source": str(PRICE_PATH).replace("\\", "/"),
        "universe_source": str(UNIVERSE_PATH).replace("\\", "/"),
        "cost_source": "twse_factor_lab.backtest.costs.CostModel",
        "existing_backtest_sources": old_backtests,
        "existing_oos_sources": old_oos,
        "existing_sources_identity_bound_to_frozen_strategy": False,
        "warnings": warnings,
    }


def pit_audit(
    records: pd.DataFrame, observations: pd.DataFrame | None = None
) -> dict[str, Any]:
    required = {
        "ticker",
        "metric",
        "period_end",
        "publication_date",
        "available_date",
        "value",
        "source",
        "pit_status",
    }
    missing = sorted(required - set(records.columns))
    if "metric" not in records:
        relevant = records.iloc[0:0].copy()
    else:
        relevant = records.loc[
            records["metric"].isin(["operating_income", "eps"])
        ].copy()
    for column in ("period_end", "publication_date", "available_date"):
        if column in relevant:
            relevant[column] = pd.to_datetime(relevant[column], errors="coerce")
    duplicate_count = (
        int(
            relevant.duplicated(
                ["ticker", "metric", "period_end", "publication_date"]
            ).sum()
        )
        if not missing
        else 0
    )
    source_failures = 0
    if not missing:
        source_failures = int(
            relevant[list(required)].isna().any(axis=1).sum()
            + (relevant["period_end"] > relevant["publication_date"]).sum()
            + (relevant["publication_date"] > relevant["available_date"]).sum()
            + (~relevant["pit_status"].eq("PUBLICATION_DATE_AWARE")).sum()
        )
    used_failures = 0
    if observations is not None and not observations.empty:
        used_failures = int(
            (observations["fiscal_period"] > observations["announcement_date"]).sum()
            + (
                observations["announcement_date"]
                > observations["effective_date"]
            ).sum()
            + (
                observations["effective_date"]
                > observations["used_on_trade_date"]
            ).sum()
        )
    passed = not missing and not duplicate_count and not source_failures and not used_failures
    return {
        "status": "PASS" if passed else "BLOCKED_PIT",
        "reason": None if passed else "PIT_PROVENANCE_INVALID",
        "required_columns": sorted(required),
        "missing_columns": missing,
        "source_rows": len(relevant),
        "used_observations": 0 if observations is None else len(observations),
        "duplicate_records": duplicate_count,
        "source_ordering_failures": source_failures,
        "used_on_trade_date_failures": used_failures,
        "future_leakage_detected": used_failures > 0,
        "leakage": "NONE_DETECTED" if passed else "DETECTED_OR_UNVERIFIABLE",
    }


def _factor_snapshots(
    inputs: HistoricalInputs, policy: dict[str, Any]
) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    stride = int(policy["factor_sample_stride"])
    sample_dates = set(inputs.sessions[::stride])
    for interval in (40, 60, 80):
        sample_dates.update(inputs.sessions[::interval])
    sample_dates.add(inputs.sessions[-1])
    by_date = {
        pd.Timestamp(date): group
        for date, group in inputs.universe.loc[
            inputs.universe["date"].between(inputs.start, inputs.end)
        ].groupby("date", sort=False)
    }
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    observation_rows: list[dict[str, Any]] = []
    for date in sorted(sample_dates):
        day = by_date.get(date)
        if day is None:
            continue
        values = canonical_factor_values(
            inputs.records, day, date.date().isoformat()
        )
        if values.empty:
            continue
        values["normalized_g2"] = values["g2_raw"].rank(
            method="average", pct=True
        )
        values["normalized_g3"] = values["g3_raw"].rank(
            method="average", pct=True
        )
        values["composite_score"] = (
            values["normalized_g2"] + values["normalized_g3"]
        ) / 2
        snapshots[date] = values
        for factor, prefix in (("G2", "g2"), ("G3", "g3")):
            current = values.dropna(
                subset=[
                    f"{prefix}_period_end",
                    f"{prefix}_publication_date",
                    f"{prefix}_available_date",
                ]
            )
            observation_rows.extend(
                {
                    "ticker": row.ticker,
                    "factor_id": factor,
                    "fiscal_period": getattr(row, f"{prefix}_period_end"),
                    "announcement_date": getattr(
                        row, f"{prefix}_publication_date"
                    ),
                    "effective_date": getattr(row, f"{prefix}_available_date"),
                    "used_on_trade_date": date,
                }
                for row in current.itertuples()
            )
    observations = pd.DataFrame(observation_rows)
    for column in (
        "fiscal_period",
        "announcement_date",
        "effective_date",
        "used_on_trade_date",
    ):
        observations[column] = pd.to_datetime(observations[column])
    return snapshots, observations


def _block_ci(values: pd.Series, policy: dict[str, Any]) -> dict[str, Any]:
    settings = policy["bootstrap"]
    clean = values.dropna().to_numpy(dtype=float)
    if not settings["enabled"] or len(clean) < int(settings["minimum_observations"]):
        return {"status": "INSUFFICIENT_DATA", "low": None, "high": None}
    block = min(int(settings["block_length"]), len(clean))
    draws = int(settings["draws"])
    rng = np.random.default_rng(int(settings["seed"]))
    estimates = np.empty(draws)
    for draw in range(draws):
        sampled: list[float] = []
        while len(sampled) < len(clean):
            start = int(rng.integers(0, len(clean) - block + 1))
            sampled.extend(clean[start : start + block])
        estimates[draw] = np.mean(sampled[: len(clean)])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"status": "AVAILABLE", "low": float(low), "high": float(high)}


def _factor_validation(
    factor: str,
    column: str,
    snapshots: dict[pd.Timestamp, pd.DataFrame],
    inputs: HistoricalInputs,
    policy: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    top_sets: list[set[str]] = []
    stride_dates = set(inputs.sessions[:: int(policy["factor_sample_stride"])])
    index_lookup = {date: index for index, date in enumerate(inputs.sessions)}
    for date in sorted(set(snapshots) & stride_dates):
        frame = snapshots[date].dropna(subset=[column]).copy()
        eligible_count = int(
            inputs.universe.loc[
                inputs.universe["date"].eq(date)
                & inputs.universe["is_eligible"].astype(bool),
                "ticker",
            ].nunique()
        )
        for horizon in policy["forward_horizons"]:
            future_index = index_lookup[date] + int(horizon)
            if future_index >= len(inputs.sessions):
                continue
            future = inputs.sessions[future_index]
            tickers = frame["ticker"].astype(str)
            current_price = inputs.close.loc[date].reindex(tickers).to_numpy(float)
            future_price = inputs.close.loc[future].reindex(tickers).to_numpy(float)
            frame["forward_return"] = future_price / current_price - 1.0
            valid = frame.replace([np.inf, -np.inf], np.nan).dropna(
                subset=[column, "forward_return"]
            )
            ic = valid[column].rank().corr(valid["forward_return"].rank())
            record: dict[str, Any] = {
                "factor": factor,
                "date": date,
                "horizon": int(horizon),
                "ic": ic,
                "observations": len(valid),
                "eligible_count": eligible_count,
                "coverage": len(valid) / eligible_count if eligible_count else np.nan,
                "missing_ratio": 1 - len(valid) / eligible_count
                if eligible_count
                else np.nan,
            }
            q = int(policy["quantiles"])
            if len(valid) >= q:
                valid = valid.copy()
                valid["quantile"] = pd.qcut(
                    valid[column].rank(method="first"), q, labels=range(1, q + 1)
                ).astype(int)
                means = valid.groupby("quantile")["forward_return"].mean()
                record["top_bottom_spread"] = float(means.loc[q] - means.loc[1])
                record["monotonic"] = bool(means.is_monotonic_increasing)
                for quantile, group in valid.groupby("quantile"):
                    quantile_rows.append(
                        {
                            "factor": factor,
                            "date": date,
                            "horizon": int(horizon),
                            "quantile": int(quantile),
                            "mean_forward_return": group["forward_return"].mean(),
                            "median_forward_return": group[
                                "forward_return"
                            ].median(),
                            "hit_ratio": (group["forward_return"] > 0).mean(),
                            "observations": len(group),
                        }
                    )
                if int(horizon) == int(policy["primary_horizon"]):
                    top_sets.append(
                        set(valid.loc[valid["quantile"].eq(q), "ticker"].astype(str))
                    )
            else:
                record["top_bottom_spread"] = np.nan
                record["monotonic"] = None
            rows.append(record)
    daily = pd.DataFrame(rows)
    quantiles = pd.DataFrame(quantile_rows)
    primary = daily.loc[daily["horizon"].eq(policy["primary_horizon"])].copy()
    ic = primary["ic"].dropna()
    spread = primary["top_bottom_spread"].dropna()
    annual = primary.assign(year=pd.to_datetime(primary["date"]).dt.year).groupby(
        "year"
    )["ic"].agg(["mean", "median", "count"])
    q_summary = (
        quantiles.loc[quantiles["horizon"].eq(policy["primary_horizon"])]
        .groupby("quantile")
        .agg(
            mean_forward_return=("mean_forward_return", "mean"),
            median_forward_return=("median_forward_return", "median"),
            hit_ratio=("hit_ratio", "mean"),
            observations=("observations", "sum"),
        )
        .reset_index()
    )
    turnovers = [
        1 - len(left & right) / max(len(left), 1)
        for left, right in zip(top_sets, top_sets[1:], strict=False)
    ]
    top_returns = q_summary.loc[
        q_summary["quantile"].eq(policy["quantiles"]), "mean_forward_return"
    ]
    approximate_return = float(top_returns.iloc[0]) if not top_returns.empty else np.nan
    scale = 252 / int(policy["primary_horizon"])
    primary_top = (
        quantiles.loc[
            quantiles["horizon"].eq(policy["primary_horizon"])
            & quantiles["quantile"].eq(policy["quantiles"]),
            "mean_forward_return",
        ]
        .dropna()
        .reset_index(drop=True)
    )
    factor_drawdown = (
        (1 + primary_top).cumprod().div((1 + primary_top).cumprod().cummax()).sub(1)
        if not primary_top.empty
        else pd.Series(dtype=float)
    )
    summary = {
        "factor": factor,
        "primary_horizon": int(policy["primary_horizon"]),
        "rank_ic": float(ic.mean()) if not ic.empty else None,
        "ic_mean": float(ic.mean()) if not ic.empty else None,
        "ic_median": float(ic.median()) if not ic.empty else None,
        "ic_positive_ratio": float((ic > 0).mean()) if not ic.empty else None,
        "ic_information_ratio": float(ic.mean() / ic.std(ddof=0))
        if len(ic) > 1 and ic.std(ddof=0)
        else None,
        "ic_observations": len(ic),
        "annual_ic": [
            {
                "year": int(year),
                "mean": row["mean"],
                "median": row["median"],
                "observations": int(row["count"]),
            }
            for year, row in annual.iterrows()
        ],
        "positive_ic_year_ratio": float((annual["mean"] > 0).mean())
        if not annual.empty
        else None,
        "rolling_ic": [
            {"date": date.date().isoformat(), "value": value}
            for date, value in zip(
                pd.to_datetime(primary["date"]),
                primary["ic"].rolling(
                    int(policy["rolling_ic_observations"]),
                    min_periods=int(policy["rolling_ic_observations"]),
                ).mean(),
                strict=True,
            )
            if pd.notna(value)
        ],
        "top_quantile_return": approximate_return,
        "bottom_quantile_return": (
            float(q_summary.iloc[0]["mean_forward_return"])
            if not q_summary.empty
            else None
        ),
        "top_bottom_spread": float(spread.mean()) if not spread.empty else None,
        "quantile_monotonicity_ratio": float(primary["monotonic"].dropna().mean())
        if primary["monotonic"].notna().any()
        else None,
        "turnover": float(np.mean(turnovers)) if turnovers else None,
        "coverage": float(primary["coverage"].mean()) if not primary.empty else None,
        "missing_data_ratio": float(primary["missing_ratio"].mean())
        if not primary.empty
        else None,
        "annualized_return": (1 + approximate_return) ** scale - 1
        if pd.notna(approximate_return) and approximate_return > -1
        else None,
        "sharpe": float(primary_top.mean() / primary_top.std(ddof=0) * np.sqrt(scale))
        if len(primary_top) > 1 and primary_top.std(ddof=0)
        else None,
        "max_drawdown": float(factor_drawdown.min())
        if not factor_drawdown.empty
        else None,
        "bootstrap_ic_ci": _block_ci(ic, policy),
        "bootstrap_top_bottom_spread_ci": _block_ci(spread, policy),
        "quantiles": q_summary.to_dict("records"),
    }
    return summary, daily, quantiles


def _target_events(
    inputs: HistoricalInputs,
    snapshots: dict[pd.Timestamp, pd.DataFrame],
    *,
    interval: int = 60,
    top_n: int = 5,
    weighting: str = "SCORE_WEIGHTED",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_dates = inputs.sessions[::interval]
    session_index = {date: index for index, date in enumerate(inputs.sessions)}
    weights: list[dict[str, Any]] = []
    rebalances: list[dict[str, Any]] = []
    by_date = {
        pd.Timestamp(date): group
        for date, group in inputs.universe.loc[
            inputs.universe["date"].between(inputs.start, inputs.end)
        ].groupby("date", sort=False)
    }
    for signal_date in signal_dates:
        position = session_index[signal_date]
        if position + 1 >= len(inputs.sessions):
            continue
        execution_date = inputs.sessions[position + 1]
        day = by_date.get(signal_date)
        if day is None:
            continue
        canonical = pd.DataFrame(
            canonical_factor_rows(
                inputs.records,
                day,
                signal_date.date().isoformat(),
                rebalance_flag=True,
                data_source=str(PIT_PATH).replace("\\", "/"),
            )
        ).sort_values(["composite_score", "ticker"], ascending=[False, True])
        selected = canonical.head(top_n).copy()
        if weighting == "SCORE_WEIGHTED":
            selected["diagnostic_weight"] = selected["composite_score"] / selected[
                "composite_score"
            ].sum()
        elif weighting == "EQUAL_WEIGHT":
            selected["diagnostic_weight"] = 1 / len(selected)
        else:
            raise HistoricalEvidenceError("UNSUPPORTED_DIAGNOSTIC_WEIGHTING")
        for row in selected.itertuples():
            target = (
                float(row.target_weight)
                if interval == 60 and top_n == 5 and weighting == "SCORE_WEIGHTED"
                else float(row.diagnostic_weight)
            )
            weights.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "ticker": str(row.ticker),
                    "target_weight": target,
                }
            )
            rebalances.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "ticker": str(row.ticker),
                    "rank": int(row.rank),
                    "g2_raw": float(row.g2_raw),
                    "g3_raw": float(row.g3_raw),
                    "normalized_g2": float(row.normalized_g2),
                    "normalized_g3": float(row.normalized_g3),
                    "composite_score": float(row.composite_score),
                    "target_weight": target,
                    "fundamental_period_end": row.fundamental_period_end,
                    "publication_date": row.publication_date,
                    "fundamental_available_date": row.fundamental_available_date,
                    "strategy_id": STRATEGY_ID,
                    "strategy_fingerprint": STRATEGY_FINGERPRINT,
                }
            )
    return pd.DataFrame(weights), pd.DataFrame(rebalances)


def _daily_weights(
    events: pd.DataFrame, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
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


def _strategy_replay(
    inputs: HistoricalInputs,
    events: pd.DataFrame,
    cost: CostModel,
) -> dict[str, pd.DataFrame | pd.Series]:
    weights = _daily_weights(events, inputs.sessions)
    close = inputs.close.reindex(columns=weights.columns)
    results, returns, turnover, orders = canonical_replay(
        close, weights, cost, INITIAL_CASH
    )
    results.index = inputs.sessions
    returns.index = inputs.sessions
    return_frame = pd.DataFrame(
        {
            "date": inputs.sessions,
            "daily_return": returns.to_numpy(),
            "gross_return": results["gross_returns"].to_numpy(),
            "cost_return": results["cost_returns"].to_numpy(),
            "portfolio_value": results["equity"].to_numpy(),
            "turnover": turnover.reindex(inputs.sessions).fillna(0).to_numpy(),
        }
    )
    position_rows: list[dict[str, Any]] = []
    for date, result in results.iterrows():
        equity = float(result["equity"])
        cash = float(result["cash"])
        for ticker in weights.columns:
            value = float(result[f"position:{ticker}"])
            position_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
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
                    "commission": components["buy_fee"]
                    + components["sell_fee"],
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
        "results": results,
    }


def _annual_performance(
    returns: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    data = returns.copy()
    data["year"] = pd.to_datetime(data["date"]).dt.year
    rows = []
    for year, group in data.groupby("year"):
        series = pd.Series(group["daily_return"].to_numpy(), index=group["date"])
        metrics = compute_metrics(series, turnover=group["turnover"])
        costs = transactions.loc[
            pd.to_datetime(transactions["date"]).dt.year.eq(year), "total_cost"
        ].sum()
        rows.append(
            {
                "year": int(year),
                "sessions": len(group),
                "return": metrics["total_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["turnover"],
                "transaction_cost": costs,
                "excess_return": np.nan,
                "benchmark_status": "UNAVAILABLE",
            }
        )
    return pd.DataFrame(rows)


def _concentration_and_attribution(
    replay: dict[str, pd.DataFrame | pd.Series],
    inputs: HistoricalInputs,
    events: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    positions = replay["positions"]
    assert isinstance(positions, pd.DataFrame)
    weights = positions.pivot(index="date", columns="ticker", values="weight")
    absolute = weights.abs()
    daily = pd.DataFrame(
        {
            "position_count": absolute.gt(1e-12).sum(axis=1),
            "largest_position_weight": absolute.max(axis=1),
            "top_3_weight": np.sort(absolute.to_numpy(), axis=1)[:, -3:].sum(axis=1),
            "top_5_weight": np.sort(absolute.to_numpy(), axis=1)[:, -5:].sum(axis=1),
            "hhi": absolute.pow(2).sum(axis=1),
        },
        index=absolute.index,
    )
    price_returns = inputs.close.reindex(columns=weights.columns).pct_change()
    contributions = weights.shift(1).fillna(0).mul(price_returns.fillna(0))
    by_security = contributions.sum().sort_values(ascending=False)
    absolute_total = float(by_security.abs().sum())
    shares = by_security.abs() / absolute_total if absolute_total else by_security * 0
    attribution_rows: list[dict[str, Any]] = [
        {
            "attribution_type": "SECURITY",
            "key": ticker,
            "contribution": contribution,
            "absolute_contribution_share": shares[ticker],
        }
        for ticker, contribution in by_security.items()
    ]
    yearly = contributions.sum(axis=1).groupby(contributions.index.year).sum()
    attribution_rows.extend(
        {
            "attribution_type": "YEAR",
            "key": str(year),
            "contribution": value,
            "absolute_contribution_share": abs(value)
            / float(yearly.abs().sum())
            if yearly.abs().sum()
            else 0.0,
        }
        for year, value in yearly.items()
    )
    execution_dates = pd.DatetimeIndex(sorted(events["execution_date"].unique()))
    labels = pd.Series(index=contributions.index, dtype="object")
    for index, execution_date in enumerate(execution_dates):
        end = (
            execution_dates[index + 1]
            if index + 1 < len(execution_dates)
            else inputs.end + pd.Timedelta(days=1)
        )
        labels.loc[
            (labels.index >= execution_date) & (labels.index < end)
        ] = execution_date.date().isoformat()
    by_rebalance = contributions.sum(axis=1).groupby(labels).sum()
    attribution_rows.extend(
        {
            "attribution_type": "REBALANCE",
            "key": str(key),
            "contribution": value,
            "absolute_contribution_share": abs(value)
            / float(by_rebalance.abs().sum())
            if by_rebalance.abs().sum()
            else 0.0,
        }
        for key, value in by_rebalance.items()
    )
    summary = {
        "formula": "HHI = sum(abs(position_weight)^2)",
        "average_position_count": float(daily["position_count"].mean()),
        "largest_position_weight": float(daily["largest_position_weight"].max()),
        "average_largest_position_weight": float(
            daily["largest_position_weight"].mean()
        ),
        "average_top_3_weight": float(daily["top_3_weight"].mean()),
        "average_top_5_weight": float(daily["top_5_weight"].mean()),
        "average_hhi": float(daily["hhi"].mean()),
        "top_1_stock_contribution_share": float(shares.iloc[0])
        if not shares.empty
        else None,
        "top_3_stock_contribution_share": float(shares.iloc[:3].sum())
        if not shares.empty
        else None,
    }
    return summary, pd.DataFrame(attribution_rows)


def _sensitivity(
    inputs: HistoricalInputs,
    snapshots: dict[pd.Timestamp, pd.DataFrame],
    cost: CostModel,
) -> pd.DataFrame:
    scenarios = [
        ("TOP3", 3, 60, "SCORE_WEIGHTED"),
        ("FROZEN_TOP5_REB60_SCORE", 5, 60, "SCORE_WEIGHTED"),
        ("TOP10", 10, 60, "SCORE_WEIGHTED"),
        ("REB40", 5, 40, "SCORE_WEIGHTED"),
        ("REB80", 5, 80, "SCORE_WEIGHTED"),
        ("EQUAL_WEIGHT", 5, 60, "EQUAL_WEIGHT"),
    ]
    rows = []
    for name, top_n, interval, weighting in scenarios:
        events, _ = _target_events(
            inputs,
            snapshots,
            interval=interval,
            top_n=top_n,
            weighting=weighting,
        )
        replay = _strategy_replay(inputs, events, cost)
        returns = replay["returns"]
        assert isinstance(returns, pd.DataFrame)
        metrics = compute_metrics(returns["daily_return"])
        rows.append(
            {
                "scenario": name,
                "top_n": top_n,
                "rebalance_sessions": interval,
                "weighting": weighting,
                "diagnostic_only": True,
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def classify_historical_evidence(
    *,
    policy: dict[str, Any],
    pit: dict[str, Any],
    sessions: int,
    composite: dict[str, Any],
    strategy: dict[str, Any],
    annual: pd.DataFrame,
    concentration: dict[str, Any],
    sensitivity: pd.DataFrame,
) -> dict[str, Any]:
    rules = policy["classification"]
    if (
        pit["status"] != "PASS"
        or sessions < int(policy["minimum_sessions"])
        or int(composite.get("ic_observations") or 0)
        < int(rules["minimum_composite_ic_observations"])
    ):
        return {
            "classification": "INSUFFICIENT_DATA",
            "policy_version": policy["policy_version"],
            "score": None,
            "checks": {},
        }
    positive_year_ratio = float((annual["return"] > 0).mean())
    positive_sensitivity_ratio = float((sensitivity["total_return"] > 0).mean())
    checks = {
        "composite_ic": (composite.get("rank_ic") or -np.inf)
        >= float(rules["composite_ic"]),
        "ic_positive_ratio": (composite.get("ic_positive_ratio") or -np.inf)
        >= float(rules["ic_positive_ratio"]),
        "top_bottom_spread": (composite.get("top_bottom_spread") or -np.inf)
        > float(rules["top_bottom_spread"]),
        "positive_ic_year_ratio": (
            composite.get("positive_ic_year_ratio") or -np.inf
        )
        >= float(rules["positive_ic_year_ratio"]),
        "net_total_return": strategy["total_return"] > 0,
        "sharpe": strategy["sharpe"] >= float(rules["sharpe"]),
        "max_drawdown": strategy["max_drawdown"]
        >= float(rules["max_drawdown_floor"]),
        "positive_year_ratio": positive_year_ratio
        >= float(rules["positive_year_ratio"]),
        "cost_adjusted_return": strategy["total_return"] > 0,
        "top1_contribution": (
            concentration["top_1_stock_contribution_share"] or np.inf
        )
        <= float(rules["maximum_top1_contribution"]),
        "top3_contribution": (
            concentration["top_3_stock_contribution_share"] or np.inf
        )
        <= float(rules["maximum_top3_contribution"]),
        "sensitivity": positive_sensitivity_ratio
        >= float(rules["minimum_positive_sensitivity_ratio"]),
    }
    score = sum(checks.values()) / len(checks)
    adverse = (
        strategy["sharpe"] < 0
        or (
            strategy["total_return"] <= 0
            and (composite.get("rank_ic") or 0) <= 0
        )
        or (
            (composite.get("rank_ic") or 0) <= 0
            and (composite.get("top_bottom_spread") or 0) <= 0
        )
    )
    classification = (
        "ADVERSE"
        if adverse
        else (
            "HISTORICALLY_SUPPORTIVE"
            if score >= float(rules["supportive_score_ratio"])
            else "MIXED"
        )
    )
    return {
        "classification": classification,
        "policy_version": policy["policy_version"],
        "score": score,
        "checks": checks,
        "positive_year_ratio": positive_year_ratio,
        "positive_sensitivity_ratio": positive_sensitivity_ratio,
        "benchmark_check": "UNAVAILABLE_NOT_IMPUTED",
    }


def _factor_charts(
    output: Path,
    validations: dict[str, dict[str, Any]],
    daily: pd.DataFrame,
    quantiles: pd.DataFrame,
    policy: dict[str, Any],
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    primary = int(policy["primary_horizon"])
    names = {"G2": "g2", "G3": "g3", "COMPOSITE": "composite"}
    generated: list[str] = []

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output / name, dpi=140, bbox_inches="tight")
        plt.close()
        generated.append(name)

    for factor, slug in names.items():
        frame = daily.loc[
            daily["factor"].eq(factor) & daily["horizon"].eq(primary)
        ]
        if not frame.empty:
            plt.figure(figsize=(9, 4.5))
            plt.plot(pd.to_datetime(frame["date"]), frame["ic"], linewidth=1)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.title(f"{factor} {primary}D Rank IC")
            plt.ylabel("Spearman IC")
            save(f"0{list(names).index(factor) + 1}_{slug}_ic_timeseries.png")
        quantile = quantiles.loc[
            quantiles["factor"].eq(factor) & quantiles["horizon"].eq(primary)
        ]
        if not quantile.empty:
            summary = quantile.groupby("quantile")["mean_forward_return"].mean()
            plt.figure(figsize=(7, 4.5))
            plt.bar([f"Q{int(item)}" for item in summary.index], summary.values)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.title(f"{factor} {primary}D Quantile Return")
            save(f"0{list(names).index(factor) + 4}_{slug}_quantile_return.png")
        if not frame.empty and frame["top_bottom_spread"].notna().any():
            plt.figure(figsize=(9, 4.5))
            plt.plot(
                pd.to_datetime(frame["date"]),
                frame["top_bottom_spread"],
                linewidth=1,
            )
            plt.axhline(0, color="black", linewidth=0.8)
            plt.title(f"{factor} {primary}D Top-Bottom Spread")
            save(f"0{list(names).index(factor) + 7}_{slug}_top_bottom_spread.png")
    comparison = pd.DataFrame(
        {
            factor: frame.set_index(pd.to_datetime(frame["date"]))["ic"]
            for factor, frame in daily.loc[daily["horizon"].eq(primary)].groupby(
                "factor"
            )
        }
    )
    if not comparison.empty:
        comparison.plot(figsize=(9, 4.5), linewidth=1)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("Factor Rank IC Comparison")
        plt.ylabel("Spearman IC")
        save("10_factor_ic_comparison.png")
    annual_rows = []
    for factor, validation in validations.items():
        annual_rows.extend(
            {"factor": factor, **row} for row in validation["annual_ic"]
        )
    annual = pd.DataFrame(annual_rows)
    if not annual.empty:
        annual.pivot(index="year", columns="factor", values="mean").plot.bar(
            figsize=(9, 4.5)
        )
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("Annual Factor IC Consistency")
        save("11_factor_annual_consistency.png")
    coverage = daily.loc[daily["horizon"].eq(primary)].pivot(
        index="date", columns="factor", values="coverage"
    )
    if not coverage.empty:
        coverage.index = pd.to_datetime(coverage.index)
        coverage.plot(figsize=(9, 4.5), linewidth=1)
        plt.ylim(0, 1)
        plt.title("Factor Coverage")
        save("12_factor_coverage.png")
    return generated


def _factor_chart_interpretations(
    output: Path, summaries: dict[str, dict[str, Any]], charts: list[str]
) -> None:
    """Write plain-language evidence for every generated factor chart."""

    interpretations = {}
    for chart in charts:
        factor = next(
            (
                name
                for name, slug in {
                    "G2": "g2",
                    "G3": "g3",
                    "COMPOSITE": "composite",
                }.items()
                if f"_{slug}_" in chart
            ),
            None,
        )
        summary = summaries.get(factor, {}) if factor else {}
        interpretations[chart.removesuffix(".png")] = {
            "chart_id": chart.removesuffix(".png"),
            "title": chart.removesuffix(".png").replace("_", " ").title(),
            "what_it_is": "A descriptive historical factor diagnostic over PIT-aligned observations.",
            "how_to_read": "Higher IC, quantile return, or top-minus-bottom spread indicates stronger positive cross-sectional association; zero is neutral.",
            "observations": [],
            "key_numbers": {
                "rank_ic": summary.get("rank_ic"),
                "ic_positive_ratio": summary.get("ic_positive_ratio"),
                "top_bottom_spread": summary.get("top_bottom_spread"),
            },
            "interpretation": "The values describe past association only and do not establish future profitability.",
            "warning": "Coverage and survivorship limitations are recorded in historical_evidence_inventory.json.",
        }
    _write_json(output / "factor_chart_interpretations.json", interpretations)


def _strategy_summary(
    replay: dict[str, pd.DataFrame | pd.Series],
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    returns = replay["returns"]
    assert isinstance(returns, pd.DataFrame)
    data = PerformanceData.from_frames(
        "BACKTEST",
        returns[["date", "daily_return"]],
        positions=positions,
        transactions=transactions,
        source={"historical_evidence": True},
    )
    performance = calculate_metrics(data)
    values = {
        name: item.value for name, item in performance.metrics.items()
    }
    return values, performance.to_dict()


def _historical_report(payload: dict[str, Any]) -> str:
    inventory = payload["data_coverage"]
    factors = payload["factor_evidence"]
    strategy = payload["strategy_evidence"]
    annual = payload["annual_performance"]
    classification = payload["historical_evidence"]["classification"]
    sections = [
        ("01 Executive Summary", [
            f"Historical Evidence = {classification}",
            f"Historical OOS = {payload['historical_oos']['status']}",
            f"Fresh OOS = {payload['fresh_oos']['fresh_oos_status']}",
            f"Production Ready = {'YES' if payload['production_gate']['production_ready'] else 'NO'}",
        ]),
        ("02 Strategy Definition", [
            "G2 Operating Income YoY + G3 EPS YoY, 50/50; Top5; Score Weighted; REB60.",
            f"Fingerprint = {payload['strategy_fingerprint']}",
        ]),
        ("03 Historical Data Coverage", [
            f"{inventory['usable_start']} → {inventory['usable_end']} | {inventory['trading_sessions']} sessions | {inventory['years_covered']:.2f} years",
            "Universe membership basis is current-listed-only; survivorship bias is a known limitation.",
        ]),
        ("04 PIT / Lookahead Audit", [
            f"Status = {payload['pit_audit']['status']}",
            f"Leakage = {payload['pit_audit']['leakage']}",
        ]),
        ("05 G2 Factor Evidence", [json.dumps(factors["G2"], ensure_ascii=False, default=_json_default)]),
        ("06 G3 Factor Evidence", [json.dumps(factors["G3"], ensure_ascii=False, default=_json_default)]),
        ("07 Composite Factor Evidence", [json.dumps(factors["COMPOSITE"], ensure_ascii=False, default=_json_default)]),
        ("08 Quantile Analysis", ["Q1–Q5 results are stored in factor validation CSV files; monotonicity is diagnostic, not a pass condition."]),
        ("09 Canonical Frozen Strategy Backtest", [
            f"Total Return = {strategy.get('total_return')}",
            f"CAGR = {strategy.get('cagr')}",
            f"Sharpe = {strategy.get('sharpe')}",
            f"Sortino = {strategy.get('sortino')}",
        ]),
        ("10 Benchmark Comparison", ["UNAVAILABLE — the repository explicitly has no canonical market benchmark; no proxy was fabricated."]),
        ("11 Drawdown", [f"Max Drawdown = {strategy.get('max_drawdown')}"]),
        ("12 Rolling Stability", ["Rolling strategy metrics are in the performance tear sheet; rolling factor IC is in factor JSON and charts."]),
        ("13 Year-by-Year Robustness", [json.dumps(annual, ensure_ascii=False, default=_json_default)]),
        ("14 Market Regime Robustness", ["UNAVAILABLE — regime labels require a canonical market series."]),
        ("15 Turnover & Cost", [
            f"Turnover = {strategy.get('turnover')}",
            f"Transaction Cost = {strategy.get('transaction_cost')}",
            f"Cost Drag = {strategy.get('cost_drag')}",
        ]),
        ("16 Concentration", [json.dumps(payload["concentration"], ensure_ascii=False, default=_json_default)]),
        ("17 Return Attribution", ["Security, year, and rebalance contribution rows are stored in attribution.csv."]),
        ("18 Sensitivity Diagnostics", ["Top-N, REB40/60/80, and equal/score weight results are diagnostics only and never alter the frozen strategy."]),
        ("19 Historical OOS Status", ["INSUFFICIENT_DATA — no immutable evidence proves an untouched historical selection window."]),
        ("20 Fresh OOS Status", [
            f"{payload['fresh_oos']['fresh_oos_status']} — {payload['fresh_oos']['reason']}"
        ]),
        ("21 Limitations", inventory["warnings"]),
        ("22 Historical Evidence Conclusion", [
            f"{classification} describes retrospective feasibility only. It is not proof of future edge and does not change the production gate."
        ]),
    ]
    lines = ["# Historical Strategy Validation Report", ""]
    for title, content in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {item}" for item in content)
        lines.append("")
    return "\n".join(lines)


def generate_historical_evidence(root: str | Path) -> dict[str, Any]:
    """Generate deterministic historical robustness evidence and source binding."""

    root = Path(root).resolve()
    spec = FundamentalStrategySpec()
    spec.validate()
    policy = load_policy(root)
    output = root / EVIDENCE_DIR
    output.mkdir(parents=True, exist_ok=True)
    inventory = historical_inventory(root, policy)
    _write_json(output / "historical_evidence_inventory.json", inventory)
    inputs = _load_inputs(root, policy)
    snapshots, observations = _factor_snapshots(inputs, policy)
    observations.to_parquet(output / "historical_pit_observations.parquet", index=False)
    pit = pit_audit(inputs.records, observations)
    _write_json(output / "historical_pit_audit.json", pit)
    if pit["status"] != "PASS":
        raise HistoricalEvidenceError("BLOCKED_PIT")

    factor_definitions = {
        "G2": "g2_raw",
        "G3": "g3_raw",
        "COMPOSITE": "composite_score",
    }
    factor_summaries: dict[str, dict[str, Any]] = {}
    daily_frames = []
    quantile_frames = []
    for factor, column in factor_definitions.items():
        summary, daily, quantiles = _factor_validation(
            factor, column, snapshots, inputs, policy
        )
        factor_summaries[factor] = summary
        daily_frames.append(daily)
        quantile_frames.append(quantiles)
        slug = factor.lower()
        _write_json(output / f"factor_{slug}_validation.json", summary)
        daily.to_csv(output / f"factor_{slug}_validation.csv", index=False)
    factor_daily = pd.concat(daily_frames, ignore_index=True)
    factor_quantiles = pd.concat(quantile_frames, ignore_index=True)
    factor_daily.to_csv(output / "factor_validation.csv", index=False)
    factor_quantiles.to_csv(output / "factor_quantile_validation.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                **{factor: factor_summaries[factor].get(key) for factor in factor_definitions},
            }
            for metric, key in (
                ("Rank IC", "rank_ic"),
                ("IC IR", "ic_information_ratio"),
                ("Positive IC %", "ic_positive_ratio"),
                ("Top-Bottom Spread", "top_bottom_spread"),
                ("Annualized Return", "annualized_return"),
                ("Sharpe", "sharpe"),
                ("Max Drawdown", "max_drawdown"),
                ("Turnover", "turnover"),
            )
        ]
    )
    comparison.to_csv(output / "factor_comparison.csv", index=False)
    charts = _factor_charts(
        root / NAMESPACE / "performance/charts/factors",
        factor_summaries,
        factor_daily,
        factor_quantiles,
        policy,
    )
    _factor_chart_interpretations(
        root / NAMESPACE / "performance/charts/factors", factor_summaries, charts
    )

    cost = CostModel()
    events, rebalances = _target_events(inputs, snapshots)
    replay = _strategy_replay(inputs, events, cost)
    returns = replay["returns"]
    positions = replay["positions"]
    transactions = replay["transactions"]
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(positions, pd.DataFrame)
    assert isinstance(transactions, pd.DataFrame)
    returns_path = output / "canonical_backtest_returns.parquet"
    positions_path = output / "canonical_backtest_positions.parquet"
    rebalances_path = output / "canonical_backtest_rebalances.parquet"
    transactions_path = output / "canonical_backtest_transactions.parquet"
    benchmark_path = output / "canonical_benchmark_returns.parquet"
    returns.to_parquet(returns_path, index=False)
    positions.to_parquet(positions_path, index=False)
    rebalances.to_parquet(rebalances_path, index=False)
    transactions.to_parquet(transactions_path, index=False)
    pd.DataFrame(columns=["date", "daily_return"]).to_parquet(
        benchmark_path, index=False
    )

    annual = _annual_performance(returns, transactions)
    annual.to_csv(output / "annual_performance.csv", index=False)
    regime = pd.DataFrame(
        [
            {
                "Regime": "UNAVAILABLE",
                "Sessions": 0,
                "Return": np.nan,
                "Sharpe": np.nan,
                "Max DD": np.nan,
                "Win Rate": np.nan,
                "Reason": "NO_CANONICAL_MARKET_BENCHMARK_IN_REPOSITORY",
            }
        ]
    )
    regime.to_csv(output / "regime_performance.csv", index=False)
    concentration, attribution = _concentration_and_attribution(
        replay, inputs, events
    )
    attribution.to_csv(output / "attribution.csv", index=False)
    sensitivity = _sensitivity(inputs, snapshots, cost)
    sensitivity.to_csv(output / "sensitivity.csv", index=False)
    strategy, detailed_performance = _strategy_summary(
        replay, positions, transactions
    )
    classification = classify_historical_evidence(
        policy=policy,
        pit=pit,
        sessions=len(inputs.sessions),
        composite=factor_summaries["COMPOSITE"],
        strategy=strategy,
        annual=annual,
        concentration=concentration,
        sensitivity=sensitivity,
    )
    gate_path = root / NAMESPACE / "production_final_gate.json"
    gate = (
        json.loads(gate_path.read_text(encoding="utf-8"))
        if gate_path.exists()
        else {}
    )
    fresh = fresh_oos_audit()
    historical = {
        "strategy_id": STRATEGY_ID,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "generated_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "data_coverage": inventory,
        "pit_audit": pit,
        "factor_evidence": factor_summaries,
        "factor_comparison": comparison.to_dict("records"),
        "strategy_evidence": strategy,
        "performance_details": detailed_performance,
        "annual_performance": annual.to_dict("records"),
        "market_regime": {
            "status": "UNAVAILABLE",
            "reason": "NO_CANONICAL_MARKET_BENCHMARK_IN_REPOSITORY",
        },
        "benchmark": {
            "status": "UNAVAILABLE",
            "benchmark_id": None,
            "reason": "NO_CANONICAL_MARKET_BENCHMARK_IN_REPOSITORY",
        },
        "concentration": concentration,
        "attribution": {
            "status": "AVAILABLE",
            "rows": len(attribution),
            "source": "attribution.csv",
        },
        "sensitivity": {
            "status": "AVAILABLE",
            "diagnostic_only": True,
            "rows": sensitivity.to_dict("records"),
        },
        "statistical_uncertainty_analysis": "AVAILABLE",
        "historical_evidence": classification,
        "historical_oos": {
            "status": "INSUFFICIENT_DATA",
            "reason": "NO_IMMUTABLE_CONTAMINATION_EVIDENCE_FOR_UNTOUCHED_WINDOW",
            "retrospective_relabeling": False,
        },
        "fresh_oos": fresh,
        "production_gate": {
            "status": gate.get("production_eligibility", "BLOCKED"),
            "reason": gate.get("production_block_reason", "INSUFFICIENT_OOS"),
            "production_ready": bool(gate.get("production_ready", False)),
        },
        "factor_charts": charts,
    }
    _write_json(output / "historical_strategy_validation.json", historical)
    (output / "historical_strategy_validation_report.md").write_text(
        _historical_report(historical), encoding="utf-8"
    )

    artifact_paths = {
        path.name: path
        for path in (
            returns_path,
            positions_path,
            rebalances_path,
            transactions_path,
            benchmark_path,
        )
    }
    manifest = {
        "manifest_version": "canonical-historical-backtest-v1",
        "strategy_id": STRATEGY_ID,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "strategy_config_hash": sha256_payload(spec.as_dict()),
        "data_snapshot_hash": sha256_payload(
            {
                str(PIT_PATH): _file_sha(root / PIT_PATH),
                str(PRICE_PATH): _file_sha(root / PRICE_PATH),
                str(UNIVERSE_PATH): _file_sha(root / UNIVERSE_PATH),
            }
        ),
        "universe_hash": _file_sha(root / UNIVERSE_PATH),
        "cost_model_hash": sha256_payload(cost.summary()),
        "cost_model": cost.summary(),
        "benchmark_id": None,
        "benchmark_status": "UNAVAILABLE",
        "benchmark_reason": "NO_CANONICAL_MARKET_BENCHMARK_IN_REPOSITORY",
        "start_date": inputs.start.date().isoformat(),
        "end_date": inputs.end.date().isoformat(),
        "trading_sessions": len(inputs.sessions),
        "generated_at": historical["generated_at"],
        "code_commit": _git_commit(root),
        "artifact_sha256": {
            name: _file_sha(path) for name, path in artifact_paths.items()
        },
        "pit_audit_status": pit["status"],
        "historical_label": "HISTORICAL_ROBUSTNESS",
        "historical_oos_status": "INSUFFICIENT_DATA",
        "fresh_oos_status": fresh["fresh_oos_status"],
    }
    _write_json(output / "canonical_backtest_manifest.json", manifest)
    source_manifest = {
        "strategy_id": STRATEGY_ID,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "periods": {
            "BACKTEST": {
                "returns": str(EVIDENCE_DIR / returns_path.name).replace("\\", "/"),
                "positions": str(EVIDENCE_DIR / positions_path.name).replace(
                    "\\", "/"
                ),
                "transactions": str(EVIDENCE_DIR / transactions_path.name).replace(
                    "\\", "/"
                ),
                "canonical_manifest": str(
                    EVIDENCE_DIR / "canonical_backtest_manifest.json"
                ).replace("\\", "/"),
                "benchmark_id": None,
                "benchmark_source": None,
                "benchmark_reason": "NO_CANONICAL_MARKET_BENCHMARK_IN_REPOSITORY",
                "historical_label": "HISTORICAL_ROBUSTNESS",
            },
            "HISTORICAL_OOS": {
                "status": "INSUFFICIENT_DATA",
                "reason": "NO_IMMUTABLE_CONTAMINATION_EVIDENCE_FOR_UNTOUCHED_WINDOW",
            },
            "FRESH_OOS": {
                "status": fresh["fresh_oos_status"],
                "reason": fresh["reason"],
            },
        },
    }
    _write_json(root / NAMESPACE / "performance_source_manifest.json", source_manifest)
    FundamentalStrategySpec().validate()
    return historical


__all__ = [
    "HistoricalEvidenceError",
    "classify_historical_evidence",
    "generate_historical_evidence",
    "historical_inventory",
    "load_policy",
    "pit_audit",
]

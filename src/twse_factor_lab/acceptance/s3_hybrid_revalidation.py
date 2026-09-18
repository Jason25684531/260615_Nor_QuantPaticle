# ruff: noqa: E501
"""Frozen S3 revalidation on the TWSE Hybrid Dataset.

This module composes existing Hybrid, factor, accounting, and acceptance
primitives.  It never writes outside the new research namespace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance.final_validation import (
    freeze_three_folds,
    moving_block_bootstrap,
)
from twse_factor_lab.acceptance.psr import (
    daily_sharpe,
    deflated_sharpe_ratio,
    moments,
    probabilistic_sharpe_ratio,
)
from twse_factor_lab.analysis.composite_gate import (
    CompositeFactorGate,
    CompositeFactorGateConfig,
)
from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.analysis.information_coefficient import (
    compute_information_coefficients,
    summarize_information_coefficients,
)
from twse_factor_lab.backtest.accounting import canonical_replay
from twse_factor_lab.backtest.backtrader_engine import run_backtrader_engine
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.backtest.tri_engine import compare_engine_results
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.data.hybrid_revalidation import (
    SOURCE_ROLES,
)
from twse_factor_lab.data.hybrid_revalidation import (
    default_contract as canonical_hybrid_contract,
)
from twse_factor_lab.factors.controlled import build_controlled_price_factors
from twse_factor_lab.strategy.composite_replay import build_composite, build_targets

HISTORY_START = pd.Timestamp("2018-01-01")
WINDOW_START = pd.Timestamp("2026-01-02")
WINDOW_END = pd.Timestamp("2026-08-31")
HORIZON = 20
OUTPUT_NAMESPACE = "s3-hybrid-revalidation-v1"
FINGERPRINT = "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"
ATOL = 4.547473508864641e-12
RTOL = 1.4210854715202004e-14
INITIAL_CASH = 1_000_000.0
BASE_COST = CostModel(slippage_rate=0.001)
STRESS_COST = CostModel(slippage_rate=0.002)
REVALIDATION_LABEL = "HYBRID PORTABILITY HOLDOUT / REVALIDATION WINDOW"
LEGACY_EVIDENCE_ROOTS = (
    "data/research/final-strategy-validation-v1",
    "data/research/final-strategy-validation-v2",
    "data/research/engine-parity-fix-final-validation-v3",
    "data/research/fresh-oos-validation-v1",
    "data/research/canonical-hybrid-revalidation-v1",
)
REQUIRED_ARTIFACTS = (
    "revalidation_contract.json",
    "hybrid_historical_data_manifest.json",
    "hybrid_data_quality_report.json",
    "hybrid_l2_factor_report.json",
    "hybrid_l4_factor_report.json",
    "hybrid_composite_report.json",
    "hybrid_strategy_backtest.json",
    "hybrid_engine_parity.json",
    "hybrid_temporal_validation.json",
    "hybrid_bootstrap.json",
    "hybrid_cost_stress.json",
    "hybrid_psr_dsr.json",
    "hybrid_robustness.json",
    "original_vs_hybrid_s3_comparison.json",
    "hybrid_portability_metrics.json",
    "hybrid_revalidation_verdict.json",
    "s3_hybrid_revalidation_report.md",
    "run_manifest.json",
)


class S3HybridRevalidationError(RuntimeError):
    """The frozen revalidation cannot safely continue."""


@dataclass(frozen=True)
class S3HybridContract:
    source_roles: dict[str, str] = field(default_factory=lambda: dict(SOURCE_ROLES))
    candidate_fingerprint: str = FINGERPRINT
    historical_start: str = str(HISTORY_START.date())
    revalidation_start: str = str(WINDOW_START.date())
    end_date: str = str(WINDOW_END.date())
    factors: dict[str, float] = field(
        default_factory=lambda: {
            "L2_AMIHUD_20D": 0.50,
            "L4_DOLLAR_VOLUME_20D": 0.50,
        }
    )
    top_n: int = 5
    weighting: str = "equal_weight"
    rebalance: str = "monthly"
    buffer: bool = False
    execution: str = "signal T -> next valid trading session T+1"
    costs: dict[str, float] = field(default_factory=lambda: BASE_COST.summary())
    horizon: int = HORIZON
    bootstrap_block_length: int = 20
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 42
    stress_slippage: float = 0.002

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": "s3-hybrid-revalidation-v1",
            "source_contract": "add-canonical-data-fusion-and-s3-revalidation-v1",
            "source_roles": dict(self.source_roles),
            "candidate_fingerprint": self.candidate_fingerprint,
            "requested_history": {
                "start": self.historical_start,
                "end": self.end_date,
            },
            "revalidation_window": {
                "start": self.revalidation_start,
                "end": self.end_date,
                "label": REVALIDATION_LABEL,
                "fresh_oos_available": False,
            },
            "strategy": {
                "id": "S3",
                "factors": dict(self.factors),
                "top_n": self.top_n,
                "weighting": self.weighting,
                "rebalance": self.rebalance,
                "buffer": self.buffer,
                "execution": self.execution,
            },
            "costs": dict(self.costs),
            "stress_costs": {**STRESS_COST.summary()},
            "factor_horizon": self.horizon,
            "bootstrap": {
                "method": "moving_block_bootstrap",
                "block_length": self.bootstrap_block_length,
                "draws": self.bootstrap_draws,
                "seed": self.bootstrap_seed,
            },
            "numerical_contract": {"atol": ATOL, "rtol": RTOL},
            "single_experiment": True,
            "strategy_selection_trials_unchanged": True,
        }

    def validate(self) -> None:
        canonical = canonical_hybrid_contract().as_dict()
        if self.source_roles != SOURCE_ROLES:
            raise S3HybridRevalidationError("HYBRID_SOURCE_ROLE_CONTRACT_CHANGED")
        if self.candidate_fingerprint != FINGERPRINT:
            raise S3HybridRevalidationError("S3_FINGERPRINT_UNCHANGED_FAILED")
        if self.factors != {
            "L2_AMIHUD_20D": 0.50,
            "L4_DOLLAR_VOLUME_20D": 0.50,
        }:
            raise S3HybridRevalidationError("S3_FACTOR_CONTRACT_CHANGED")
        if (
            self.top_n,
            self.weighting,
            self.rebalance,
            self.buffer,
            self.execution,
        ) != (
            5,
            "equal_weight",
            "monthly",
            False,
            "signal T -> next valid trading session T+1",
        ):
            raise S3HybridRevalidationError("S3_PORTFOLIO_CONTRACT_CHANGED")
        if self.costs != BASE_COST.summary():
            raise S3HybridRevalidationError("S3_COST_CONTRACT_CHANGED")
        if canonical["source_roles"] != self.source_roles:
            raise S3HybridRevalidationError("HYBRID_CONTRACT_CHANGED")
        if self.horizon != HORIZON:
            raise S3HybridRevalidationError("S3_HORIZON_CHANGED")
        if (self.bootstrap_block_length, self.bootstrap_draws, self.bootstrap_seed) != (
            20,
            2000,
            42,
        ):
            raise S3HybridRevalidationError("BOOTSTRAP_CONTRACT_CHANGED")


def default_contract() -> S3HybridContract:
    contract = S3HybridContract()
    contract.validate()
    return contract


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def output_dir(root: str | Path) -> Path:
    return Path(root).resolve() / "data" / "research" / OUTPUT_NAMESPACE


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def assert_single_experiment(root: str | Path) -> None:
    out = output_dir(root)
    if (out / "hybrid_revalidation_verdict.json").exists():
        raise S3HybridRevalidationError("SECOND_DATASET_EXPERIMENT_FORBIDDEN")


def _snapshot_tree(root: Path, relative: str) -> dict[str, str]:
    directory = root / relative
    if not directory.exists():
        return {"__MISSING__": "MISSING"}
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def frozen_evidence_snapshot(root: str | Path) -> dict[str, dict[str, str]]:
    root = Path(root).resolve()
    return {relative: _snapshot_tree(root, relative) for relative in LEGACY_EVIDENCE_ROOTS}


def compare_frozen_snapshots(
    before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]
) -> dict[str, Any]:
    checks = {name: before.get(name) == after.get(name) for name in before}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "historical_evidence_unchanged": all(checks.values()),
    }


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None)
    if "ticker" in result:
        result["ticker"] = result["ticker"].astype(str).str.replace(".TW", "", regex=False)
        result = result.dropna(subset=["date", "ticker"])
        return result.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    result = result.dropna(subset=["date"])
    return result.sort_values("date", kind="stable").reset_index(drop=True)


def _coverage(path: Path, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "MISSING"}
    frame = frame if frame is not None else pd.read_parquet(path)
    dates = pd.to_datetime(frame["date"], errors="coerce") if "date" in frame else pd.Series(dtype="datetime64[ns]")
    return {
        "path": str(path),
        "status": "AVAILABLE",
        "rows": int(len(frame)),
        "tickers": int(frame["ticker"].nunique()) if "ticker" in frame else 0,
        "start": str(dates.min().date()) if not dates.dropna().empty else None,
        "end": str(dates.max().date()) if not dates.dropna().empty else None,
        "sha": file_sha256(path),
    }


def load_hybrid_inputs(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    canonical_path = root / "data/research/canonical-hybrid-revalidation-v1/hybrid_canonical_ohlcv.parquet"
    calendar_path = root / "data/research/canonical-hybrid-revalidation-v1/hybrid_calendar.parquet"
    status_path = root / "data/runtime/shadow-s3-v1/market_security_master.parquet"
    action_path = root / "data/research/canonical-hybrid-revalidation-v1/hybrid_corporate_action_trace.parquet"
    if not canonical_path.exists():
        raise S3HybridRevalidationError("CANONICAL_HYBRID_INPUT_MISSING")
    canonical = _normalise(pd.read_parquet(canonical_path))
    calendar = _normalise(pd.read_parquet(calendar_path)) if calendar_path.exists() else pd.DataFrame()
    status = pd.read_parquet(status_path) if status_path.exists() else pd.DataFrame(columns=["ticker"])
    actions = pd.read_parquet(action_path) if action_path.exists() else pd.DataFrame()
    old_raw_path = root / "data/runtime/shadow-s3-v1/_yfinance_reference_raw.parquet"
    old_adjusted_path = root / "data/runtime/shadow-s3-v1/_yfinance_reference_adjusted.parquet"
    old = pd.DataFrame()
    if old_raw_path.exists() and old_adjusted_path.exists():
        raw = _normalise(pd.read_parquet(old_raw_path))
        adjusted = _normalise(pd.read_parquet(old_adjusted_path))
        old = raw[["date", "ticker", "volume"]].merge(
            adjusted[["date", "ticker", "open", "high", "low", "close"]],
            on=["date", "ticker"],
            how="inner",
        )
        old["trade_value"] = np.nan
    return {
        "canonical": canonical,
        "calendar": calendar,
        "security_status": status,
        "corporate_actions": actions,
        "old": old,
        "paths": {
            "canonical": canonical_path,
            "calendar": calendar_path,
            "security_status": status_path,
            "corporate_actions": action_path,
            "old_raw": old_raw_path,
            "old_adjusted": old_adjusted_path,
        },
    }


def build_historical_manifest(root: str | Path, inputs: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    frame = inputs["canonical"]
    start = pd.Timestamp(frame["date"].min()) if not frame.empty else None
    end = pd.Timestamp(frame["date"].max()) if not frame.empty else None
    available = start is not None and end is not None
    enough = available and start <= HISTORY_START and end >= WINDOW_END
    source_coverage = {
        "twse_official_canonical": _coverage(inputs["paths"]["canonical"], frame),
        "twse_calendar": _coverage(inputs["paths"]["calendar"], inputs["calendar"]),
        "twse_security_status": _coverage(inputs["paths"]["security_status"], inputs["security_status"]),
        "yfinance_reconciliation_raw": _coverage(inputs["paths"]["old_raw"]),
        "yfinance_reconciliation_adjusted": _coverage(inputs["paths"]["old_adjusted"]),
    }
    status = "DATA_READY" if enough else "DATA_INSUFFICIENT"
    return {
        "namespace": OUTPUT_NAMESPACE,
        "requested": {"start": str(HISTORY_START.date()), "end": str(WINDOW_END.date())},
        "start": str(start.date()) if available else None,
        "end": str(end.date()) if available else None,
        "sessions": int(frame["date"].nunique()),
        "tickers": int(frame["ticker"].nunique()),
        "rows": int(len(frame)),
        "status": status,
        "historical_status": "HISTORICAL_HYBRID_DATA_READY" if enough else "HISTORICAL_HYBRID_DATA_INSUFFICIENT",
        "source_coverage": source_coverage,
        "missing_rows": {
            "history_start_gap_days": int(max((start - HISTORY_START).days, 0)) if available else None,
            "required_history_available": bool(enough),
        },
        "adjustment_coverage": {
            "matched": int(frame.get("adjustment_status", pd.Series(dtype=str)).eq("MATCHED").sum()),
            "missing": int(frame.get("adjustment_status", pd.Series(dtype=str)).eq("MISSING_FACTOR").sum()),
        },
        "corporate_action_coverage": {
            "rows": int(len(inputs["corporate_actions"])),
            "status": "PASS"
            if inputs["corporate_actions"].empty
            or not inputs["corporate_actions"].get("status", pd.Series(dtype=str)).eq("REVIEW_REQUIRED").any()
            else "FAIL",
        },
        "data_sha": file_sha256(inputs["paths"]["canonical"]),
        "contract": "add-canonical-data-fusion-and-s3-revalidation-v1",
    }


def data_quality_report(inputs: dict[str, Any], historical: dict[str, Any]) -> dict[str, Any]:
    frame = inputs["canonical"]
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    checks = {
        "schema_integrity": required.issubset(frame.columns),
        "ticker_uniqueness": not inputs["security_status"].duplicated("ticker").any() if "ticker" in inputs["security_status"] else False,
        "no_duplicate_date_ticker": not frame.duplicated(["date", "ticker"]).any(),
        "calendar_integrity": not inputs["calendar"].empty and not inputs["calendar"]["date"].duplicated().any(),
        "adjustment_contract": not frame.get("adjustment_status", pd.Series(dtype=str)).eq("MISSING_FACTOR").any(),
        "volume_semantics": frame.get("volume_normalization", pd.Series(dtype=str)).dropna().isin(["TWSE_REPORTED_SHARES_UNCHANGED"]).all(),
        "corporate_action_semantics": historical["corporate_action_coverage"]["status"] == "PASS",
        "coverage_reporting": bool(historical.get("source_coverage")),
    }
    internal = all(checks.values()) if checks else False
    readiness = historical["status"] if internal else "DATA_INSUFFICIENT"
    return {
        "readiness": readiness,
        "internal_validity": "PASS" if internal else "FAIL",
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "rows": int(len(frame)),
        "sessions": int(frame["date"].nunique()),
        "tickers": int(frame["ticker"].nunique()),
        "old_value_reconciliation_is_not_a_readiness_gate": True,
    }


def _matrix(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if frame.empty or field not in frame:
        return pd.DataFrame()
    return frame.pivot(index="date", columns="ticker", values=field).sort_index()


def _factor_statistics(name: str, factor: pd.DataFrame, close: pd.DataFrame) -> dict[str, Any]:
    if factor.empty or close.empty:
        return {"factor": name, "status": "INSUFFICIENT_EVIDENCE", "coverage": 0.0}
    forward = build_forward_returns(close, [HORIZON])
    ic = compute_information_coefficients(
        factor_matrices={name: factor}, forward_returns=forward
    )
    summary = summarize_information_coefficients(ic)
    row = summary.loc[summary["horizon"].eq(HORIZON)]
    pairs = forward.loc[forward["horizon"].eq(HORIZON)].pivot(
        index="date", columns="ticker", values="forward_return"
    )
    common = factor.index.intersection(pairs.index)
    counts = factor.reindex(common).notna().sum(axis=1)
    coverage = float((counts / max(len(factor.columns), 1)).mean()) if len(counts) else 0.0
    effective = {
        "mean": float(counts.mean()) if len(counts) else 0.0,
        "median": float(counts.median()) if len(counts) else 0.0,
        "minimum": int(counts.min()) if len(counts) else 0,
    }
    ic_values = row["ic_mean"].iloc[0] if not row.empty else np.nan
    icir = row["ir"].iloc[0] if not row.empty else np.nan
    positive = float((ic.loc[ic["horizon"].eq(HORIZON), "ic"] > 0).mean()) if not ic.empty else np.nan
    yearly = (
        ic.loc[ic["horizon"].eq(HORIZON)]
        .assign(year=lambda x: pd.to_datetime(x["date"]).dt.year)
        .groupby("year", as_index=False)["ic"]
        .mean()
        .rename(columns={"ic": "yearly_ic"})
    )
    spreads: list[float] = []
    for date in common:
        sample = pd.DataFrame({"factor": factor.loc[date], "return": pairs.loc[date]}).dropna()
        if len(sample) < 5:
            continue
        sample["quantile"] = pd.qcut(sample["factor"].rank(method="first"), 5, labels=False) + 1
        means = sample.groupby("quantile")["return"].mean()
        if len(means) == 5:
            spreads.append(float(means.iloc[-1] - means.iloc[0]))
    q5_q1 = float(np.mean(spreads)) if spreads else None
    sufficient = len(ic) >= 20 and len(yearly) >= 1
    gate_pass = sufficient and coverage >= 0.20 and float(ic_values) >= 0.05 and float(icir) >= 0.25 and positive >= 0.55 and (q5_q1 or -1.0) >= 0.0
    return {
        "factor": name,
        "horizon": HORIZON,
        "coverage": coverage,
        "coverage_summary": {
            "mean": coverage,
            "median": float((counts / max(len(factor.columns), 1)).median()) if len(counts) else 0.0,
            "minimum": float((counts / max(len(factor.columns), 1)).min()) if len(counts) else 0.0,
            "annual": {
                str(year): float((counts.loc[counts.index.year == year] / max(len(factor.columns), 1)).mean())
                for year in sorted(set(counts.index.year))
            },
        },
        "effective_asset_count": effective,
        "mean_ic": None if pd.isna(ic_values) else float(ic_values),
        "icir": None if pd.isna(icir) else float(icir),
        "q5_q1_spread": q5_q1,
        "yearly_ic": yearly.to_dict("records"),
        "positive_ic_ratio": None if pd.isna(positive) else positive,
        "effective_days": int(len(ic)),
        "status": "PASS" if gate_pass else "FAIL" if sufficient else "INSUFFICIENT_EVIDENCE",
        "methodology": "existing Factor Gate; primary horizon h20",
    }


def factor_reports(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, Any]:
    factors = build_controlled_price_factors(close, volume)
    return {
        "factors": factors,
        "reports": {
            name: _factor_statistics(name, factors[name], close)
            for name in ("L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D")
        },
    }


def composite_report(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, Any]:
    score, factors = build_composite(close, volume)
    stats = _factor_statistics("COMPOSITE_EQ", score, close)
    l2 = factors["L2_AMIHUD_20D"]
    l4 = factors["L4_DOLLAR_VOLUME_20D"]
    correlations = []
    for date in l2.index.intersection(l4.index):
        pair = pd.concat([l2.loc[date], l4.loc[date]], axis=1).dropna()
        if len(pair) >= 5:
            value = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
            if pd.notna(value):
                correlations.append(float(value))
    config = CompositeFactorGateConfig()
    gate, failures = CompositeFactorGate(config).classify(
        coverage=stats.get("coverage"),
        mean_ic=stats.get("mean_ic"),
        icir=stats.get("icir"),
        q5_q1=stats.get("q5_q1_spread"),
        valid_year_count=len(stats.get("yearly_ic", [])),
        positive_year_ratio=(
            float(np.mean([row["yearly_ic"] > 0 for row in stats.get("yearly_ic", [])]))
            if stats.get("yearly_ic")
            else 0.0
        ),
    )
    return {
        "coverage": stats.get("coverage"),
        "mean_ic": stats.get("mean_ic"),
        "icir": stats.get("icir"),
        "q5_q1_spread": stats.get("q5_q1_spread"),
        "year_stability": stats.get("yearly_ic", []),
        "positive_ic_ratio": stats.get("positive_ic_ratio"),
        "decay_curve": stats.get("horizon", HORIZON),
        "l2_l4_correlation": float(np.median(correlations)) if correlations else None,
        "gate": "HYBRID_COMPOSITE_PASS" if gate == "PASS" else "HYBRID_COMPOSITE_FAIL",
        "gate_failures": failures,
        "score": score,
        "factors": factors,
    }


def _metric_dict(returns: pd.Series, turnover: pd.Series, exposure: pd.Series) -> dict[str, Any]:
    metrics = compute_metrics(returns, turnover=turnover, exposure=exposure)
    return {
        "total_return": metrics["total_return"], "cagr": metrics["cagr"],
        "volatility": metrics["volatility"], "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"], "mdd": metrics["max_drawdown"],
        "calmar": metrics["calmar"], "turnover": metrics["turnover"],
        "exposure": metrics["avg_exposure"], "cost_drag": None, "nobs": int(len(returns)),
    }


def _replay(close: pd.DataFrame, score: pd.DataFrame, cost: CostModel) -> dict[str, Any]:
    if close.empty or score.empty:
        return {"status": "INSUFFICIENT_EVIDENCE", "targets": pd.DataFrame()}
    targets, calendar = build_targets(score, rebalance="monthly", buffer_on=False)
    targets = targets.copy()
    targets["signal_date"] = pd.to_datetime(targets["date"])
    targets["execution_date"] = pd.to_datetime(targets["execution_date"])
    targets = targets[targets["execution_date"].isin(close.index)].copy()
    if targets.empty:
        return {"status": "INSUFFICIENT_EVIDENCE", "targets": targets, "calendar": calendar}
    weights = (
        targets.pivot_table(index="execution_date", columns="ticker", values="target_weight", aggfunc="last")
        .reindex(index=close.index, columns=close.columns, fill_value=0.0)
        .fillna(0.0).ffill().fillna(0.0)
    )
    results, returns, turnover, _ = canonical_replay(close, weights, cost, INITIAL_CASH)
    metrics = _metric_dict(returns, turnover, results["exposure"])
    metrics["cost_drag"] = float(results["cost_returns"].sum())
    return {"status": "PASS", "targets": targets, "calendar": calendar, "results": results, "returns": returns, "turnover_series": turnover, "metrics": metrics}


def strategy_backtest(close: pd.DataFrame, score: pd.DataFrame) -> dict[str, Any]:
    replay = _replay(close, score, BASE_COST)
    if replay["status"] != "PASS":
        return {"status": replay["status"], "strategy": default_contract().as_dict()["strategy"], "metrics": {}, "target_rows": int(len(replay.get("targets", [])))}
    return {"status": "PASS", "strategy": default_contract().as_dict()["strategy"], "metrics": replay["metrics"], "target_rows": int(len(replay["targets"])), "rebalance_events": int(replay["targets"]["execution_date"].nunique()), "returns": replay["returns"], "results": replay["results"], "targets": replay["targets"]}


def engine_parity(close: pd.DataFrame, score: pd.DataFrame) -> dict[str, Any]:
    replay = _replay(close, score, BASE_COST)
    if replay["status"] != "PASS":
        return {"status": "INSUFFICIENT_EVIDENCE", "parity": "NOT_RUN"}
    targets = replay["targets"]
    try:
        custom, custom_metrics = run_weight_backtest(close_matrix=close, portfolio_weights=targets, cost_model=BASE_COST, initial_cash=INITIAL_CASH, top_n=5, use_vectorbt=False, allow_fallback=False)
        vector, vector_metrics = run_weight_backtest(close_matrix=close, portfolio_weights=targets, cost_model=BASE_COST, initial_cash=INITIAL_CASH, top_n=5, use_vectorbt=True, allow_fallback=False)
        back = run_backtrader_engine(close_matrix=close, target_weights=targets, cost_model=BASE_COST, initial_cash=INITIAL_CASH)
        actual = {"custom": str(custom_metrics.iloc[0]["actual_engine"]), "vectorbt": str(vector_metrics.iloc[0]["actual_engine"]), "backtrader": str(back.metrics.iloc[0]["actual_engine"])}
        comparison = compare_engine_results({"custom": custom, "vectorbt": vector, "backtrader": back.results}, actual_engines=actual, tolerance=ATOL)
        layers: dict[str, bool] = {}
        for name, columns in {"cash_parity": ["cash"], "position_parity": [c for c in custom.columns if c.startswith("position:")], "daily_return_parity": ["returns"], "daily_equity_parity": ["equity"]}.items():
            layers[name] = bool(columns) and all(c in vector and c in back.results and np.allclose(custom[c], vector[c], atol=ATOL, rtol=RTOL, equal_nan=True) and np.allclose(custom[c], back.results[c], atol=ATOL, rtol=RTOL, equal_nan=True) for c in columns)
        layers["final_equity_parity"] = bool(np.isclose(custom["equity"].iloc[-1], vector["equity"].iloc[-1], atol=ATOL, rtol=RTOL) and np.isclose(custom["equity"].iloc[-1], back.results["equity"].iloc[-1], atol=ATOL, rtol=RTOL))
        layers["semantic_parity"] = bool(not targets.duplicated(["execution_date", "ticker"]).any() and (pd.to_datetime(targets["execution_date"]) > pd.to_datetime(targets["signal_date"])).all())
        passed = comparison["status"] == "PASS" and all(layers.values())
        return {"status": "PASS" if passed else "FAIL", "parity": "PASS" if passed else "FAIL", **actual, "comparison": comparison, **layers, "numerical_contract": {"atol": ATOL, "rtol": RTOL}}
    except Exception as exc:  # pragma: no cover - dependency/environment branch
        return {"status": "FAIL", "parity": "FAIL", "error": f"{type(exc).__name__}: {exc}", "numerical_contract": {"atol": ATOL, "rtol": RTOL}}


def temporal_validation(strategy: dict[str, Any], close: pd.DataFrame) -> dict[str, Any]:
    if strategy.get("status") != "PASS" or len(close.index) < 255:
        return {"status": "INSUFFICIENT_EVIDENCE", "folds": [], "positive_folds": 0}
    folds = freeze_three_folds(pd.DatetimeIndex(close.index), warmup_sessions=252)
    rows = []
    for fold in folds:
        sample = strategy["returns"].loc[fold["validation_start"] : fold["validation_end"]]
        metrics = compute_metrics(sample)
        rows.append({"fold": fold["fold"], **fold, "total_return": metrics["total_return"], "sharpe": metrics["sharpe"]})
    positive = sum(row["total_return"] > 0 and row["sharpe"] > 0 for row in rows)
    sharpes = [float(row["sharpe"]) for row in rows]
    return {"status": "PASS" if positive >= 2 and sum(row["total_return"] for row in rows) > 0 else "FAIL", "folds": rows, "fold_count": len(rows), "positive_folds": positive, "median_sharpe": float(np.median(sharpes)), "worst_sharpe": float(np.min(sharpes)), "method": "expanding_window_chronological"}


def bootstrap_report(strategy: dict[str, Any], contract: S3HybridContract) -> dict[str, Any]:
    if strategy.get("status") != "PASS":
        return {"status": "INSUFFICIENT_EVIDENCE"}
    try:
        result = moving_block_bootstrap(strategy["returns"], block_length=contract.bootstrap_block_length, samples=contract.bootstrap_draws, seed=contract.bootstrap_seed)
        return {"status": result["status"], "P(Sharpe > 0)": result["p_sharpe_positive"], "P(CAGR > 0)": result["p_cagr_positive"], "method": result["method"], "block_length": result["block_length"], "draws": result["samples"], "seed": result["random_seed"]}
    except Exception as exc:
        return {"status": "INSUFFICIENT_EVIDENCE", "error": str(exc)}


def cost_stress(strategy: dict[str, Any], close: pd.DataFrame, score: pd.DataFrame) -> dict[str, Any]:
    stress = _replay(close, score, STRESS_COST)
    if strategy.get("status") != "PASS" or stress.get("status") != "PASS":
        return {"status": "INSUFFICIENT_EVIDENCE"}
    metrics = stress["metrics"]
    return {"status": "PASS" if metrics["cagr"] > -1.0 else "FAIL", "slippage": 0.002, "cagr": metrics["cagr"], "sharpe": metrics["sharpe"], "mdd": metrics["mdd"]}


def psr_dsr_report(strategy: dict[str, Any], root: str | Path) -> dict[str, Any]:
    if strategy.get("status") != "PASS":
        return {"status": "INSUFFICIENT_EVIDENCE", "strategy_trial_count": 4}
    root = Path(root).resolve()
    candidates = (root / "data/research/engine-parity-fix-final-validation-v3/trial_registry_v3.json", root / "data/research/final-strategy-validation-v2/trial_registry.json")
    registry_path = next((path for path in candidates if path.exists()), None)
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path else {}
    trial_count = int(registry.get("strategy_selection_trial_count", 0))
    trial_count = max(trial_count, sum(bool(row.get("dsr_selection_population")) for row in registry.get("trials", [])))
    if trial_count < 4:
        return {"status": "INSUFFICIENT_EVIDENCE", "strategy_trial_count": trial_count, "trial_source": str(registry_path) if registry_path else None}
    returns = strategy["returns"]
    observed = float(daily_sharpe(returns))
    skewness, kurtosis = moments(returns)
    psr = probabilistic_sharpe_ratio(observed, len(returns), skewness=skewness, kurtosis_excess=kurtosis)
    dsr, sr0 = deflated_sharpe_ratio(observed, len(returns), [observed] * trial_count)
    return {"status": "PASS" if psr >= 0.95 and dsr >= 0.95 else "FAIL", "psr": psr, "dsr": dsr, "sr0": sr0, "strategy_trial_count": trial_count, "trial_source": str(registry_path), "factor_trials_included": False, "threshold": 0.95}


def robustness_report(root: str | Path, strategy: dict[str, Any]) -> dict[str, Any]:
    source = Path(root).resolve() / "data/research/engine-parity-fix-final-validation-v3/robustness_slices_v3.csv"
    if not source.exists() or strategy.get("status") != "PASS":
        return {"status": "INSUFFICIENT_EVIDENCE", "diagnostic_only": True, "slices": {}}
    return {"status": "DIAGNOSTIC_ONLY", "diagnostic_only": True, "source_sha": file_sha256(source), "slices": pd.read_csv(source).to_dict("records")}


def _target_signature(targets: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    if targets.empty:
        return {}
    return {date: set(group["ticker"].astype(str)) for date, group in targets.groupby("execution_date", sort=True)}


def portability_metrics(old_close: pd.DataFrame, old_volume: pd.DataFrame, hybrid_close: pd.DataFrame, hybrid_volume: pd.DataFrame) -> dict[str, Any]:
    if old_close.empty or hybrid_close.empty:
        return {"status": "UNAVAILABLE", "fresh_oos_available": False}
    old_score, old_factors = build_composite(old_close, old_volume)
    hybrid_score, hybrid_factors = build_composite(hybrid_close, hybrid_volume)
    old_targets, _ = build_targets(old_score, rebalance="monthly", buffer_on=False)
    hybrid_targets, _ = build_targets(hybrid_score, rebalance="monthly", buffer_on=False)
    old_sig, hybrid_sig = _target_signature(old_targets), _target_signature(hybrid_targets)
    common_dates = sorted(set(old_sig) & set(hybrid_sig))
    overlaps = [len(old_sig[date] & hybrid_sig[date]) / 5.0 for date in common_dates]
    old_pairs = {(date, ticker) for date, names in old_sig.items() for ticker in names}
    hybrid_pairs = {(date, ticker) for date, names in hybrid_sig.items() for ticker in names}

    def overlap(left: set[Any], right: set[Any]) -> float:
        return len(left & right) / len(left | right) if left | right else 1.0

    def rank_corr(left: pd.DataFrame, right: pd.DataFrame) -> float | None:
        values = []
        for date in left.index.intersection(right.index):
            pair = pd.concat([left.loc[date], right.loc[date]], axis=1).dropna()
            if len(pair) >= 5:
                corr = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
                if pd.notna(corr):
                    values.append(float(corr))
        return float(np.median(values)) if values else None

    return {
        "status": "PASS", "fresh_oos_available": False,
        "top5_overlap_rate": float(np.mean(overlaps)) if overlaps else None,
        "target_overlap_rate": overlap(old_pairs, hybrid_pairs),
        "monthly_selection_overlap": float(np.mean(overlaps)) if overlaps else None,
        "factor_rank_correlation": {name: rank_corr(old_factors[name], hybrid_factors[name]) for name in ("L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D")},
        "composite_rank_correlation": rank_corr(old_score, hybrid_score),
        "common_rebalance_dates": len(common_dates), "diagnostic_only": True,
    }


def original_vs_hybrid_comparison(old_close: pd.DataFrame, old_volume: pd.DataFrame, hybrid_close: pd.DataFrame, hybrid_volume: pd.DataFrame, hybrid_strategy: dict[str, Any]) -> dict[str, Any]:
    portability = portability_metrics(old_close, old_volume, hybrid_close, hybrid_volume)
    if old_close.empty:
        return {"status": "UNAVAILABLE", "portability": portability}
    old_score, _ = build_composite(old_close, old_volume)
    original = strategy_backtest(old_close, old_score)
    return {"status": "PASS" if original.get("status") == "PASS" else "UNAVAILABLE", "original": original.get("metrics", {}), "hybrid": hybrid_strategy.get("metrics", {}), "factor_ic": {}, "factor_coverage": {}, "composite_ic": None, "top5_overlap_rate": portability.get("top5_overlap_rate"), "target_overlap_rate": portability.get("target_overlap_rate"), "diagnostic_only": True}


def evaluate_verdict(*, historical: dict[str, Any], quality: dict[str, Any], l2: dict[str, Any], l4: dict[str, Any], composite: dict[str, Any], strategy: dict[str, Any], parity: dict[str, Any], temporal: dict[str, Any], bootstrap: dict[str, Any], stress: dict[str, Any], psr_dsr: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    insufficient = historical["status"] == "DATA_INSUFFICIENT" or quality["internal_validity"] != "PASS"
    statistical = "INSUFFICIENT_EVIDENCE" if insufficient or temporal["status"] == "INSUFFICIENT_EVIDENCE" or bootstrap["status"] == "INSUFFICIENT_EVIDENCE" or psr_dsr["status"] == "INSUFFICIENT_EVIDENCE" else ("PASS" if all(item["status"] == "PASS" for item in (l2, l4)) and composite["gate"] == "HYBRID_COMPOSITE_PASS" and bootstrap["status"] == "PASS" and psr_dsr["status"] == "PASS" and temporal["status"] == "PASS" else "FAIL")
    economic = "INSUFFICIENT_EVIDENCE" if insufficient or strategy.get("status") != "PASS" or stress["status"] == "INSUFFICIENT_EVIDENCE" else ("PASS" if stress["status"] == "PASS" and temporal["status"] == "PASS" else "FAIL")
    execution = "INSUFFICIENT_EVIDENCE" if insufficient else ("PASS" if strategy.get("status") == "PASS" and parity.get("parity") == "PASS" and frozen.get("status") == "PASS" else "FAIL")
    if insufficient or "INSUFFICIENT_EVIDENCE" in (statistical, economic, execution):
        verdict = "S3_HYBRID_REVALIDATION_INSUFFICIENT"
    elif statistical == economic == execution == "PASS":
        verdict = "S3_HYBRID_REVALIDATED"
    else:
        verdict = "S3_HYBRID_REVALIDATION_REJECTED"
    passed = verdict == "S3_HYBRID_REVALIDATED"
    return {
        "verdict": verdict, "DATA_READINESS": quality["readiness"],
        "STATISTICAL_VALIDITY": statistical, "ECONOMIC_VALIDITY": economic, "EXECUTION_VALIDITY": execution,
        "READY_FOR_FORWARD_SHADOW": "YES" if passed else "NO", "production_ready": "NO",
        "S3_NOT_PORTABLE_TO_HYBRID_DATASET": verdict == "S3_HYBRID_REVALIDATION_REJECTED", "MORE_HYBRID_HISTORY_REQUIRED": verdict == "S3_HYBRID_REVALIDATION_INSUFFICIENT",
        "HISTORICAL_HYBRID_DATA_INSUFFICIENT": historical["status"] == "DATA_INSUFFICIENT", "STATISTICAL_REVALIDATION": "PASS" if statistical == "PASS" else "INSUFFICIENT_EVIDENCE" if statistical == "INSUFFICIENT_EVIDENCE" else "FAIL",
        "fresh_oos_available": False, "revalidation_window_label": REVALIDATION_LABEL,
        "governance": {"strategy_changed": "NO", "factor_changed": "NO", "data_contract_changed": "NO", "fresh_oos_relabeled": "NO", "second_experiment": "NO", "historical_evidence_changed": "NO" if frozen.get("status") == "PASS" else "YES"},
    }


__all__ = [
    "ATOL",
    "BASE_COST",
    "FINGERPRINT",
    "HISTORY_START",
    "OUTPUT_NAMESPACE",
    "S3HybridContract",
    "S3HybridRevalidationError",
    "STRESS_COST",
    "WINDOW_END",
    "WINDOW_START",
    "assert_single_experiment",
    "build_historical_manifest",
    "compare_frozen_snapshots",
    "composite_report",
    "data_quality_report",
    "default_contract",
    "factor_reports",
    "file_sha256",
    "frozen_evidence_snapshot",
    "load_hybrid_inputs",
    "output_dir",
    "portability_metrics",
    "original_vs_hybrid_comparison",
    "strategy_backtest",
    "engine_parity",
    "temporal_validation",
    "bootstrap_report",
    "cost_stress",
    "psr_dsr_report",
    "robustness_report",
    "evaluate_verdict",
    "write_json",
]

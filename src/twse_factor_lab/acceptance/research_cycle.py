"""OOS, statistical acceptance, and freeze primitives for a new cycle.

Legacy RC1 acceptance modules remain read-only and are intentionally not used
for writing these artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance.psr import (
    daily_sharpe,
    deflated_sharpe_ratio,
    moments,
    probabilistic_sharpe_ratio,
)
from twse_factor_lab.analysis.research_robustness import config_fingerprint
from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.governance import (
    ExperimentRecord,
    load_experiment_registry,
    load_research_manifest,
    register_experiment,
    update_experiment_status,
)
from twse_factor_lab.governance.isolation import assert_write_allowed

METHOD_VERSION = "oos-acceptance-freeze-mvp-v1"
STATISTICS_VERSION = "acceptance.psr-dsr-v1"
FREEZE_VERSION = "research-freeze-v1"
ACCEPTANCE_VERSION = "acceptance-matrix-v1"
ACCEPTANCE_SECTIONS = (
    "factor_evidence",
    "strategy_incremental_value",
    "execution",
    "performance",
    "attribution",
    "robustness",
    "oos",
    "statistics",
)


class ResearchCycleError(ValueError):
    """Invalid new-cycle OOS, acceptance, or freeze input."""


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, pd.NA.__class__):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _returns(values: pd.Series, name: str = "returns") -> pd.Series:
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ResearchCycleError(f"{name} index must be a DatetimeIndex")
    if values.index.has_duplicates or not values.index.is_monotonic_increasing:
        raise ResearchCycleError(f"{name} index must be unique and sorted")
    try:
        values = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ResearchCycleError(f"{name} must be numeric") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise ResearchCycleError(f"{name} must contain finite values")
    return values


def _records(series: pd.Series) -> list[dict[str, Any]]:
    return [
        {"date": pd.Timestamp(date).date().isoformat(), "value": _json_value(value)}
        for date, value in series.items()
    ]


def assert_no_lookahead(
    observations: pd.DataFrame,
    *,
    signal_date_column: str = "signal_date",
    available_date_column: str = "available_date",
) -> None:
    """Require every signal's declared inputs to be available by signal date."""
    missing = {signal_date_column, available_date_column} - set(observations.columns)
    if missing:
        raise ResearchCycleError(f"missing availability columns: {sorted(missing)}")
    signal = pd.to_datetime(observations[signal_date_column], errors="coerce")
    available = pd.to_datetime(observations[available_date_column], errors="coerce")
    if signal.isna().any() or available.isna().any() or (available > signal).any():
        raise ResearchCycleError("future data is not allowed in an earlier signal")


def evaluate_oos(
    returns: pd.Series,
    *,
    oos_start: str,
    oos_end: str,
    initial_capital: float = 1_000_000.0,
    warmup_range: tuple[str, str] | None = None,
    strategy_id: str | None = None,
    research_id: str | None = None,
    candidate_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate only OOS returns with fresh capital and no inherited state."""
    values = _returns(returns)
    start, end = pd.Timestamp(oos_start), pd.Timestamp(oos_end)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ResearchCycleError("invalid OOS boundary")
    try:
        initial_capital = float(initial_capital)
    except (TypeError, ValueError) as exc:
        raise ResearchCycleError("initial_capital must be positive and finite") from exc
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ResearchCycleError("initial_capital must be positive and finite")
    oos = values.loc[(values.index >= start) & (values.index <= end)]
    if oos.empty:
        raise ResearchCycleError("OOS period has no return observations")
    if warmup_range is None:
        warmup = values.loc[values.index < start]
        warmup_range = (
            warmup.index.min().date().isoformat() if not warmup.empty else None,
            warmup.index.max().date().isoformat() if not warmup.empty else None,
        )
    nav = initial_capital * (1.0 + oos).cumprod()
    metrics = compute_metrics(oos)
    metrics.update(
        {
            "observation_count": int(len(oos)),
            "start_date": oos.index.min().date().isoformat(),
            "end_date": oos.index.max().date().isoformat(),
        }
    )
    candidate = None if candidate_config is None else dict(candidate_config)
    return {
        "strategy_id": strategy_id,
        "research_id": research_id,
        "candidate_config": candidate,
        "candidate_fingerprint": (
            None if candidate is None else config_fingerprint(candidate)
        ),
        "is_period": None,
        "oos_period": {
            "start": oos.index.min().date().isoformat(),
            "end": oos.index.max().date().isoformat(),
        },
        "initial_capital": initial_capital,
        "initial_positions": {},
        "inherited_is_state": False,
        "warmup_range": {"start": warmup_range[0], "end": warmup_range[1]},
        "warmup_pnl": 0.0,
        "oos_returns": _records(oos),
        "oos_nav": _records(nav),
        "oos_metrics": _json_value(metrics),
        "lookahead": {
            "status": "PASS",
            "rule": "only data available by T; signal T executes T+1",
        },
        "method": METHOD_VERSION,
    }


def run_oos_evaluation(
    *,
    root: str | Path,
    research_id: str,
    strategy_id: str,
    experiment_id: str,
    returns: pd.Series,
    candidate_config: Mapping[str, Any],
    initial_capital: float = 1_000_000.0,
    warmup_range: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Run fixed-candidate OOS as one diagnostic-only experiment."""
    root = Path(root)
    research = load_research_manifest(root, research_id)
    register_experiment(
        ExperimentRecord(
            experiment_id=experiment_id,
            research_id=research_id,
            config={
                "experiment_type": "oos_evaluation",
                "strategy_id": strategy_id,
                "candidate_config": dict(candidate_config),
                "candidate_fingerprint": config_fingerprint(candidate_config),
                "selection_relevant": False,
            },
            dataset_version=research.dataset_version,
            experiment_type="diagnostic",
            status="running",
            selection_relevant=False,
        ),
        root,
    )
    try:
        result = evaluate_oos(
            returns,
            oos_start=research.oos_start,
            oos_end=research.oos_end,
            initial_capital=initial_capital,
            warmup_range=warmup_range,
            strategy_id=strategy_id,
            research_id=research_id,
            candidate_config=candidate_config,
        )
        directory = root / "data" / "research" / research_id / "oos" / strategy_id
        path = _write_json(root, directory / "oos_evaluation.json", result)
        result["artifact_path"] = path
        update_experiment_status(root, research_id, experiment_id, "completed", result)
        return result
    except Exception as exc:
        update_experiment_status(
            root,
            research_id,
            experiment_id,
            "failed",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def build_research_trial_inventory(
    root: str | Path, research_id: str
) -> tuple[pd.DataFrame, int]:
    """Build the complete inventory from the append-only Experiment Registry."""
    records = load_experiment_registry(root, research_id)
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.experiment_id):
        result = record.result if isinstance(record.result, Mapping) else {}
        config = dict(record.config)
        rows.append(
            {
                "experiment_id": record.experiment_id,
                "research_id": record.research_id,
                "experiment_type": record.experiment_type,
                "status": record.status,
                "selection_relevant": record.selection_relevant,
                "dataset_version": record.dataset_version,
                "config_fingerprint": config_fingerprint(config),
                "strategy_id": config.get("strategy_id", result.get("strategy_id")),
                "factor_id": config.get("factor_id"),
                "config": config,
                "result": dict(result),
            }
        )
    columns = [
        "experiment_id",
        "research_id",
        "experiment_type",
        "status",
        "selection_relevant",
        "dataset_version",
        "config_fingerprint",
        "strategy_id",
        "factor_id",
        "config",
        "result",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    effective = int(
        frame.loc[frame["selection_relevant"], "config_fingerprint"].nunique()
        if not frame.empty
        else 0
    )
    return frame, effective


def _trial_sharpes(
    inventory: pd.DataFrame, observed_daily: float, effective: int
) -> list[float]:
    if effective <= 0:
        return []
    candidates = inventory[inventory["selection_relevant"]].copy()
    candidates = candidates.drop_duplicates("config_fingerprint")
    values: list[float] = []
    for result in candidates["result"]:
        metric = result.get("sharpe") if isinstance(result, Mapping) else None
        if metric is None and isinstance(result, Mapping):
            metric = result.get("oos_metrics", {}).get("sharpe")
        values.append(
            float(metric) / math.sqrt(252) if metric is not None else observed_daily
        )
    return values + [observed_daily] * max(0, effective - len(values))


def compute_statistical_acceptance(
    returns: pd.Series,
    inventory: pd.DataFrame,
    *,
    benchmark_sharpe: float = 0.0,
    acceptance_threshold: float = 0.95,
) -> dict[str, Any]:
    """Reuse canonical PSR/DSR functions with registry-derived trial count."""
    values = _returns(returns)
    if len(values) <= 1:
        raise ResearchCycleError(
            "statistical acceptance needs at least two observations"
        )
    effective = int(
        inventory.loc[
            inventory["selection_relevant"], "config_fingerprint"
        ].nunique()
        if not inventory.empty
        else 0
    )
    observed = daily_sharpe(values)
    skewness, kurtosis = moments(values)
    psr = probabilistic_sharpe_ratio(
        observed,
        len(values),
        benchmark_sharpe=benchmark_sharpe,
        skewness=skewness,
        kurtosis_excess=kurtosis - 3.0,
    )
    trial_sharpes = _trial_sharpes(inventory, observed, effective)
    dsr, sr0 = (
        deflated_sharpe_ratio(observed, len(values), trial_sharpes)
        if trial_sharpes
        else (None, None)
    )
    status = (
        "PASS"
        if effective and dsr is not None and dsr >= acceptance_threshold
        else "INSUFFICIENT"
        if not effective
        else "REJECT"
    )
    return {
        "status": status,
        "psr": float(psr),
        "dsr": None if dsr is None else float(dsr),
        "sr0_daily": None if sr0 is None else float(sr0),
        "effective_trials": effective,
        "observed_sharpe_daily": float(observed),
        "benchmark_sharpe": float(benchmark_sharpe),
        "sample_size": int(len(values)),
        "skew": float(skewness),
        "kurtosis": float(kurtosis),
        "annualization_sessions": 252,
        "acceptance_threshold": float(acceptance_threshold),
        "method": STATISTICS_VERSION,
        "effective_trial_count_source": (
            "Experiment Registry unique config_fingerprint where "
            "selection_relevant=true"
        ),
        "missing_trial_result_assumption": (
            "missing terminal result uses observed daily Sharpe; count is retained"
        ),
        "risk_free_rate": 0.0,
    }


def _section_status(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("status", value.get("verdict", value.get("acceptance")))
    if value is None:
        return "UNAVAILABLE"
    return str(value).upper()


def build_acceptance_matrix(
    evidence: Mapping[str, Any],
    *,
    real_candidate: bool = False,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine all gates while keeping platform and strategy verdicts separate."""
    sections = {
        name: {
            "status": _section_status(evidence.get(name)),
            "evidence": evidence.get(name),
        }
        for name in ACCEPTANCE_SECTIONS
    }
    platform_failures = [
        name
        for name in ("execution", "performance")
        if sections[name]["status"] in {"FAIL", "REJECT"}
    ]
    platform = "FAIL" if platform_failures else "PASS"
    reasons: list[str] = []
    if platform_failures:
        reasons.append("platform evidence failed: " + ", ".join(platform_failures))
    if not real_candidate:
        strategy = "NOT_EVALUATED"
        evaluation_status = "INFRASTRUCTURE_VALIDATION_ONLY"
        reasons.append(
            "INFRASTRUCTURE_VALIDATION_ONLY: no real Day 3 admission / "
            "Day 4 selected strategy evidence"
        )
    else:
        evaluation_status = "REAL_RESEARCH"
        statuses = [item["status"] for item in sections.values()]
        if any(status in {"FAIL", "REJECT", "FRAGILE"} for status in statuses):
            strategy = "REJECT"
            reasons.append("one or more pre-declared acceptance gates failed")
        elif all(
            status in {"PASS", "ACCEPT", "ACCEPTABLE", "ROBUST", "SUFFICIENT"}
            for status in statuses
        ):
            strategy = "ACCEPT"
            reasons.append("all pre-declared acceptance gates passed")
        else:
            strategy = "CANDIDATE"
            reasons.append("evidence is mixed or incomplete")
    return {
        "version": ACCEPTANCE_VERSION,
        "sections": sections,
        "thresholds": dict(thresholds or {}),
        "research_platform_verdict": platform,
        "strategy_verdict": strategy,
        "strategy_verdict_available": real_candidate,
        "evaluation_status": evaluation_status,
        "reasons": reasons,
    }


def _write_json(root: Path, path: Path, payload: Any) -> str:
    root = root.resolve()
    target = assert_write_allowed(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return target.relative_to(root).as_posix()


def _hash_files(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths, key=lambda item: item.as_posix())
    }


def freeze_research_cycle(
    *,
    root: str | Path,
    research_id: str,
    candidate_config: Mapping[str, Any] | None,
    acceptance: Mapping[str, Any],
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Write a self-contained, hash-verified freeze below the research namespace."""
    root = Path(root).resolve()
    research = load_research_manifest(root, research_id)
    research_root = root / "data" / "research" / research_id
    freeze_root = research_root / "freeze"
    freeze_manifest_path = freeze_root / "research_freeze_manifest.json"
    if freeze_manifest_path.exists():
        raise ResearchCycleError("research cycle is already frozen")
    records = load_experiment_registry(root, research_id)
    running = [
        record.experiment_id
        for record in records
        if record.status in {"planned", "running"}
    ]
    if running:
        raise ResearchCycleError(
            "cannot freeze with non-terminal experiments: " + ", ".join(running)
        )
    inventory, effective = build_research_trial_inventory(root, research_id)
    inventory_payload = {
        "research_id": research_id,
        "effective_trials": effective,
        "records": inventory.to_dict("records"),
        "method": "Experiment Registry canonical inventory v1",
    }
    inventory_path = freeze_root / "trial_inventory.json"
    _write_json(root, inventory_path, inventory_payload)
    handoff = {
        "research_id": research_id,
        "dataset_version": research.dataset_version,
        "candidate_config": None
        if candidate_config is None
        else dict(candidate_config),
        "candidate_fingerprint": (
            None if candidate_config is None else config_fingerprint(candidate_config)
        ),
        "effective_trials": effective,
        "research_platform_verdict": acceptance.get("research_platform_verdict"),
        "strategy_verdict": acceptance.get("strategy_verdict"),
        "evaluation_status": acceptance.get("evaluation_status"),
        "acceptance": dict(acceptance),
        "method": FREEZE_VERSION,
    }
    handoff_path = freeze_root / "acceptance_handoff.json"
    _write_json(root, handoff_path, handoff)
    source_files = [
        path
        for path in research_root.rglob("*")
        if path.is_file() and freeze_root not in path.parents
    ]
    source_files.extend([inventory_path, handoff_path])
    hashes = _hash_files(root, source_files)
    hashes_path = freeze_root / "artifact_hashes.json"
    _write_json(
        root,
        hashes_path,
        {"method": FREEZE_VERSION, "artifacts": hashes},
    )
    freeze_payload = {
        "freeze_version": FREEZE_VERSION,
        "research_id": research.research_id,
        "dataset_version": research.dataset_version,
        "candidate_config": None
        if candidate_config is None
        else dict(candidate_config),
        "candidate_fingerprint": (
            None if candidate_config is None else config_fingerprint(candidate_config)
        ),
        "research_platform_verdict": acceptance.get("research_platform_verdict"),
        "strategy_verdict": acceptance.get("strategy_verdict"),
        "evaluation_status": acceptance.get("evaluation_status"),
        "effective_trials": effective,
        "artifact_hashes": hashes,
        "code_revision": code_revision,
        "artifact_hashes_path": hashes_path.relative_to(root).as_posix(),
    }
    _write_json(root, freeze_manifest_path, freeze_payload)
    reproducibility = {
        "manifest_version": FREEZE_VERSION,
        "research_id": research_id,
        "research_manifest": "data/research/" + research_id + "/research_manifest.json",
        "dataset_version": research.dataset_version,
        "experiment_registry": "data/research/"
        + research_id
        + "/experiment_registry.json",
        "candidate_config": freeze_payload["candidate_config"],
        "effective_trials": effective,
        "artifact_hashes": hashes,
        "freeze_manifest": freeze_manifest_path.relative_to(root).as_posix(),
        "code_revision": code_revision,
    }
    _write_json(root, freeze_root / "reproducibility_manifest.json", reproducibility)
    return verify_research_freeze(root, research_id)


def verify_research_freeze(root: str | Path, research_id: str) -> dict[str, Any]:
    """Verify every frozen new-cycle artifact and reject tampering."""
    root = Path(root).resolve()
    freeze_root = root / "data" / "research" / research_id / "freeze"
    manifest_path = freeze_root / "research_freeze_manifest.json"
    hashes_path = freeze_root / "artifact_hashes.json"
    if not manifest_path.exists() or not hashes_path.exists():
        raise ResearchCycleError("research freeze manifest is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hash_payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes = hash_payload.get("artifacts", {})
    if manifest.get("artifact_hashes") != hashes:
        raise ResearchCycleError("freeze hash manifest is inconsistent")
    checked: dict[str, str] = {}
    for relative, expected in hashes.items():
        path = root / relative
        if not path.exists():
            raise ResearchCycleError(f"frozen artifact missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ResearchCycleError(f"frozen artifact hash mismatch: {relative}")
        checked[relative] = actual
    inventory_path = freeze_root / "trial_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if int(inventory["effective_trials"]) != int(manifest["effective_trials"]):
        raise ResearchCycleError("frozen effective trial count mismatch")
    return {
        "status": "PASS",
        "research_id": research_id,
        "effective_trials": int(manifest["effective_trials"]),
        "artifact_count": len(checked),
        "artifact_hashes": checked,
        "freeze_path": manifest_path.relative_to(root).as_posix(),
    }


__all__ = [
    "ACCEPTANCE_SECTIONS",
    "ACCEPTANCE_VERSION",
    "FREEZE_VERSION",
    "METHOD_VERSION",
    "ResearchCycleError",
    "assert_no_lookahead",
    "build_acceptance_matrix",
    "build_research_trial_inventory",
    "compute_statistical_acceptance",
    "evaluate_oos",
    "freeze_research_cycle",
    "run_oos_evaluation",
    "verify_research_freeze",
]

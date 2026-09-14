"""Load canonical research artifacts into the normalized report model."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import ReportSection, ResearchReportModel
from .provenance import (
    REPORT_SCHEMA_VERSION,
    freeze_manifest_sha256,
    resolve_dependency_version,
)


class ReportLoaderError(RuntimeError):
    """Raised when a mandatory research artifact cannot be loaded."""


_FACTOR_FAMILIES = {
    "pe": "VALUE",
    "pb": "VALUE",
    "dividend_yield": "VALUE",
    "eps": "QUALITY",
    "roe": "QUALITY",
    "revenue_yoy": "GROWTH",
    "momentum_40d": "MOMENTUM",
    "momentum_60d": "MOMENTUM",
    "risk_adjusted_momentum": "MOMENTUM",
}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportLoaderError(f"cannot read artifact: {path}") from exc


def _research_root(root: str | Path, research_id: str) -> tuple[Path, Path]:
    base = Path(root).resolve()
    candidates = (
        base / "data" / "research" / research_id,
        base / "research" / research_id,
        base / research_id,
    )
    for candidate in candidates:
        if (candidate / "research_manifest.json").exists():
            return base, candidate
    return base, candidates[0]


def _records(registry: Any) -> list[dict[str, Any]]:
    if isinstance(registry, list):
        return [item for item in registry if isinstance(item, dict)]
    if isinstance(registry, dict):
        value = registry.get("records", registry.get("experiments", []))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _section_evidence(matrix: dict[str, Any], name: str) -> dict[str, Any]:
    section = matrix.get("sections", {}).get(name, {})
    evidence = section.get("evidence", {}) if isinstance(section, dict) else {}
    if not isinstance(evidence, dict):
        return {}
    report = evidence.get("report")
    return report if isinstance(report, dict) else evidence


def _source_path(root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _status(source: Any, default: str = "AVAILABLE") -> str:
    if isinstance(source, dict) and source.get("status") is not None:
        return str(source["status"])
    return default


def _normalize_trial_counts(evidence: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(evidence)
    normalized["total_selection_relevant_trials"] = evidence.get(
        "total_selection_relevant_experiments",
        evidence.get("total_selection_relevant_trials"),
    )
    normalized["dsr_effective_strategy_trials"] = evidence.get(
        "effective_trials", evidence.get("dsr_effective_strategy_trials")
    )
    normalized["diagnostic_trial_count"] = evidence.get(
        "diagnostic_count", evidence.get("diagnostic_trial_count")
    )
    total = normalized["total_selection_relevant_trials"]
    factor = normalized.get("factor_selection_trial_count")
    strategy = normalized.get("strategy_selection_trial_count")
    normalized["trial_count_consistency"] = (
        "PASS"
        if isinstance(factor, (int, float))
        and isinstance(strategy, (int, float))
        and total == factor + strategy
        else "MISMATCH"
    )
    return normalized


def _attribution_evidence_status(attribution: dict[str, Any]) -> str:
    if attribution.get("status") == "INSUFFICIENT_DATA":
        return "INSUFFICIENT_DATA"
    available = attribution.get("available_descriptors")
    if not available:
        return "UNAVAILABLE"
    return "PARTIAL" if attribution.get("unavailable_descriptors") else "AVAILABLE"


def _primary_config(
    research_root: Path,
    matrix: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    freeze = research_root / "freeze" / "research_freeze_manifest.json"
    if freeze.exists():
        candidate = _json(freeze).get("candidate_config")
        if isinstance(candidate, dict):
            return candidate
    candidate = matrix.get("candidate_config")
    if isinstance(candidate, dict):
        return candidate
    trials = [
        record
        for record in records
        if record.get("experiment_type") == "strategy_backtest"
        and record.get("selection_relevant") is True
    ]
    for record in trials:
        result = record.get("result", {})
        if result.get("status") in {"selected", "locked", "winner"}:
            return dict(record.get("config", {}))
    return dict(trials[0].get("config", {})) if trials else {}


def _factor_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("experiment_type") != "factor_test":
            continue
        config = record.get("config", {})
        result = record.get("result", {})
        factor_id = result.get("factor_id", config.get("factor_id"))
        horizons = result.get("horizon_metrics", [])
        primary = result.get("primary_horizon") or result.get("selected_horizon")
        if primary is None:
            primary = config.get("thresholds", {}).get("primary_horizon")
        metric = next((item for item in horizons if item.get("horizon") == primary), {})
        family = (
            result.get("family")
            or config.get("family")
            or config.get("factor_family")
            or _FACTOR_FAMILIES.get(str(factor_id), "UNAVAILABLE")
        )
        rows.append(
            {
                "factor": factor_id,
                "family": family,
                "direction": result.get("direction", "UNAVAILABLE"),
                "primary_horizon": primary,
                "mean_ic": metric.get("mean_ic", result.get("mean_ic")),
                "icir": metric.get("icir", result.get("icir")),
                "positive_ic_ratio": metric.get(
                    "positive_ic_ratio", result.get("positive_ic_ratio")
                ),
                "quantile_spread": metric.get(
                    "top_bottom_spread", result.get("top_bottom_spread")
                ),
                "coverage": metric.get("coverage", result.get("coverage")),
                "turnover": metric.get("turnover", result.get("turnover")),
                "rank_autocorrelation": metric.get(
                    "rank_autocorrelation", result.get("rank_autocorrelation")
                ),
                "verdict": result.get("verdict", "UNAVAILABLE"),
                "reasons": result.get("reasons", metric.get("reasons", [])),
                "experiment_id": record.get("experiment_id"),
            }
        )
    return sorted(rows, key=lambda row: str(row.get("factor")))


def _strategy_rows(
    records: list[dict[str, Any]], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    locked_id = candidate.get("strategy_id")
    rows: list[dict[str, Any]] = []
    for record in records:
        if (
            record.get("experiment_type") != "strategy_backtest"
            or record.get("selection_relevant") is not True
        ):
            continue
        config = record.get("config", {})
        result = record.get("result", {})
        strategy_id = result.get("strategy_id", config.get("strategy_id"))
        rows.append(
            {
                "trial_id": strategy_id or record.get("experiment_id"),
                "factors": result.get("factor_ids", config.get("factor_ids", [])),
                "weights": result.get(
                    "factor_weights", config.get("factor_weights", {})
                ),
                "top_n": result.get("top_n", config.get("top_n")),
                "rebalance": result.get(
                    "rebalance_frequency", config.get("rebalance_frequency")
                ),
                "buffer": result.get(
                    "buffer",
                    result.get(
                        "drop_rank_buffer",
                        config.get("drop_rank_buffer", config.get("buffer")),
                    ),
                ),
                "cost": result.get("cost_scenario", config.get("cost_scenario")),
                "is_return": result.get("total_return"),
                "is_sharpe": result.get("sharpe"),
                "mdd": result.get("max_drawdown"),
                "selection_status": (
                    "LOCKED CANDIDATE"
                    if strategy_id == locked_id
                    else result.get("selection_status", "SELECTION-RELEVANT")
                ),
                "locked": strategy_id == locked_id,
                "experiment_id": record.get("experiment_id"),
                "actual_engine": result.get(
                    "actual_engine", config.get("execution_engine")
                ),
                "requested_engine": result.get(
                    "requested_engine", config.get("execution_engine")
                ),
                "exposure": result.get("exposure"),
                "turnover": result.get("turnover"),
            }
        )
    return sorted(rows, key=lambda row: str(row.get("trial_id")))


def _find_handoff(research_root: Path, strategy_id: str | None) -> Path | None:
    if not strategy_id:
        return None
    exact = research_root / "strategy_handoffs" / strategy_id
    if exact.is_dir():
        return exact
    matches = sorted(
        path
        for path in (research_root / "strategy_handoffs").glob(f"*{strategy_id}*")
        if path.is_dir()
    )
    return matches[0] if matches else None


def _find_primary_performance(
    research_root: Path, strategy_id: str | None
) -> Path | None:
    if not strategy_id:
        return None
    path = research_root / "performance" / strategy_id
    return path if path.is_dir() else None


def _load_optional(path: Path | None) -> Any:
    return _json(path) if path is not None and path.exists() else None


def _limitations(
    performance: dict[str, Any],
    pyfolio: dict[str, Any],
    attribution: dict[str, Any],
    research: dict[str, Any],
    dataset: dict[str, Any],
) -> list[str]:
    limitations: list[str] = []
    transactions = pyfolio.get("transactions", performance.get("transactions"))
    if isinstance(transactions, dict) and (
        transactions.get("status") == "UNAVAILABLE"
        or transactions.get("available") is False
    ):
        limitations.append("transactions unavailable")
    if performance.get("benchmark") is None and pyfolio.get("benchmark") is None:
        limitations.append("benchmark unavailable")
    if attribution.get("pit_industry_status") == "PIT_INDUSTRY_UNAVAILABLE":
        limitations.append("PIT industry unavailable")
    text = json.dumps(
        {"research": research, "dataset": dataset}, ensure_ascii=False
    ).lower()
    if "survivorship" in text or "current-listed" in text:
        limitations.append("current-listed survivorship")
    if "partial" in text or "13-ticker" in text or "partial_universe" in text:
        limitations.append("partial universe coverage")
    return limitations


def load_research_report_model(
    root: str | Path, research_id: str
) -> ResearchReportModel:
    """Load a report model without mutating or recomputing research evidence."""

    repo_root, research_root = _research_root(root, research_id)
    mandatory = {
        name: research_root / name
        for name in (
            "research_manifest.json",
            "experiment_registry.json",
            "acceptance_matrix.json",
        )
    }
    missing = [name for name, path in mandatory.items() if not path.exists()]
    if missing:
        raise ReportLoaderError(
            "missing mandatory artifact(s): " + ", ".join(sorted(missing))
        )

    manifest = _json(mandatory["research_manifest.json"])
    registry = _json(mandatory["experiment_registry.json"])
    matrix = _json(mandatory["acceptance_matrix.json"])
    records = _records(registry)
    candidate = _primary_config(research_root, matrix, records)
    candidate_id = candidate.get("strategy_id")

    dataset_manifest = _load_optional(research_root / "dataset_manifests.json")
    if isinstance(dataset_manifest, list):
        dataset_records = dataset_manifest
    elif isinstance(dataset_manifest, dict):
        dataset_records = dataset_manifest.get("records", [dataset_manifest])
    else:
        dataset_records = []

    factor_rows = _factor_rows(records)
    strategy_rows = _strategy_rows(records, candidate)
    factor_status = _status(matrix.get("sections", {}).get("factor_evidence"))

    handoff = _find_handoff(research_root, candidate_id)
    performance_dir = _find_primary_performance(research_root, candidate_id)
    canonical = _load_optional(
        performance_dir / "canonical_metrics.json" if performance_dir else None
    )
    pyfolio_metadata = _load_optional(
        performance_dir / "pyfolio_metadata.json" if performance_dir else None
    )
    pyfolio_metrics = _load_optional(
        performance_dir / "pyfolio_metrics.json" if performance_dir else None
    )
    performance_report = {
        "strategy_id": candidate_id,
        "canonical_metrics": canonical or {},
        "cross_check": (pyfolio_metadata or {}).get("cross_check", {}),
        "transactions": (pyfolio_metadata or {}).get("transactions", {}),
        "transaction_dependent_diagnostics": (pyfolio_metadata or {}).get(
            "transaction_dependent_diagnostics", {}
        ),
        "benchmark": (pyfolio_metadata or {}).get("benchmark")
        if isinstance(pyfolio_metadata, dict)
        else None,
        "portfolio_exposure": {
            "status": "AVAILABLE"
            if handoff is not None and (handoff / "positions.csv").exists()
            else "UNAVAILABLE"
        },
        "source_files": {
            "repository_root": str(repo_root),
            "handoff_dir": _relative(repo_root, handoff),
            "returns": _relative(
                repo_root, handoff / "returns.csv" if handoff else None
            ),
            "nav": _relative(repo_root, handoff / "nav.csv" if handoff else None),
            "positions": _relative(
                repo_root, handoff / "positions.csv" if handoff else None
            ),
            "canonical_metrics": _relative(
                repo_root,
                performance_dir / "canonical_metrics.json" if performance_dir else None,
            ),
            "pyfolio_metadata": _relative(
                repo_root,
                performance_dir / "pyfolio_metadata.json" if performance_dir else None,
            ),
        },
    }
    if isinstance(canonical, dict):
        performance_report["canonical_metrics"] = dict(canonical)

    parity_path = (
        research_root
        / "execution_validation"
        / str(candidate_id)
        / f"tri-engine-{candidate_id}"
        / "parity.json"
        if candidate_id
        else None
    )
    parity = _load_optional(parity_path)
    if parity is None:
        candidates = sorted(
            (research_root / "execution_validation").glob("**/parity.json")
        )
        parity = _load_optional(candidates[0]) if candidates else None

    attribution_path = (
        research_root / "attribution" / str(candidate_id) / "attribution.json"
        if candidate_id
        else None
    )
    attribution = _load_optional(attribution_path) or {}
    robustness_path = (
        research_root
        / "robustness"
        / str(candidate_id)
        / f"robustness-{candidate_id}.json"
        if candidate_id
        else None
    )
    robustness = _load_optional(robustness_path) or {}
    oos_path = (
        research_root / "oos" / str(candidate_id) / "fresh_state_oos.json"
        if candidate_id
        else None
    )
    oos = _load_optional(oos_path)
    if (
        not isinstance(oos, dict)
        or oos.get("evidence_type") != "fresh_state_reexecution"
    ):
        oos = None

    freeze_path = research_root / "freeze" / "research_freeze_manifest.json"
    if freeze_path.parent.exists() and not freeze_path.exists():
        raise ReportLoaderError(
            "frozen research cycle is missing source freeze manifest: "
            f"{freeze_path}"
        )
    freeze = _load_optional(freeze_path)
    reproducibility = {}
    if isinstance(freeze, dict):
        reproducibility = {
            "source_freeze_manifest": _relative(repo_root, freeze_path),
            "source_freeze_manifest_sha256": freeze_manifest_sha256(freeze_path),
            "source_code_revision": freeze.get("code_revision"),
            "source_git_dirty": freeze.get("git_dirty"),
            "candidate_fingerprint": freeze.get("candidate_fingerprint"),
            "freeze_version": freeze.get("freeze_version"),
            "total_selection_relevant_trials": freeze.get("effective_trials"),
            "dependency_snapshot": freeze.get("dependency_snapshot", {}),
            "source_environment": {
                "dependency_snapshot_recorded_in_freeze": freeze.get(
                    "dependency_snapshot", {}
                )
            },
            "report_generator_environment": {
                "pyfolio": resolve_dependency_version(
                    "pyfolio", ("pyfolio-reloaded", "pyfolio")
                )
            },
            "report_schema_version": REPORT_SCHEMA_VERSION,
        }
    else:
        reproducibility = {
            "source_environment": {
                "dependency_snapshot_recorded_in_freeze": None
            },
            "report_generator_environment": {
                "pyfolio": resolve_dependency_version(
                    "pyfolio", ("pyfolio-reloaded", "pyfolio")
                )
            },
            "report_schema_version": REPORT_SCHEMA_VERSION,
        }

    acceptance = {
        "research_platform_verdict": matrix.get("research_platform_verdict"),
        "strategy_verdict": matrix.get("strategy_verdict"),
        "strategy_verdict_available": matrix.get("strategy_verdict_available"),
        "evaluation_status": matrix.get("evaluation_status"),
        "reasons": matrix.get("reasons", []),
        "section_statuses": {
            name: section.get("status")
            for name, section in matrix.get("sections", {}).items()
            if isinstance(section, dict)
        },
    }
    execution = parity or {}
    if execution:
        execution = dict(execution)
        engines = execution.get("engines", {})
        fallback = any("fallback" in str(value).lower() for value in engines.values())
        execution["parity_status"] = "FALLBACK" if fallback else execution.get("status")
    oos_data = {}
    if oos is not None:
        oos_data = {
            key: oos.get(key)
            for key in (
                "research_id",
                "strategy_id",
                "warmup_start",
                "warmup_end",
                "oos_start",
                "oos_end",
                "initial_cash",
                "initial_positions",
                "evidence_type",
                "fresh_state",
                "inherited_is_state",
                "oos_metrics",
                "lookahead",
            )
            if key in oos
        }
        oos_data["source_file"] = _relative(repo_root, oos_path)

    pyfolio = {
        "metadata": pyfolio_metadata or {},
        "metrics": pyfolio_metrics or {},
        "cross_check": performance_report["cross_check"],
    }
    statistics = _normalize_trial_counts(_section_evidence(matrix, "statistics"))
    execution_status = _status(attribution, "UNAVAILABLE")
    evidence_status = _attribution_evidence_status(attribution)
    attribution_data = {
        key: attribution.get(key)
        for key in (
            "status",
            "method",
            "available_descriptors",
            "unavailable_descriptors",
            "pit_industry_status",
            "neutralization",
            "exposure",
            "reconciliation_error",
        )
        if key in attribution
    }
    attribution_data["execution_status"] = execution_status
    attribution_data["evidence_status"] = evidence_status
    robustness_data = robustness
    robustness_status = _status(
        robustness,
        _status(matrix.get("sections", {}).get("robustness"), "UNAVAILABLE"),
    )
    limitations = _limitations(
        performance_report,
        pyfolio_metadata or {},
        attribution,
        manifest,
        dataset_manifest or {},
    )

    def section(status: str, data: dict[str, Any] | None = None) -> ReportSection:
        return ReportSection(status=status, data=data or {})

    return ResearchReportModel(
        research=section("AVAILABLE", dict(manifest)),
        dataset=section(
            "AVAILABLE"
            if dataset_records or manifest.get("dataset_version")
            else "UNAVAILABLE",
            {
                "manifests": dataset_records,
                "dataset_version": manifest.get("dataset_version"),
            },
        ),
        factor_evidence=section(factor_status, {"rows": factor_rows}),
        strategy_trials=section(
            "AVAILABLE" if strategy_rows else "UNAVAILABLE", {"rows": strategy_rows}
        ),
        locked_candidate=section(
            "AVAILABLE" if candidate else "UNAVAILABLE", {"config": candidate}
        ),
        execution_validation=section(_status(parity, "UNAVAILABLE"), execution),
        performance=section(
            "AVAILABLE" if canonical is not None else "UNAVAILABLE", performance_report
        ),
        pyfolio=section(
            "AVAILABLE"
            if pyfolio_metadata is not None or pyfolio_metrics is not None
            else "UNAVAILABLE",
            pyfolio,
        ),
        attribution=section(evidence_status, attribution_data),
        robustness=section(robustness_status, robustness_data),
        oos=section("AVAILABLE" if oos is not None else "UNAVAILABLE", oos_data),
        statistics=section(
            _status(matrix.get("sections", {}).get("statistics"), "UNAVAILABLE"),
            statistics,
        ),
        acceptance=section("AVAILABLE", acceptance),
        reproducibility=section(
            "AVAILABLE" if freeze is not None else "UNAVAILABLE", reproducibility
        ),
        limitations=section("AVAILABLE", {"items": limitations}),
    )


__all__ = ["ReportLoaderError", "load_research_report_model"]

"""Build provenance for a derived research report."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from .model import ResearchReportModel

REPORT_SCHEMA_VERSION = "research-report-v1.1"


def resolve_dependency_version(
    import_name: str, distribution_candidates: tuple[str, ...]
) -> dict[str, str | None]:
    """Resolve an import name through its installed distribution names."""

    for distribution in distribution_candidates:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
        return {
            "import_name": import_name,
            "distribution": distribution,
            "version": version,
        }
    return {"import_name": import_name, "distribution": None, "version": None}


def freeze_manifest_sha256(manifest_path: str | Path) -> str:
    """Hash the exact bytes of a freeze manifest."""

    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def _git_revision(root: Path) -> str | None:
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


def build_provenance(
    model: ResearchReportModel,
    root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(root).resolve()
    reproducibility = model.reproducibility.data
    freeze_rel = reproducibility.get("source_freeze_manifest")
    freeze_path = root / freeze_rel if freeze_rel else None
    if freeze_path is not None and not freeze_path.exists():
        raise FileNotFoundError(f"source freeze manifest not found: {freeze_path}")
    frozen = freeze_path is not None
    freeze_sha = freeze_manifest_sha256(freeze_path) if frozen else None
    acceptance = model.acceptance.data
    statistics = model.statistics.data
    source_environment = reproducibility.get("source_environment")
    if not isinstance(source_environment, dict):
        source_environment = {
            "dependency_snapshot_recorded_in_freeze": reproducibility.get(
                "dependency_snapshot", {}
            )
        }
    report_generator_environment = reproducibility.get("report_generator_environment")
    if not isinstance(report_generator_environment, dict):
        report_generator_environment = {
            "pyfolio": resolve_dependency_version(
                "pyfolio", ("pyfolio-reloaded", "pyfolio")
            )
        }
    return {
        "research_id": model.research.data.get("research_id"),
        "source_freeze_manifest": freeze_rel if frozen else None,
        "source_freeze_manifest_sha256": freeze_sha,
        "source_code_revision": reproducibility.get("source_code_revision")
        if frozen
        else None,
        "source_git_dirty": reproducibility.get("source_git_dirty") if frozen else None,
        "freeze_version": reproducibility.get("freeze_version") if frozen else None,
        "candidate_fingerprint": reproducibility.get("candidate_fingerprint")
        if frozen
        else None,
        "report_generator_code_revision": _git_revision(root),
        "report_generated_at": datetime.now(UTC).isoformat(),
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "pyfolio_version": report_generator_environment.get("pyfolio", {}).get(
            "version"
        ),
        "source_environment": source_environment,
        "report_generator_environment": report_generator_environment,
        "factor_selection_trial_count": statistics.get("factor_selection_trial_count"),
        "strategy_selection_trial_count": statistics.get("strategy_selection_trial_count"),
        "total_selection_relevant_trials": statistics.get(
            "total_selection_relevant_trials"
        ),
        "dsr_effective_strategy_trials": statistics.get(
            "dsr_effective_strategy_trials"
        ),
        "diagnostic_trial_count": statistics.get("diagnostic_trial_count"),
        "trial_count_consistency": statistics.get("trial_count_consistency"),
        "platform_verdict": acceptance.get("research_platform_verdict"),
        "strategy_verdict": acceptance.get("strategy_verdict"),
        "pre_freeze": not frozen,
    }


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_provenance",
    "freeze_manifest_sha256",
    "resolve_dependency_version",
]

"""Deterministic repository-boundary and cleanup-inventory checks."""

from __future__ import annotations

import ast
import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIFECYCLES = frozenset(
    {"supported", "compatibility", "diagnostic", "operational-job", "retired"}
)
CLEANUP_STATUSES = frozenset({"retain", "migrate", "deprecate", "delete", "unknown"})
RETENTION_CLASSES = frozenset(
    {
        "frozen",
        "canonical",
        "legacy",
        "research",
        "diagnostic",
        "runtime",
        "transient",
        "unknown",
    }
)
WRITE_AUTHORITIES = frozenset(
    {
        "replay",
        "legacy-runner",
        "application-command",
        "domain",
        "transient",
        "none",
        "unknown",
    }
)
COHORT_STATUSES = frozenset(
    {
        "pilot",
        "frozen-replay",
        "compatibility-pending",
        "supported-pending",
        "diagnostic-pending",
        "operational-pending",
    }
)
RETAINED_EVIDENCE = frozenset(
    {
        "d4_acceptance_handoff.json",
        "final_acceptance_handoff.json",
        "final_artifact_inventory.json",
        "reproducibility_manifest.json",
        "research_trial_inventory.parquet",
        "statistical_acceptance.parquet",
        "strategy_freeze_manifest.json",
        "trade_excursions.parquet",
    }
)
MATERIAL_IGNORED_PREFIXES = (".tokensave/", "data/", "reports/")


class ArchitectureValidationError(ValueError):
    """Raised when repository ownership or lifecycle metadata is invalid."""


@dataclass(frozen=True)
class RunnerRecord:
    path: str
    lifecycle: str
    owner: str
    invocation: str | None
    output_namespace: str | None
    compatibility_status: str
    migration_target: str | None
    notes: str | None
    retention_class: str
    write_authority: str
    retirement_condition: str
    cohort: str
    frozen_preservation: bool


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchitectureValidationError(f"cannot read {path}: {exc}") from exc


def load_runner_inventory(path: str | Path) -> tuple[RunnerRecord, ...]:
    payload = _json(Path(path))
    if not isinstance(payload, dict) or not isinstance(payload.get("runners"), list):
        raise ArchitectureValidationError(
            "runner inventory must contain a runners list"
        )
    records: list[RunnerRecord] = []
    seen: set[str] = set()
    matrix_by_path: dict[str, dict[str, Any]] = {}
    matrix_path = Path(path).parent / "runner_cohort_matrix.json"
    if matrix_path.is_file():
        matrix_payload = _json(matrix_path)
        matrix_entries = (
            matrix_payload.get("runners", [])
            if isinstance(matrix_payload, dict)
            else []
        )
        if isinstance(matrix_entries, list):
            matrix_by_path = {
                str(item.get("path", "")).replace("\\", "/"): item
                for item in matrix_entries
                if isinstance(item, dict)
            }
    defaults = payload.get("metadata_defaults", {})
    if not isinstance(defaults, dict):
        raise ArchitectureValidationError("metadata_defaults must be an object")
    retention_defaults = defaults.get("retention_class_by_lifecycle", {})
    authority_defaults = defaults.get("write_authority_by_lifecycle", {})
    review_defaults = defaults.get("retirement_condition_by_lifecycle", {})
    for raw in payload["runners"]:
        if not isinstance(raw, dict):
            raise ArchitectureValidationError(
                "runner inventory entries must be objects"
            )
        required = {"path", "lifecycle", "owner", "compatibility_status"}
        missing = required - raw.keys()
        if missing:
            raise ArchitectureValidationError(
                f"runner entry missing fields: {sorted(missing)}"
            )
        path_value = str(raw["path"]).replace("\\", "/")
        lifecycle = str(raw["lifecycle"])
        if path_value in seen:
            raise ArchitectureValidationError(f"duplicate runner entry: {path_value}")
        if lifecycle not in LIFECYCLES:
            raise ArchitectureValidationError(
                f"invalid lifecycle {lifecycle!r} for {path_value}"
            )
        seen.add(path_value)
        retention_class = str(
            raw.get("retention_class") or retention_defaults.get(lifecycle, "unknown")
        )
        write_authority = str(
            raw.get("write_authority") or authority_defaults.get(lifecycle, "unknown")
        )
        retirement_condition = str(
            raw.get("retirement_condition") or review_defaults.get(lifecycle, "")
        )
        matrix_entry = matrix_by_path.get(path_value, {})
        cohort = str(raw.get("cohort") or matrix_entry.get("cohort") or "unclassified")
        if retention_class not in RETENTION_CLASSES:
            raise ArchitectureValidationError(
                f"invalid retention class {retention_class!r} for {path_value}"
            )
        if write_authority not in WRITE_AUTHORITIES:
            raise ArchitectureValidationError(
                f"invalid write authority {write_authority!r} for {path_value}"
            )
        if not retirement_condition:
            raise ArchitectureValidationError(
                f"runner entry lacks retirement condition: {path_value}"
            )
        records.append(
            RunnerRecord(
                path=path_value,
                lifecycle=lifecycle,
                owner=str(raw["owner"]),
                invocation=(str(raw["invocation"]) if raw.get("invocation") else None),
                output_namespace=(
                    str(raw["output_namespace"])
                    if raw.get("output_namespace")
                    else None
                ),
                compatibility_status=str(raw["compatibility_status"]),
                migration_target=(
                    str(raw["migration_target"])
                    if raw.get("migration_target")
                    else None
                ),
                notes=str(raw["notes"]) if raw.get("notes") else None,
                retention_class=retention_class,
                write_authority=write_authority,
                retirement_condition=retirement_condition,
                cohort=cohort,
                frozen_preservation=bool(
                    raw.get("frozen_preservation", cohort == "frozen-replay")
                ),
            )
        )
    return tuple(records)


def tracked_runner_paths(root: str | Path) -> set[str]:
    root = Path(root)
    paths = {path.relative_to(root).as_posix() for path in root.glob("run_*.py")}
    paths.update(
        path.relative_to(root).as_posix()
        for path in (root / "jobs").glob("*.py")
        if path.name != "__init__.py"
    )
    return paths


def validate_runner_inventory(
    root: str | Path, inventory_path: str | Path
) -> tuple[RunnerRecord, ...]:
    root = Path(root)
    records = load_runner_inventory(inventory_path)
    listed = {record.path for record in records}
    actual = tracked_runner_paths(root)
    missing = sorted(actual - listed)
    stale = sorted(listed - actual)
    if missing or stale:
        details = []
        if missing:
            details.append(f"unclassified: {missing}")
        if stale:
            details.append(f"missing files: {stale}")
        raise ArchitectureValidationError("; ".join(details))
    matrix_path = root / "docs" / "architecture" / "runner_cohort_matrix.json"
    if matrix_path.is_file():
        matrix = _json(matrix_path)
        entries = matrix.get("runners") if isinstance(matrix, dict) else None
        if not isinstance(entries, list):
            raise ArchitectureValidationError(
                "runner cohort matrix must contain runners"
            )
        matrix_paths = {
            str(item.get("path", "")).replace("\\", "/")
            for item in entries
            if isinstance(item, dict)
        }
        if matrix_paths != actual:
            raise ArchitectureValidationError(
                "runner cohort matrix mismatch: "
                f"missing={sorted(actual - matrix_paths)}, "
                f"stale={sorted(matrix_paths - actual)}"
            )
        for item in entries:
            if not isinstance(item, dict) or item.get("cohort") not in COHORT_STATUSES:
                raise ArchitectureValidationError(
                    "runner cohort entries require a valid cohort"
                )
            if not item.get("blockers"):
                raise ArchitectureValidationError(
                    "runner cohort entries require blockers/evidence"
                )
    elif root.resolve() == Path(inventory_path).resolve().parents[2]:
        raise ArchitectureValidationError(
            f"missing runner cohort matrix: {matrix_path}"
        )
    for record in records:
        if record.migration_target and not _module_exists(
            root, record.migration_target
        ):
            raise ArchitectureValidationError(
                "migration target does not exist: "
                f"{record.path} -> {record.migration_target}"
            )
        if record.compatibility_status == "adapter":
            imports = {module for _, module in _iter_imports(root / record.path)}
            if not record.migration_target or record.migration_target not in imports:
                raise ArchitectureValidationError(
                    f"compatibility adapter does not delegate to target: {record.path}"
                )
        if record.lifecycle != "retired" and record.cohort == "unclassified":
            raise ArchitectureValidationError(f"runner lacks cohort: {record.path}")
    validate_frozen_write_policy(records)
    supported_owners = [
        record.owner for record in records if record.lifecycle == "supported"
    ]
    if len(supported_owners) != len(set(supported_owners)):
        raise ArchitectureValidationError(
            "duplicate owning implementation for supported runner"
        )
    return records


def _module_exists(root: Path, module: str) -> bool:
    if not module.startswith("twse_factor_lab."):
        return False
    relative = Path("src") / Path(*module.split("."))
    return (root / relative.with_suffix(".py")).is_file() or (
        root / relative / "__init__.py"
    ).is_file()


def validate_frozen_write_policy(records: Iterable[RunnerRecord]) -> None:
    for record in records:
        if not record.migration_target or not record.migration_target.startswith(
            "twse_factor_lab.application.commands."
        ):
            continue
        if record.lifecycle not in {
            "supported",
            "compatibility",
            "diagnostic",
            "operational-job",
        }:
            raise ArchitectureValidationError(
                f"command target has invalid lifecycle: {record.path}"
            )
        if record.output_namespace.startswith(
            ("data/processed/", "reports/final/", "reports/rc1/")
        ):
            raise ArchitectureValidationError(
                f"active command targets frozen namespace: {record.path}"
            )


def _iter_imports(path: Path) -> Iterable[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ArchitectureValidationError(f"cannot parse {path}: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module


def validate_dependency_direction(root: str | Path) -> None:
    """Reject domain imports that point back into application/presentation code."""

    source_root = Path(root) / "src" / "twse_factor_lab"
    violations: list[str] = []
    for path in source_root.glob("**/*.py"):
        relative = path.relative_to(source_root).as_posix()
        if relative.startswith("application/"):
            continue
        for line, module in _iter_imports(path):
            prohibited = (
                module == "twse_factor_lab.application"
                or module.startswith("twse_factor_lab.application.")
                or module.startswith("twse_factor_lab.reporting.render_")
                or module.startswith("run_")
            )
            if prohibited:
                violations.append(
                    f"{path.relative_to(root).as_posix()}:{line} -> {module}"
                )
    if violations:
        raise ArchitectureValidationError(
            "prohibited dependency direction: " + ", ".join(violations)
        )


def validate_cleanup_ledger(path: str | Path) -> None:
    payload = _json(Path(path))
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ArchitectureValidationError(
            "cleanup ledger must contain a candidates list"
        )
    seen: set[str] = set()
    protected = payload.get("protected_paths", [])
    if not isinstance(protected, list):
        raise ArchitectureValidationError("protected_paths must be a list")
    for item in protected:
        if (
            not isinstance(item, dict)
            or not item.get("path")
            or not item.get("evidence")
        ):
            raise ArchitectureValidationError(
                "protected paths require path and evidence"
            )
        if item.get("action") != "no-move-no-delete":
            raise ArchitectureValidationError(
                f"protected path lacks no-move-no-delete action: {item.get('path')}"
            )
    for raw in payload["candidates"]:
        if not isinstance(raw, dict) or not raw.get("path") or not raw.get("status"):
            raise ArchitectureValidationError("cleanup entries require path and status")
        candidate = str(raw["path"]).replace("\\", "/")
        status = str(raw["status"])
        if candidate in seen:
            raise ArchitectureValidationError(
                f"duplicate cleanup candidate: {candidate}"
            )
        if status not in CLEANUP_STATUSES:
            raise ArchitectureValidationError(f"invalid cleanup status {status!r}")
        if status == "delete" and candidate in RETAINED_EVIDENCE:
            raise ArchitectureValidationError(
                f"retained evidence cannot be deleted: {candidate}"
            )
        if status == "delete" and not raw.get("evidence"):
            raise ArchitectureValidationError(
                f"delete candidate lacks evidence: {candidate}"
            )
        if status == "migrate":
            relocation = raw.get("relocation")
            required = {"source", "target", "hash_impact", "rollback", "verification"}
            if not isinstance(relocation, dict) or not required <= relocation.keys():
                raise ArchitectureValidationError(
                    f"migrate candidate lacks relocation evidence: {candidate}"
                )
        assessment = raw.get("cleanup_assessment")
        if assessment is not None:
            required = {
                "size_bytes",
                "references_searched",
                "active_workflow",
                "retention_result",
                "disposition",
                "rollback",
            }
            if not isinstance(assessment, dict) or not required <= assessment.keys():
                raise ArchitectureValidationError(
                    f"cleanup assessment lacks evidence: {candidate}"
                )
        seen.add(candidate)


def run_architecture_checks(root: str | Path) -> None:
    root = Path(root).resolve()
    validate_runner_inventory(
        root, root / "docs" / "architecture" / "runner_inventory.json"
    )
    validate_cleanup_ledger(root / "docs" / "architecture" / "cleanup_ledger.json")
    validate_artifact_inventory(
        root / "docs" / "architecture" / "artifact_inventory.json"
    )
    validate_tracked_ignored_artifacts(
        root, root / "docs" / "architecture" / "artifact_inventory.json"
    )
    validate_dependency_direction(root)


def validate_artifact_inventory(path: str | Path) -> None:
    payload = _json(Path(path))
    entries = payload.get("namespaces") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ArchitectureValidationError("artifact inventory must contain namespaces")
    required = {
        "path",
        "classification",
        "owner",
        "retention",
        "write_authority",
        "evidence",
    }
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not required <= entry.keys():
            raise ArchitectureValidationError(
                "artifact namespace entries lack ownership metadata"
            )
        path_value = str(entry["path"]).replace("\\", "/")
        if path_value in seen:
            raise ArchitectureValidationError(
                f"duplicate artifact namespace: {path_value}"
            )
        if entry["classification"] not in {
            "frozen",
            "canonical",
            "research",
            "diagnostic",
            "runtime",
            "transient",
            "unknown",
        }:
            raise ArchitectureValidationError(
                f"invalid artifact classification: {path_value}"
            )
        if entry["classification"] == "frozen" and entry["write_authority"] != "replay":
            raise ArchitectureValidationError(
                f"frozen artifact must be replay-owned: {path_value}"
            )
        seen.add(path_value)


def tracked_ignored_material_paths(root: str | Path) -> tuple[str, ...]:
    """Return tracked ignored data, reports, and local tool metadata paths."""

    root = Path(root)
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-ci", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise ArchitectureValidationError(
            f"cannot list tracked ignored paths: {completed.stderr.strip()}"
        )
    return tuple(
        path
        for path in completed.stdout.splitlines()
        if path.startswith(MATERIAL_IGNORED_PREFIXES)
    )


def validate_tracked_ignored_artifacts(
    root: str | Path, inventory_path: str | Path
) -> None:
    """Require material tracked-ignored files to be covered by a namespace."""

    payload = _json(Path(inventory_path))
    namespaces = payload.get("namespaces", []) if isinstance(payload, dict) else []
    declared = [
        str(entry.get("path", "")).replace("\\", "/").rstrip("/")
        for entry in namespaces
        if isinstance(entry, dict)
    ]
    missing = [
        path
        for path in tracked_ignored_material_paths(root)
        if not any(path == item or path.startswith(f"{item}/") for item in declared)
    ]
    if missing:
        raise ArchitectureValidationError(
            f"unclassified tracked ignored artifacts: {sorted(missing)}"
        )


__all__ = [
    "ArchitectureValidationError",
    "CLEANUP_STATUSES",
    "LIFECYCLES",
    "RunnerRecord",
    "load_runner_inventory",
    "run_architecture_checks",
    "tracked_runner_paths",
    "validate_cleanup_ledger",
    "validate_artifact_inventory",
    "validate_tracked_ignored_artifacts",
    "validate_dependency_direction",
    "validate_frozen_write_policy",
    "validate_runner_inventory",
    "tracked_ignored_material_paths",
]

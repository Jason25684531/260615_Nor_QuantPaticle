"""Deterministic repository-boundary and cleanup-inventory checks."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIFECYCLES = frozenset(
    {"supported", "compatibility", "diagnostic", "operational-job", "retired"}
)
CLEANUP_STATUSES = frozenset(
    {"retain", "migrate", "deprecate", "delete", "unknown"}
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
    supported_owners = [
        record.owner for record in records if record.lifecycle == "supported"
    ]
    if len(supported_owners) != len(set(supported_owners)):
        raise ArchitectureValidationError(
            "duplicate owning implementation for supported runner"
        )
    return records


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
        seen.add(candidate)


def run_architecture_checks(root: str | Path) -> None:
    root = Path(root).resolve()
    validate_runner_inventory(
        root, root / "docs" / "architecture" / "runner_inventory.json"
    )
    validate_cleanup_ledger(root / "docs" / "architecture" / "cleanup_ledger.json")
    validate_dependency_direction(root)


__all__ = [
    "ArchitectureValidationError",
    "CLEANUP_STATUSES",
    "LIFECYCLES",
    "RunnerRecord",
    "load_runner_inventory",
    "run_architecture_checks",
    "tracked_runner_paths",
    "validate_cleanup_ledger",
    "validate_dependency_direction",
    "validate_runner_inventory",
]

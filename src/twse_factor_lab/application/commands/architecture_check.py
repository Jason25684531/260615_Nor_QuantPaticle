"""Validate the repository architecture contract."""

from __future__ import annotations

from pathlib import Path

from twse_factor_lab.application.architecture import (
    ArchitectureValidationError,
    load_runner_inventory,
    run_architecture_checks,
    tracked_runner_paths,
)
from twse_factor_lab.application.support import repository_root


def main(root: str | Path | None = None) -> int:
    root = Path(root).resolve() if root else repository_root(__file__)
    try:
        run_architecture_checks(root)
    except ArchitectureValidationError as exc:
        print("ARCHITECTURE_CLOSURE")
        print(f"error = {exc}")
        print("FINAL_STATUS = FAIL")
        return 1

    records = load_runner_inventory(
        root / "docs" / "architecture" / "runner_inventory.json"
    )
    actual = tracked_runner_paths(root)
    listed = {record.path for record in records}
    print("ARCHITECTURE_CLOSURE")
    for check in (
        "runner_inventory",
        "runner_cohorts",
        "cleanup_ledger",
        "artifact_inventory",
        "dependency_direction",
        "frozen_policy",
    ):
        print(f"{check} = PASS")
    print(f"runner_total = {len(actual)}")
    print(f"unclassified = {len(actual - listed)}")
    print("FINAL_STATUS = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

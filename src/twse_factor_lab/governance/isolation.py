"""Frozen RC1 baseline isolation guard for new research-cycle artifacts."""

from __future__ import annotations

from pathlib import Path

from twse_factor_lab.governance.schemas import GovernanceError

RESEARCH_NAMESPACE = Path("data") / "research"

# Frozen RC1 evidence: never created, modified, or overwritten by governance.
FROZEN_FILES = (
    "strategy_freeze_manifest.json",
    "reproducibility_manifest.json",
    "d4_acceptance_handoff.json",
    "final_acceptance_handoff.json",
    "final_artifact_inventory.json",
    "research_trial_inventory.parquet",
    "statistical_acceptance.parquet",
    "trade_excursions.parquet",
)
FROZEN_DIRS = (
    Path("data") / "processed",
    Path("reports") / "final",
    Path("openspec") / "changes" / "archive",
)

# Research identity of the frozen RC1 cycle; a new cycle must never reuse it.
FROZEN_RESEARCH_IDS = frozenset({"multi-factor-research-v1"})


class IsolationError(GovernanceError):
    pass


def assert_research_id_allowed(research_id: str) -> None:
    if research_id in FROZEN_RESEARCH_IDS:
        raise IsolationError(
            f"research_id {research_id!r} is the frozen RC1 research identity"
        )


def assert_write_allowed(path: str | Path, root: str | Path) -> Path:
    """Return the resolved target if it lies inside the research namespace."""
    root = Path(root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise IsolationError(f"write outside repository root: {target}") from exc
    if relative.name in FROZEN_FILES and len(relative.parts) == 1:
        raise IsolationError(f"write to frozen RC1 evidence refused: {relative}")
    for frozen_dir in FROZEN_DIRS:
        if relative.parts[: len(frozen_dir.parts)] == frozen_dir.parts:
            raise IsolationError(
                f"write into frozen RC1 location refused: {relative}"
            )
    if relative.parts[: len(RESEARCH_NAMESPACE.parts)] != RESEARCH_NAMESPACE.parts:
        raise IsolationError(
            f"governance artifacts must live under {RESEARCH_NAMESPACE}: {relative}"
        )
    return target

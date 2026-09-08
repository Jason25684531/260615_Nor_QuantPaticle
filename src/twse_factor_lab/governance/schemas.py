"""Typed governance contracts for new research cycles (post-RC1)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

RESEARCH_STATUSES = frozenset({"draft", "active", "completed", "abandoned"})
EXPERIMENT_STATUSES = frozenset(
    {"planned", "running", "completed", "failed", "aborted"}
)
EXPERIMENT_TYPES = frozenset(
    {"factor_test", "strategy_backtest", "robustness", "attribution", "diagnostic"}
)


class GovernanceError(ValueError):
    pass


def _require_slug(value: str, name: str) -> None:
    if not value or not ID_PATTERN.match(value):
        raise GovernanceError(f"{name} must be a non-empty filesystem-safe slug")


def _require_nonempty(value: str, name: str) -> None:
    if not value:
        raise GovernanceError(f"{name} must be non-empty")


def _require_iso_date(value: str, name: str) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise GovernanceError(f"{name} must be an ISO date: {value!r}") from exc


@dataclass(frozen=True)
class ResearchManifest:
    research_id: str
    hypothesis: str
    factor_candidates: list[str]
    universe: str
    dataset_version: str
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    rebalance_search_space: list[str]
    top_n_search_space: list[int]
    cost_scenarios: list[str]
    selection_relevant: bool
    status: str

    def validate(self) -> None:
        _require_slug(self.research_id, "research_id")
        _require_nonempty(self.hypothesis, "hypothesis")
        _require_nonempty(self.universe, "universe")
        _require_nonempty(self.dataset_version, "dataset_version")
        if not self.factor_candidates:
            raise GovernanceError("factor_candidates must be non-empty")
        for name in ("is_start", "is_end", "oos_start", "oos_end"):
            _require_iso_date(getattr(self, name), name)
        if not (self.is_start < self.is_end < self.oos_start <= self.oos_end):
            raise GovernanceError(
                "invalid IS/OOS ordering: require "
                "is_start < is_end < oos_start <= oos_end"
            )
        if self.status not in RESEARCH_STATUSES:
            raise GovernanceError(f"unknown research status: {self.status!r}")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    source: str
    retrieved_at: str
    schema_version: str
    processing_version: str
    pit_rule: str
    row_count: int
    coverage: float
    artifact_sha256: str

    def validate(self) -> None:
        _require_slug(self.dataset_id, "dataset_id")
        _require_nonempty(self.source, "source")
        _require_nonempty(self.retrieved_at, "retrieved_at")
        _require_nonempty(self.schema_version, "schema_version")
        _require_nonempty(self.processing_version, "processing_version")
        _require_nonempty(self.pit_rule, "pit_rule")
        _require_nonempty(self.artifact_sha256, "artifact_sha256")
        if self.row_count < 0:
            raise GovernanceError("row_count must be >= 0")
        if not 0.0 <= self.coverage <= 1.0:
            raise GovernanceError("coverage must be within [0, 1]")


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    research_id: str
    config: dict[str, Any]
    dataset_version: str
    experiment_type: str
    status: str
    selection_relevant: bool
    result: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_slug(self.experiment_id, "experiment_id")
        _require_slug(self.research_id, "research_id")
        _require_nonempty(self.dataset_version, "dataset_version")
        if self.experiment_type not in EXPERIMENT_TYPES:
            raise GovernanceError(
                f"unknown experiment type: {self.experiment_type!r}"
            )
        if self.status not in EXPERIMENT_STATUSES:
            raise GovernanceError(f"unknown experiment status: {self.status!r}")

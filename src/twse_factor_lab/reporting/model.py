"""Pure data model used by the research report renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SECTION_NAMES = (
    "research",
    "dataset",
    "universe_coverage",
    "factor_evidence",
    "factor_ablation",
    "strategy_trials",
    "locked_candidate",
    "cost_turnover",
    "execution_validation",
    "performance",
    "performance_metric_definition",
    "pyfolio",
    "attribution",
    "robustness",
    "breadth_overlay",
    "oos",
    "validation",
    "statistics",
    "acceptance",
    "reproducibility",
    "limitations",
)


@dataclass(frozen=True)
class ReportSection:
    """One normalized report section; values are copied, never evaluated."""

    status: str = "AVAILABLE"
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "data": self.data}


def _unavailable() -> ReportSection:
    return ReportSection(status="UNAVAILABLE")


@dataclass(frozen=True)
class ResearchReportModel:
    """Renderer-neutral research report model.

    The v1 core sections are required; the Research Cycle v2 sections default to
    UNAVAILABLE so a v1 frozen cycle renders gracefully without loader changes.
    """

    research: ReportSection
    dataset: ReportSection
    factor_evidence: ReportSection
    strategy_trials: ReportSection
    locked_candidate: ReportSection
    execution_validation: ReportSection
    performance: ReportSection
    pyfolio: ReportSection
    attribution: ReportSection
    robustness: ReportSection
    oos: ReportSection
    statistics: ReportSection
    acceptance: ReportSection
    reproducibility: ReportSection
    limitations: ReportSection
    # Research Cycle v2 additions (optional; default UNAVAILABLE for v1 cycles).
    universe_coverage: ReportSection = field(default_factory=_unavailable)
    factor_ablation: ReportSection = field(default_factory=_unavailable)
    cost_turnover: ReportSection = field(default_factory=_unavailable)
    performance_metric_definition: ReportSection = field(default_factory=_unavailable)
    breadth_overlay: ReportSection = field(default_factory=_unavailable)
    validation: ReportSection = field(default_factory=_unavailable)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in SECTION_NAMES}


__all__ = ["SECTION_NAMES", "ReportSection", "ResearchReportModel"]

"""Phase 10 tests: report model carries and renders the v2 sections."""

from __future__ import annotations

from twse_factor_lab.reporting.model import (
    SECTION_NAMES,
    ReportSection,
    ResearchReportModel,
)
from twse_factor_lab.reporting.render_md import render_markdown

V2_SECTIONS = (
    "universe_coverage",
    "factor_ablation",
    "cost_turnover",
    "performance_metric_definition",
    "breadth_overlay",
    "validation",
)


def _core() -> dict[str, ReportSection]:
    return {
        name: ReportSection(data={})
        for name in (
            "research", "dataset", "factor_evidence", "strategy_trials",
            "locked_candidate", "execution_validation", "performance", "pyfolio",
            "attribution", "robustness", "oos", "statistics", "acceptance",
            "reproducibility", "limitations",
        )
    }


def test_all_v2_sections_are_named() -> None:
    for section in V2_SECTIONS:
        assert section in SECTION_NAMES
    assert len(SECTION_NAMES) == 21


def test_v1_model_defaults_v2_sections_to_unavailable() -> None:
    model = ResearchReportModel(**_core())
    rendered = render_markdown(model)
    for heading in (
        "Universe / Data Coverage",
        "Factor Ablation",
        "Cost & Turnover Analysis",
        "Performance Metric Definition",
        "Market Breadth Risk Overlay",
        "Validation (Walk-forward)",
    ):
        assert heading in rendered
    # Defaults are UNAVAILABLE for a v1-style model.
    assert model.breadth_overlay.status == "UNAVAILABLE"


def test_populated_v2_sections_render_data() -> None:
    core = _core()
    model = ResearchReportModel(
        **core,
        universe_coverage=ReportSection(data={
            "methodology": "listed -> tradable -> liquid -> fundamental",
            "mean_universe_count": 5.4,
            "liquidity_measure_source": "proxy_close_times_volume",
        }),
        factor_ablation=ReportSection(data={"rows": [
            {"configuration": "eps-only", "factor_gate": "REJECT",
             "is_net_return": 0.38, "sharpe": 0.47, "sortino": None,
             "mdd": -0.2, "turnover": 0.01},
        ]}),
        performance_metric_definition=ReportSection(data={
            "metric_schema_version": "canonical-metrics-v2",
            "annualization_factor": 252,
        }),
        validation=ReportSection(data={
            "label": "WALK-FORWARD VALIDATION",
            "rows": [{"fold": 2022, "test_start": "2022-01-03",
                      "test_end": "2022-12-30", "cagr": 0.1, "sharpe": 0.5,
                      "sortino": 0.6, "mdd": -0.1}],
        }),
    )
    rendered = render_markdown(model)
    assert "proxy_close_times_volume" in rendered
    assert "canonical-metrics-v2" in rendered
    assert "WALK-FORWARD VALIDATION" in rendered
    assert "FRESH UNTOUCHED OOS" not in rendered
    assert "eps-only" in rendered

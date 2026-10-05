from __future__ import annotations

from twse_factor_lab.reporting.performance_adapter import (
    load_repository_performance_data,
)
from twse_factor_lab.reporting.performance_tearsheet import (
    generate_performance_report,
)


def test_no_fresh_oos_is_normal_and_cannot_promote(performance_root):
    data = load_repository_performance_data(performance_root)
    report = generate_performance_report(
        performance_root,
        output_format="html",
        data_override=data,
    )
    assert report.summary["fresh_oos_status"] == "INSUFFICIENT_DATA"
    assert report.summary["fresh_oos_reason"] == (
        "NO_LEGAL_UNTOUCHED_WINDOW_WITH_REQUIRED_HISTORY"
    )
    assert report.summary["performance_evidence"] == "INSUFFICIENT_DATA"
    assert report.summary["promotion_gate"]["production_ready"] is False
    final = (
        performance_root
        / "data/research/fundamental-production-final-v1"
        / "final_strategy_validation_report.md"
    ).read_text(encoding="utf-8")
    assert "Fresh OOS = INSUFFICIENT_DATA" in final
    assert "Production Ready = NO" in final

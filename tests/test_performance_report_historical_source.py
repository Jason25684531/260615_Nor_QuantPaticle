from __future__ import annotations

import json
from pathlib import Path


def test_performance_report_surfaces_historical_evidence_without_promoting() -> None:
    root = Path("data/research/fundamental-production-final-v1")
    summary = json.loads(
        (root / "performance/performance_summary.json").read_text(encoding="utf-8")
    )
    report = (root / "final_strategy_validation_report.md").read_text(encoding="utf-8")
    assert summary["periods"]["backtest"]["status"] == "AVAILABLE"
    assert summary["historical_evidence"] == "HISTORICALLY_SUPPORTIVE"
    assert "Historical Evidence = HISTORICALLY_SUPPORTIVE" in report
    assert "Production Ready = NO" in report


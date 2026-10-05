from __future__ import annotations

import json

import pytest

from twse_factor_lab.reporting.performance_adapter import (
    FrozenStrategyChanged,
    PerformanceData,
)
from twse_factor_lab.reporting.performance_tearsheet import (
    generate_performance_report,
)


def _unavailable_data():
    return {
        period: PerformanceData.unavailable(period, "missing")
        for period in ("BACKTEST", "HISTORICAL_OOS", "FRESH_OOS")
    }


def test_report_generation_and_idempotent_core_metrics(performance_root):
    first = generate_performance_report(
        performance_root,
        output_format="html",
        data_override=_unavailable_data(),
    )
    stale_pdf = first.output_dir / "performance_tearsheet_report.pdf"
    stale_pdf.write_bytes(b"stale")
    second = generate_performance_report(
        performance_root,
        output_format="html",
        force=True,
        data_override=_unavailable_data(),
    )
    for name in (
        "performance_summary.json",
        "performance_metrics.csv",
        "chart_interpretations.json",
        "performance_tearsheet_report.html",
    ):
        assert (first.output_dir / name).exists()
    assert not list((first.output_dir / "charts").glob("*.png"))
    assert not stale_pdf.exists()
    assert first.summary["periods"] == second.summary["periods"]
    rendered = (first.output_dir / "performance_tearsheet_report.html").read_text(
        encoding="utf-8"
    )
    assert rendered.count("<h2>") == 21
    summary = json.loads(
        (first.output_dir / "performance_summary.json").read_text(encoding="utf-8")
    )
    assert summary["performance_evidence"] == "INSUFFICIENT_DATA"


def test_changed_frozen_contract_blocks_report(performance_root):
    contract = (
        performance_root
        / "data/research/fundamental-production-final-v1"
        / "production_runtime_contract.json"
    )
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["top_n"] = 6
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrozenStrategyChanged, match="FROZEN_STRATEGY_CHANGED"):
        generate_performance_report(
            performance_root,
            output_format="html",
            data_override=_unavailable_data(),
        )

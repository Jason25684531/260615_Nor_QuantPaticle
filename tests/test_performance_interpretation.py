from __future__ import annotations

from twse_factor_lab.reporting.performance_adapter import PerformanceData
from twse_factor_lab.reporting.performance_metrics import calculate_metrics
from twse_factor_lab.reporting.performance_tearsheet import PerformanceCharts


def test_every_chart_has_five_part_grounded_interpretation(tmp_path):
    data = {
        period: PerformanceData.unavailable(period, "canonical source missing")
        for period in ("BACKTEST", "HISTORICAL_OOS", "FRESH_OOS")
    }
    metrics = {period: calculate_metrics(value) for period, value in data.items()}
    records = PerformanceCharts.build(data, metrics, tmp_path).interpretations()
    assert len(records) == 24
    for record in records:
        assert record["status"] == "UNAVAILABLE"
        assert record["reason"]
        assert len(record["sections"]) == 5
        assert "未建立空白或推測圖表" in record["interpretation"]

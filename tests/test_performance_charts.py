from __future__ import annotations

import matplotlib.pyplot as plt

from twse_factor_lab.reporting.performance_adapter import PerformanceData
from twse_factor_lab.reporting.performance_metrics import calculate_metrics
from twse_factor_lab.reporting.performance_tearsheet import PerformanceCharts


def test_charts_are_conditional_headless_and_non_empty(
    tmp_path, sample_performance_data
):
    data = {
        "BACKTEST": sample_performance_data,
        "HISTORICAL_OOS": PerformanceData.unavailable("HISTORICAL_OOS", "missing"),
        "FRESH_OOS": PerformanceData.unavailable("FRESH_OOS", "missing"),
    }
    metrics = {period: calculate_metrics(value) for period, value in data.items()}
    charts = PerformanceCharts.build(data, metrics, tmp_path, ("BACKTEST",))
    assert len(charts.results) == 24
    assert all(item.path.stat().st_size > 0 for item in charts.generated)
    assert next(
        item for item in charts.results if item.chart_id == "22_backtest_vs_oos"
    ).status == "UNAVAILABLE"
    assert plt.get_fignums() == []


def test_missing_positions_create_no_fake_exposure_chart(tmp_path):
    data = {
        period: PerformanceData.unavailable(period, "missing")
        for period in ("BACKTEST", "HISTORICAL_OOS", "FRESH_OOS")
    }
    metrics = {period: calculate_metrics(value) for period, value in data.items()}
    charts = PerformanceCharts.build(data, metrics, tmp_path)
    exposure = next(
        item for item in charts.results if item.chart_id == "15_gross_exposure"
    )
    assert exposure.status == "UNAVAILABLE"
    assert not (tmp_path / "15_gross_exposure.png").exists()

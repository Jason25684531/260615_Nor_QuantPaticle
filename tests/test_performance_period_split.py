from __future__ import annotations

import pandas as pd

from twse_factor_lab.reporting.performance_adapter import PerformanceData
from twse_factor_lab.reporting.performance_metrics import calculate_metrics
from twse_factor_lab.reporting.performance_tearsheet import (
    PerformanceCharts,
    classify_performance_evidence,
)


def test_periods_are_calculated_separately_and_compared(tmp_path):
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    returns = {
        "BACKTEST": pd.Series(0.002, index=dates),
        "HISTORICAL_OOS": pd.Series(0.001, index=dates),
        "FRESH_OOS": pd.Series(0.0005, index=dates),
    }
    data = {
        period: PerformanceData.from_frames(period, values)
        for period, values in returns.items()
    }
    metrics = {period: calculate_metrics(value) for period, value in data.items()}
    totals = [metrics[period].metrics["total_return"].value for period in returns]
    assert len(set(totals)) == 3
    charts = PerformanceCharts.build(data, metrics, tmp_path)
    assert all(
        next(item for item in charts.results if item.chart_id == chart).status
        == "GENERATED"
        for chart in (
            "22_backtest_vs_oos",
            "23_historical_oos_vs_fresh_oos",
            "24_all_period_comparison",
        )
    )
    assert classify_performance_evidence(data, metrics) == "SUPPORTIVE"

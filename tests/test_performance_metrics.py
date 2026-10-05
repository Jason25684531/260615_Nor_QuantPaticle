from __future__ import annotations

import pandas as pd
import pytest

from twse_factor_lab.reporting.performance_adapter import PerformanceData
from twse_factor_lab.reporting.performance_metrics import calculate_metrics


def test_drawdown_recovery_and_canonical_metrics():
    returns = pd.Series(
        [0.10, -0.20, 0.25, 0.01],
        index=pd.date_range("2024-01-02", periods=4),
    )
    result = calculate_metrics(PerformanceData.from_frames("BACKTEST", returns))
    assert result.metrics["max_drawdown"].value == pytest.approx(-0.2)
    assert result.drawdowns[0]["recovery"] != "NOT_RECOVERED"
    assert result.metrics["total_return"].status == "AVAILABLE"


def test_zero_volatility_is_null_not_zero():
    returns = pd.Series(0.0, index=pd.date_range("2024-01-02", periods=10))
    result = calculate_metrics(PerformanceData.from_frames("BACKTEST", returns))
    assert result.metrics["sharpe"].value is None
    assert result.metrics["sharpe"].status == "INSUFFICIENT_DATA"


def test_cost_turnover_concentration_and_round_trips(sample_performance_data):
    result = calculate_metrics(sample_performance_data)
    assert result.metrics["transaction_cost"].value == 645.0
    assert result.metrics["actual_rebalance_count"].value == 2
    assert result.metrics["turnover"].status == "AVAILABLE"
    assert result.metrics["hhi"].value == pytest.approx(0.64)
    assert result.metrics["round_trip_win_rate"].value == 1.0

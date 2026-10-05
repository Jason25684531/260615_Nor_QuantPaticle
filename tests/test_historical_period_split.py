from __future__ import annotations

from twse_factor_lab.reporting.performance_adapter import (
    load_repository_performance_data,
)


def test_historical_backtest_does_not_become_oos_or_fresh_oos() -> None:
    data = load_repository_performance_data(".")
    assert data["BACKTEST"].status == "AVAILABLE"
    assert data["HISTORICAL_OOS"].status == "INSUFFICIENT_DATA"
    assert data["FRESH_OOS"].status == "INSUFFICIENT_DATA"

"""Execution dates must come from the supplied trading calendar."""

import pandas as pd

from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar


def test_t1_uses_next_trading_day():
    days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    calendar = build_rebalance_calendar(days, execution_lag_days=1)

    assert calendar.iloc[0].execution_date == pd.Timestamp("2024-01-03")


def test_friday_signal_executes_next_trading_session():
    days = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08"])
    calendar = build_rebalance_calendar(days, execution_lag_days=1)

    assert calendar.iloc[1].execution_date == pd.Timestamp("2024-01-08")


def test_holiday_gap_uses_next_trading_session():
    days = pd.DatetimeIndex(["2024-02-06", "2024-02-07", "2024-02-15"])
    calendar = build_rebalance_calendar(days, execution_lag_days=1)

    assert calendar.iloc[1].execution_date == pd.Timestamp("2024-02-15")


def test_execution_date_is_always_in_trading_calendar():
    days = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08"])
    calendar = build_rebalance_calendar(days, frequency="weekly", execution_lag_days=1)

    assert set(calendar["execution_date"]).issubset(days)

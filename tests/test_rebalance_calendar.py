"""Tests for rebalance calendar generation."""

from __future__ import annotations

import pandas as pd
import pytest

from twse_factor_lab.portfolio.rebalance import (
    build_rebalance_calendar,
    validate_factor_eligibility,
)


def _make_trading_days(n: int = 500) -> pd.DatetimeIndex:
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DatetimeIndex(dates)


class TestDailyCalendar:
    def test_every_day_is_signal_date(self) -> None:
        days = _make_trading_days(20)
        cal = build_rebalance_calendar(days, frequency="daily", execution_lag_days=1)
        assert len(cal) == len(days) - 1

    def test_execution_after_signal(self) -> None:
        days = _make_trading_days(20)
        cal = build_rebalance_calendar(days, frequency="daily", execution_lag_days=1)
        assert (cal["execution_date"] > cal["signal_date"]).all()

    def test_frequency_column(self) -> None:
        days = _make_trading_days(10)
        cal = build_rebalance_calendar(days, frequency="daily")
        assert (cal["rebalance_frequency"] == "daily").all()


class TestWeeklyCalendar:
    def test_one_signal_per_week(self) -> None:
        days = _make_trading_days(260)
        cal = build_rebalance_calendar(days, frequency="weekly", execution_lag_days=1)
        sig = pd.to_datetime(cal["signal_date"])
        week_keys = sig.apply(lambda d: (d.isocalendar().year, d.isocalendar().week))
        assert week_keys.nunique() == len(cal)

    def test_execution_after_signal(self) -> None:
        days = _make_trading_days(100)
        cal = build_rebalance_calendar(days, frequency="weekly", execution_lag_days=1)
        assert (cal["execution_date"] > cal["signal_date"]).all()

    def test_fewer_rows_than_daily(self) -> None:
        days = _make_trading_days(260)
        daily = build_rebalance_calendar(days, frequency="daily")
        weekly = build_rebalance_calendar(days, frequency="weekly")
        assert len(weekly) < len(daily)


class TestMonthlyCalendar:
    def test_one_signal_per_month(self) -> None:
        days = _make_trading_days(500)
        cal = build_rebalance_calendar(days, frequency="monthly", execution_lag_days=1)
        sig = pd.to_datetime(cal["signal_date"])
        month_keys = sig.apply(lambda d: (d.year, d.month))
        assert month_keys.nunique() == len(cal)

    def test_execution_after_signal(self) -> None:
        days = _make_trading_days(500)
        cal = build_rebalance_calendar(days, frequency="monthly", execution_lag_days=1)
        assert (cal["execution_date"] > cal["signal_date"]).all()


class TestValidateFactorEligibility:
    def test_snapshot_factor_raises(self) -> None:
        for f in [
            "pb_inverse",
            "pe_inverse",
            "dividend_yield",
            "latest_snapshot_mixed",
        ]:
            with pytest.raises(ValueError):
                validate_factor_eligibility(f)

    def test_historical_factor_ok(self) -> None:
        validate_factor_eligibility("historical_price_volume")

    def test_unknown_factor_ok(self) -> None:
        validate_factor_eligibility("some_other_historical_factor")

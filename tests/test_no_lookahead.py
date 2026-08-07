import pandas as pd

from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar
from twse_factor_lab.portfolio.selection import build_topn_positions
from twse_factor_lab.validation.no_lookahead import (
    describe_snapshot_limitation,
    uses_backward_looking_windows,
)


def test_uses_backward_looking_windows_accepts_positive_lookbacks():
    factor = pd.DataFrame(
        {"1101": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3)
    )

    result = uses_backward_looking_windows(factor, lookback=20)

    assert result["is_backward_looking"] is True
    assert result["lookback"] == 20


def test_describe_snapshot_limitation_mentions_empty_valuation_dates():
    valuation = pd.DataFrame({"date": [pd.NaT], "ticker": ["1101"]})

    message = describe_snapshot_limitation(valuation)

    assert "snapshot" in message.lower()
    assert "date" in message.lower()


def test_execution_date_is_after_signal_date():
    calendar = build_rebalance_calendar(pd.bdate_range("2024-01-01", periods=3))

    assert (calendar["execution_date"] > calendar["signal_date"]).all()


def test_signal_uses_no_future_prices():
    date = pd.Timestamp("2024-01-02")
    next_date = date + pd.Timedelta(days=1)
    factors = pd.DataFrame(
        {
            "date": [date, date, next_date, next_date],
            "ticker": ["A", "B", "A", "B"],
            "composite_score": [2.0, 1.0, 0.0, 99.0],
            "composite_type": ["historical_price_volume"] * 4,
            "is_snapshot_component_used": [False] * 4,
        }
    )

    positions = build_topn_positions(
        factors, top_n=1, rebalance_dates=pd.DatetimeIndex([date])
    )

    assert positions.loc[positions["selected"], "ticker"].tolist() == ["A"]

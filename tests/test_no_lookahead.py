import pandas as pd

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

import pandas as pd
import pytest

from twse_factor_lab.validation.ohlcv_integrity import sort_ohlcv, validate_ohlcv


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["1101"],
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [11.0],
            "volume": [100.0],
        }
    )


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("close", 0.0, "close <= 0"),
        ("high", 8.0, "high < low"),
        ("high", 10.0, "high < close"),
        ("low", 12.0, "low > close"),
    ],
)
def test_validate_ohlcv_rejects_invalid_relationships(column, value, match):
    frame = _valid_frame()
    frame[column] = value

    with pytest.raises(ValueError, match=match):
        validate_ohlcv(frame)


def test_validate_ohlcv_rejects_duplicate_and_preserves_nan_rows():
    frame = pd.concat([_valid_frame(), _valid_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(frame)

    missing = _valid_frame()
    missing.loc[0, "close"] = pd.NA
    validate_ohlcv(missing)
    assert pd.isna(missing.loc[0, "close"])


def test_validate_ohlcv_accepts_only_floating_point_rounding_noise():
    frame = _valid_frame()
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] - 1e-12
    frame["low"] = frame["close"] + 1e-12

    validate_ohlcv(frame)


def test_sort_ohlcv_is_canonical():
    frame = pd.concat([_valid_frame().assign(ticker="2330"), _valid_frame()])

    assert sort_ohlcv(frame)["ticker"].tolist() == ["1101", "2330"]

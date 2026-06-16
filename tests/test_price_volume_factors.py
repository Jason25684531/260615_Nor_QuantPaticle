import pandas as pd

from twse_factor_lab.factors.price_volume import (
    build_price_volume_factor_frame,
    low_volatility_20d,
    momentum_60d,
    volume_ratio_5d_60d,
)


def test_momentum_60d_uses_backward_looking_close_prices():
    close_matrix = pd.DataFrame(
        {"1101": range(1, 66)}, index=pd.date_range("2024-01-01", periods=65, freq="D")
    )

    factor = momentum_60d(close_matrix)

    expected = close_matrix["1101"].iloc[-1] / close_matrix["1101"].iloc[-61] - 1
    assert round(float(factor["1101"].iloc[-1]), 6) == round(float(expected), 6)


def test_low_volatility_20d_prefers_atr_when_high_and_low_exist():
    index = pd.date_range("2024-01-01", periods=25, freq="D")
    close_matrix = pd.DataFrame({"1101": [100.0] * 25}, index=index)
    high_matrix = pd.DataFrame({"1101": [105.0] * 25}, index=index)
    low_matrix = pd.DataFrame({"1101": [95.0] * 25}, index=index)

    factor = low_volatility_20d(
        close_matrix=close_matrix,
        high_matrix=high_matrix,
        low_matrix=low_matrix,
    )

    assert float(factor["1101"].iloc[-1]) > 0.0


def test_low_volatility_20d_falls_back_to_return_volatility():
    index = pd.date_range("2024-01-01", periods=25, freq="D")
    close_matrix = pd.DataFrame({"1101": range(100, 125)}, index=index)

    factor = low_volatility_20d(close_matrix=close_matrix)

    assert pd.notna(factor["1101"].iloc[-1])


def test_volume_ratio_and_factor_frame_keep_date_ticker_shape():
    index = pd.date_range("2024-01-01", periods=65, freq="D")
    close_matrix = pd.DataFrame({"1101": range(1, 66)}, index=index)
    high_matrix = close_matrix + 1
    low_matrix = close_matrix - 1
    volume_matrix = pd.DataFrame({"1101": range(100, 165)}, index=index)

    ratio = volume_ratio_5d_60d(volume_matrix)
    factor_frame = build_price_volume_factor_frame(
        close_matrix=close_matrix,
        high_matrix=high_matrix,
        low_matrix=low_matrix,
        volume_matrix=volume_matrix,
    )

    assert ratio.shape == volume_matrix.shape
    assert {
        "date",
        "ticker",
        "momentum_60d",
        "low_volatility_20d",
        "volume_ratio_5d_60d",
    } <= set(factor_frame.columns)

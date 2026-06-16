import pandas as pd
import pytest

from twse_factor_lab.data.matrix_builder import build_field_matrix, build_ohlcv_matrices


def test_build_field_matrix_pivots_long_ohlcv_to_wide_matrix():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "ticker": ["1101", "1102", "1101"],
            "close": [10.0, 20.0, 11.0],
        }
    )

    matrix = build_field_matrix(frame, "close")

    assert list(matrix.columns) == ["1101", "1102"]
    assert matrix.index.name == "date"
    assert matrix.loc[pd.Timestamp("2024-01-02"), "1101"] == 10.0
    assert matrix.loc[pd.Timestamp("2024-01-02"), "1102"] == 20.0


def test_build_field_matrix_rejects_duplicate_date_and_ticker_rows():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker": ["1101", "1101"],
            "close": [10.0, 11.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_field_matrix(frame, "close")


def test_build_ohlcv_matrices_returns_all_expected_fields():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["1101", "1101"],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.0, 11.0],
            "volume": [100.0, 200.0],
        }
    )

    matrices = build_ohlcv_matrices(frame)

    assert set(matrices) == {"close", "high", "low", "volume"}
    assert pd.api.types.is_datetime64_any_dtype(matrices["close"].index)
    assert matrices["volume"].columns.tolist() == ["1101"]

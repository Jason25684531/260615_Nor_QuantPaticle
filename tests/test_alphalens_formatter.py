import pandas as pd

from twse_factor_lab.alphalens.formatter import (
    factor_matrix_to_series,
    validate_alphalens_inputs,
)


def test_factor_matrix_to_series_returns_multiindex_series():
    factor_matrix = pd.DataFrame(
        {"1101": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    series = factor_matrix_to_series(factor_matrix)

    assert isinstance(series.index, pd.MultiIndex)
    assert series.index.names == ["date", "asset"]


def test_validate_alphalens_inputs_reports_ready_when_dates_and_assets_overlap():
    factor_matrix = pd.DataFrame(
        {"1101": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    price_matrix = pd.DataFrame(
        {"1101": [10.0, 11.0, 12.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    report = validate_alphalens_inputs(factor_matrix, price_matrix)

    assert report["is_ready"] is True

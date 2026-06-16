import pandas as pd

from twse_factor_lab.factors.valuation import build_snapshot_valuation_factors


def test_snapshot_valuation_factors_handle_non_positive_values():
    valuation = pd.DataFrame(
        {
            "date": [pd.NaT, pd.NaT],
            "ticker": ["1101", "1102"],
            "pe": [10.0, 0.0],
            "pb": [1.0, -1.0],
            "dividend_yield": [3.5, 4.0],
        }
    )

    factors = build_snapshot_valuation_factors(
        valuation,
        as_of_date=pd.Timestamp("2026-06-16"),
    )

    assert factors["is_snapshot"].tolist() == [True, True]
    assert str(factors["as_of_date"].iloc[0].date()) == "2026-06-16"
    assert factors["pb_inverse"].iloc[0] == 1.0
    assert pd.isna(factors["pb_inverse"].iloc[1])
    assert pd.isna(factors["pe_inverse"].iloc[1])

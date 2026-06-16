import pandas as pd

from twse_factor_lab.factors.ranking import rank_factor


def test_rank_factor_supports_higher_and_lower_is_better():
    factor = pd.DataFrame(
        {"1101": [1.0], "1102": [2.0], "1103": [3.0]},
        index=[pd.Timestamp("2024-01-02")],
    )

    higher = rank_factor(factor, direction="higher_is_better")
    lower = rank_factor(factor, direction="lower_is_better")

    assert (
        higher.loc[pd.Timestamp("2024-01-02"), "1103"]
        > higher.loc[pd.Timestamp("2024-01-02"), "1101"]
    )
    assert (
        lower.loc[pd.Timestamp("2024-01-02"), "1101"]
        > lower.loc[pd.Timestamp("2024-01-02"), "1103"]
    )


def test_rank_factor_preserves_missing_values():
    factor = pd.DataFrame(
        {"1101": [1.0], "1102": [None]}, index=[pd.Timestamp("2024-01-02")]
    )

    ranked = rank_factor(factor, direction="higher_is_better")

    assert pd.isna(ranked.loc[pd.Timestamp("2024-01-02"), "1102"])

import pandas as pd
import pytest

from twse_factor_lab.factors.composer import build_d35_composite

DATES = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
TICKERS = ["A", "B", "C"]
_EMPTY_COMPOSITE_COLUMNS = [
    "date",
    "ticker",
    "composite_score",
    "composite_type",
    "is_snapshot_component_used",
]


def _price_volume_frame() -> pd.DataFrame:
    rows = [
        {"date": DATES[0], "ticker": "A", "f1": 1.0, "f2": 10.0},
        {"date": DATES[0], "ticker": "B", "f1": 2.0, "f2": float("nan")},
        {"date": DATES[0], "ticker": "C", "f1": 3.0, "f2": 30.0},
        {"date": DATES[1], "ticker": "A", "f1": 3.0, "f2": 5.0},
        {"date": DATES[1], "ticker": "B", "f1": 1.0, "f2": 15.0},
        {"date": DATES[1], "ticker": "C", "f1": 2.0, "f2": 25.0},
    ]
    return pd.DataFrame(rows)


def _research_universe_frame() -> pd.DataFrame:
    rows = [
        {"date": DATES[0], "ticker": "A", "is_eligible": True},
        {"date": DATES[0], "ticker": "B", "is_eligible": True},
        {"date": DATES[0], "ticker": "C", "is_eligible": False},
        {"date": DATES[1], "ticker": "A", "is_eligible": True},
        {"date": DATES[1], "ticker": "B", "is_eligible": True},
        {"date": DATES[1], "ticker": "C", "is_eligible": True},
    ]
    return pd.DataFrame(rows)


def _weights_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor": ["f1", "f2"],
            "weight": [0.5, 0.5],
            "selected": [True, True],
        }
    )


def _row(result: pd.DataFrame, date: pd.Timestamp, ticker: str) -> pd.Series:
    matches = result[(result["date"] == date) & (result["ticker"] == ticker)]
    return matches.iloc[0]


def _build(min_valid_factor_count: int = 2) -> pd.DataFrame:
    empty_composite_factors = pd.DataFrame(columns=_EMPTY_COMPOSITE_COLUMNS)
    return build_d35_composite(
        price_volume_factors=_price_volume_frame(),
        composite_factors=empty_composite_factors,
        research_universe=_research_universe_frame(),
        weights=_weights_frame(),
        directions={"f1": "higher_is_better", "f2": "higher_is_better"},
        min_valid_factor_count=min_valid_factor_count,
    )


def test_missing_factor_below_minimum_valid_count_is_nan_not_zero_filled():
    result = _build(min_valid_factor_count=2)
    row_b_day1 = _row(result, DATES[0], "B")
    assert row_b_day1["valid_factor_count"] == 1
    assert pd.isna(row_b_day1["composite_score"])


def test_ineligible_ticker_excluded_even_with_full_valid_factors():
    result = _build(min_valid_factor_count=2)
    row_c_day1 = _row(result, DATES[0], "C")
    assert row_c_day1["valid_factor_count"] == 2
    assert bool(row_c_day1["universe_eligible"]) is False
    assert pd.isna(row_c_day1["composite_score"])


def test_valid_ticker_gets_weighted_renormalized_rank_average():
    result = _build(min_valid_factor_count=2)
    row_a_day1 = _row(result, DATES[0], "A")
    # f1 rank among {A=1,B=2,C=3} -> A pct=1/3; f2 rank among {A=10,C=30} -> A pct=1/2
    expected = 0.5 * (1 / 3) + 0.5 * (1 / 2)
    assert row_a_day1["composite_score"] == pytest.approx(expected)


def test_lower_minimum_valid_count_still_renormalizes_single_factor():
    result = _build(min_valid_factor_count=1)
    row_b_day1 = _row(result, DATES[0], "B")
    # only f1 valid for B: rank pct = 2/3, renormalized over weight_sum=0.5.
    assert row_b_day1["valid_factor_count"] == 1
    assert row_b_day1["composite_score"] == pytest.approx(2 / 3)


def test_deterministic_rerun_produces_identical_output():
    first = _build()
    second = _build()
    pd.testing.assert_frame_equal(first, second)


def test_missing_factor_selection_raises_when_nothing_selected():
    weights = _weights_frame()
    weights["selected"] = False
    with pytest.raises(ValueError, match="No factors selected"):
        build_d35_composite(
            price_volume_factors=_price_volume_frame(),
            composite_factors=pd.DataFrame(columns=_EMPTY_COMPOSITE_COLUMNS),
            research_universe=_research_universe_frame(),
            weights=weights,
            directions={"f1": "higher_is_better", "f2": "higher_is_better"},
            min_valid_factor_count=1,
        )

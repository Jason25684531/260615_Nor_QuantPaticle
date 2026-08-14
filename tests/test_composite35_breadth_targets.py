import pandas as pd
import pytest

from twse_factor_lab.portfolio.breadth import (
    breadth_to_exposure,
    compute_market_breadth,
)
from twse_factor_lab.portfolio.weights import (
    apply_gross_exposure,
    build_equal_weight_portfolio,
)

DATES = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def _matrix(values: dict) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES)


def test_breadth_excludes_unusable_ma_and_ineligible_from_both_sides():
    close = _matrix({"A": [10.0, 15.0], "B": [20.0, 18.0], "C": [30.0, 25.0]})
    ma = _matrix({"A": [8.0, 20.0], "B": [25.0, 10.0], "C": [float("nan"), 20.0]})
    eligible = _matrix({"A": [True, True], "B": [True, False], "C": [True, True]})

    breadth = compute_market_breadth(
        close_matrix=close,
        ma_matrix=ma,
        eligible_matrix=eligible,
        universe_status="PARTIAL",
    )

    day1 = breadth.iloc[0]
    # C's MA is NaN on day1 -> excluded from both numerator and denominator.
    assert day1["eligible_usable_count"] == 2
    assert day1["above_ma60_count"] == 1  # only A (10>8) above MA
    assert day1["breadth"] == pytest.approx(0.5)

    day2 = breadth.iloc[1]
    # B is ineligible on day2 -> excluded from both sides.
    assert day2["eligible_usable_count"] == 2
    assert day2["above_ma60_count"] == 1  # only C (25>20) above MA
    assert day2["breadth"] == pytest.approx(0.5)
    assert (breadth["universe_status"] == "PARTIAL").all()


def test_breadth_to_exposure_step_rule_and_boundary():
    breadth = pd.DataFrame({"date": DATES, "breadth": [0.55, 0.40]})
    result = breadth_to_exposure(
        breadth, threshold=0.40, exposure_high=1.0, exposure_low=0.5
    )
    assert result.loc[0, "gross_exposure"] == 1.0
    # exactly at threshold: rule is strictly greater-than -> low exposure.
    assert result.loc[1, "gross_exposure"] == 0.5


def test_breadth_to_exposure_carries_forward_and_defaults_conservative():
    extra_date = pd.Timestamp("2024-01-04")
    breadth = pd.DataFrame(
        {
            "date": [*DATES, extra_date],
            "breadth": [float("nan"), 0.60, float("nan")],
        }
    )
    result = breadth_to_exposure(
        breadth, threshold=0.40, exposure_high=1.0, exposure_low=0.5
    )
    # no prior valid value -> conservative default.
    assert result.loc[0, "gross_exposure"] == 0.5
    assert result.loc[1, "gross_exposure"] == 1.0
    assert result.loc[2, "gross_exposure"] == 1.0  # carried forward from day2


def test_apply_gross_exposure_scales_weights_and_applies_at_t_plus_one():
    positions = pd.DataFrame(
        {
            "date": [DATES[0], DATES[0]],
            "ticker": ["A", "B"],
            "factor_score": [0.9, 0.8],
            "rank": [1, 2],
            "selected": [True, True],
            "top_n": [2, 2],
        }
    )
    calendar = pd.DataFrame(
        {
            "signal_date": [DATES[0]],
            "execution_date": [DATES[1]],
            "execution_lag_days": [1],
        }
    )
    weights = build_equal_weight_portfolio(positions, rebalance_calendar=calendar)
    market_breadth = pd.DataFrame({"date": [DATES[0]], "gross_exposure": [0.5]})

    scaled = apply_gross_exposure(
        weights, market_breadth=market_breadth, rebalance_calendar=calendar
    )
    assert scaled["target_weight"].sum() == pytest.approx(0.5)
    # breadth observed at signal date T (DATES[0]) is applied at execution T+1.
    assert (scaled["execution_date"] == DATES[1]).all()


def test_apply_gross_exposure_defaults_to_full_when_breadth_missing_for_date():
    positions = pd.DataFrame(
        {
            "date": [DATES[0]],
            "ticker": ["A"],
            "factor_score": [0.9],
            "rank": [1],
            "selected": [True],
            "top_n": [1],
        }
    )
    calendar = pd.DataFrame(
        {
            "signal_date": [DATES[0]],
            "execution_date": [DATES[1]],
            "execution_lag_days": [1],
        }
    )
    weights = build_equal_weight_portfolio(positions, rebalance_calendar=calendar)
    empty_breadth = pd.DataFrame(columns=["date", "gross_exposure"])
    scaled = apply_gross_exposure(
        weights, market_breadth=empty_breadth, rebalance_calendar=calendar
    )
    assert scaled["target_weight"].sum() == pytest.approx(1.0)

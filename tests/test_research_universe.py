import pandas as pd
import pytest

from twse_factor_lab.data.universe import (
    build_liquidity_eligibility,
    build_listing_eligibility,
    build_research_universe,
    build_universe_coverage,
)


def _universe():
    return pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "listed_date": pd.to_datetime(["2024-01-02", pd.NaT]),
        }
    )


def _ohlcv(values=(90.0, 100.0, 110.0)):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(values)),
            "ticker": ["A"] * len(values),
            "close": values,
            "volume": [1.0] * len(values),
        }
    )


def test_listing_eligibility_uses_listing_date_and_allows_missing_date():
    result = build_listing_eligibility(
        _universe(), pd.date_range("2024-01-01", periods=3)
    )

    assert result.query("ticker == 'A'")["listing_eligible"].tolist() == [
        False,
        True,
        True,
    ]
    assert result.query("ticker == 'B'")["listing_eligible"].all()


def test_liquidity_uses_prior_window_and_strict_threshold():
    result = build_liquidity_eligibility(_ohlcv(), window=2, minimum_traded_value=100.0)

    assert result["liquidity_eligible"].tolist() == [False, False, False]
    higher = build_liquidity_eligibility(
        _ohlcv((110.0, 120.0, 130.0)), window=2, minimum_traded_value=100.0
    )
    assert higher["liquidity_eligible"].tolist() == [False, False, True]


def test_research_universe_keeps_unfetched_tickers_and_config_changes_liquidity():
    dates = pd.date_range("2024-01-01", periods=3)
    result = build_research_universe(
        _universe(),
        _ohlcv((110.0, 120.0, 130.0)),
        dates=dates,
        liquidity={"window": 2, "minimum_traded_value": 100.0},
    )

    assert not result.query("ticker == 'B'")["has_ohlcv"].any()
    assert result.query("ticker == 'A'")["is_eligible"].tolist() == [False, False, True]

    disabled = build_research_universe(
        _universe(), _ohlcv(), dates=dates, liquidity={"enabled": False}
    )
    assert disabled.query("ticker == 'A'")["liquidity_eligible"].all()


def test_coverage_uses_listing_eligible_denominator_and_rejects_inconsistency():
    research = build_research_universe(
        _universe(),
        _ohlcv((110.0, 120.0, 130.0)),
        dates=pd.date_range("2024-01-01", periods=3),
        liquidity={"window": 2, "minimum_traded_value": 100.0},
    )
    coverage = build_universe_coverage(research)

    assert coverage.loc[2, "listing_eligible_count"] == 2
    assert coverage.loc[2, "ohlcv_available_count"] == 1
    assert coverage.loc[2, "coverage_ratio"] == 0.5

    research.loc[:, "is_eligible"] = True
    with pytest.raises(ValueError, match="inconsistent"):
        build_universe_coverage(research)

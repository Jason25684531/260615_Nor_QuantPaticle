"""Phase 2 tests: liquid PIT-safe universe v2 and coverage diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twse_factor_lab.data.universe import (
    LIQUIDITY_MEASURE_SOURCE,
    build_liquid_pit_coverage,
    build_liquid_pit_universe,
)


def _ohlcv(dates: pd.DatetimeIndex, tickers: dict[str, float]) -> pd.DataFrame:
    rows = []
    for ticker, price in tickers.items():
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "close": price + offset,
                    "volume": 10_000_000,  # close*volume >> 50M threshold
                }
            )
    return pd.DataFrame(rows)


def _fundamentals(dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        for metric in ("eps", "roe"):
            for date in dates:
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "metric": metric,
                        "value": 1.0,
                        "available_date": date,
                    }
                )
    return pd.DataFrame(rows)


def _universe(tickers: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": ticker, "listed_date": listed}
            for ticker, listed in tickers.items()
        ]
    )


def test_not_yet_listed_and_missing_fundamentals_excluded() -> None:
    dates = pd.bdate_range("2020-01-01", periods=40)
    universe = _universe({"AAA": "2019-01-01", "BBB": "2020-01-20"})
    ohlcv = _ohlcv(dates, {"AAA": 100.0, "BBB": 100.0})
    # Only AAA has fundamentals; BBB also lists mid-window.
    fundamentals = _fundamentals(dates, ["AAA"])

    result = build_liquid_pit_universe(
        universe, ohlcv, fundamentals, dates=dates, window=20
    )

    bbb_early = result[(result["ticker"] == "BBB") & (result["date"] < "2020-01-20")]
    assert (bbb_early["listed_status"] == "not_yet_listed").all()
    assert not bbb_early["universe_included"].any()

    # BBB never has fundamentals -> never universe_included even once listed.
    assert not result[result["ticker"] == "BBB"]["universe_included"].any()

    # AAA becomes included only after 20-day liquidity warmup completes.
    aaa = result[result["ticker"] == "AAA"].sort_values("date")
    assert not aaa["universe_included"].iloc[:20].any()
    assert aaa["universe_included"].iloc[20:].all()
    assert result.attrs["liquidity_measure_source"] == LIQUIDITY_MEASURE_SOURCE


def test_future_price_mutation_does_not_change_past_membership() -> None:
    dates = pd.bdate_range("2020-01-01", periods=40)
    universe = _universe({"AAA": "2019-01-01"})
    ohlcv = _ohlcv(dates, {"AAA": 100.0})
    fundamentals = _fundamentals(dates, ["AAA"])

    baseline = build_liquid_pit_universe(
        universe, ohlcv, fundamentals, dates=dates, window=20
    )

    mutated = ohlcv.copy()
    future = mutated["date"] >= dates[30]
    mutated.loc[future, "volume"] = 0  # crush future liquidity

    after = build_liquid_pit_universe(
        universe, mutated, fundamentals, dates=dates, window=20
    )

    past = dates[:30]
    base_past = baseline[baseline["date"].isin(past)].reset_index(drop=True)
    after_past = after[after["date"].isin(past)].reset_index(drop=True)
    pd.testing.assert_series_equal(
        base_past["universe_included"], after_past["universe_included"]
    )


def test_coverage_is_deterministic_and_reports_counts() -> None:
    dates = pd.bdate_range("2020-01-01", periods=40)
    universe = _universe({"AAA": "2019-01-01", "BBB": "2019-01-01"})
    ohlcv = _ohlcv(dates, {"AAA": 100.0, "BBB": 100.0})
    fundamentals = _fundamentals(dates, ["AAA", "BBB"])

    v2 = build_liquid_pit_universe(
        universe, ohlcv, fundamentals, dates=dates, window=20
    )
    per_date_a, summary_a = build_liquid_pit_coverage(v2)
    per_date_b, summary_b = build_liquid_pit_coverage(v2)

    pd.testing.assert_frame_equal(per_date_a, per_date_b)
    assert summary_a == summary_b
    # After warmup both names rank; joint == eps == roe count.
    last = per_date_a[per_date_a["date"] == dates[-1]].iloc[0]
    assert last["universe_count"] == 2
    assert last["eps_valid_count"] == 2
    assert last["roe_valid_count"] == 2
    assert last["fundamental_joint_count"] == 2
    assert np.isclose(last["coverage_ratio"], 1.0)


def test_liquidity_uses_only_past_observations() -> None:
    dates = pd.bdate_range("2020-01-01", periods=25)
    universe = _universe({"AAA": "2019-01-01"})
    ohlcv = _ohlcv(dates, {"AAA": 100.0})
    fundamentals = _fundamentals(dates, ["AAA"])

    result = build_liquid_pit_universe(
        universe, ohlcv, fundamentals, dates=dates, window=20
    )
    aaa = result[result["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    # First 20 rows have no complete trailing window (shifted) -> NaN median.
    assert aaa["median_traded_value_20d"].iloc[:20].isna().all()
    assert aaa["median_traded_value_20d"].iloc[20:].notna().all()

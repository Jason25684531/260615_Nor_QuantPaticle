"""Phase 6 tests: MA60 breadth overlay wiring for the locked candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twse_factor_lab.portfolio.breadth import (
    breadth_to_exposure,
    compute_market_breadth,
)
from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar
from twse_factor_lab.portfolio.weights import apply_gross_exposure

THRESHOLD, HIGH, LOW = 0.40, 1.0, 0.5


def _matrices():
    dates = pd.bdate_range("2020-01-01", periods=6)
    tickers = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(1.0, index=dates, columns=tickers)
    ma = pd.DataFrame(1.0, index=dates, columns=tickers)
    eligible = pd.DataFrame(True, index=dates, columns=tickers)
    return dates, tickers, close, ma, eligible


def test_exposure_rule_high_and_low_around_040() -> None:
    dates, tickers, close, ma, eligible = _matrices()
    # Day 0: 3/5 = 0.6 > 0.40 -> HIGH. Day 1: 2/5 = 0.40 (not > 0.40) -> LOW.
    close.iloc[0] = [2.0, 2.0, 2.0, 0.5, 0.5]  # 3 above MA
    close.iloc[1] = [2.0, 2.0, 0.5, 0.5, 0.5]  # 2 above MA

    breadth = compute_market_breadth(
        close_matrix=close, ma_matrix=ma, eligible_matrix=eligible,
        universe_status="PARTIAL",
    )
    exposed = breadth_to_exposure(
        breadth, threshold=THRESHOLD, exposure_high=HIGH, exposure_low=LOW
    )
    assert np.isclose(exposed.loc[0, "breadth"], 0.6)
    assert exposed.loc[0, "gross_exposure"] == HIGH
    assert np.isclose(exposed.loc[1, "breadth"], 0.4)
    assert exposed.loc[1, "gross_exposure"] == LOW


def test_ma60_uses_only_contemporaneous_prices() -> None:
    dates, tickers, close, ma, eligible = _matrices()
    close.iloc[0] = [2.0, 2.0, 2.0, 0.5, 0.5]
    base = compute_market_breadth(
        close_matrix=close, ma_matrix=ma, eligible_matrix=eligible,
        universe_status="PARTIAL",
    )
    mutated = close.copy()
    mutated.iloc[3:] = 99.0  # crush the future
    after = compute_market_breadth(
        close_matrix=mutated, ma_matrix=ma, eligible_matrix=eligible,
        universe_status="PARTIAL",
    )
    assert np.isclose(base.loc[0, "breadth"], after.loc[0, "breadth"])


def test_overlay_changes_exposure_only_not_selection() -> None:
    dates, tickers, close, ma, eligible = _matrices()
    calendar = build_rebalance_calendar(dates, frequency="daily", execution_lag_days=1)
    weights = pd.DataFrame(
        {
            "date": [dates[0], dates[0]],
            "ticker": ["A", "B"],
            "target_weight": [0.5, 0.5],
            "execution_date": [dates[1], dates[1]],
            "execution_lag_days": [1, 1],
        }
    )
    # Force a risk-off day-0 signal -> 0.5 exposure at execution day-1.
    close.iloc[0] = [2.0, 0.5, 0.5, 0.5, 0.5]  # 1/5 = 0.2 <= 0.40 -> LOW
    breadth = compute_market_breadth(
        close_matrix=close, ma_matrix=ma, eligible_matrix=eligible,
        universe_status="PARTIAL",
    )
    exposed = breadth_to_exposure(
        breadth, threshold=THRESHOLD, exposure_high=HIGH, exposure_low=LOW
    )
    on = apply_gross_exposure(
        weights, market_breadth=exposed, rebalance_calendar=calendar
    )

    # Same tickers selected, weights scaled by 0.5, ranking untouched.
    assert set(on["ticker"]) == set(weights["ticker"])
    assert np.allclose(sorted(on["target_weight"]), [0.25, 0.25])

"""A rebalance replaces the whole target portfolio, including removals."""

import pandas as pd

from twse_factor_lab.backtest.vectorbt_engine import _weights_matrix


def _matrix() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=5)
    weights = pd.DataFrame(
        {
            "execution_date": [days[1], days[1], days[3], days[3]],
            "ticker": ["A", "B", "A", "C"],
            "target_weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    return _weights_matrix(weights, days, pd.Index(["A", "B", "C"]))


def test_removed_ticker_becomes_zero_on_rebalance():
    assert _matrix().iloc[3]["B"] == 0.0


def test_new_ticker_receives_target_weight_on_rebalance():
    assert _matrix().iloc[3]["C"] == 0.5


def test_no_rebalance_does_not_liquidate_portfolio():
    assert _matrix().iloc[2]["B"] == 0.5

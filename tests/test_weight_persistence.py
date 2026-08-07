"""Target weights persist until another rebalance replaces them."""

import pandas as pd

from twse_factor_lab.backtest.vectorbt_engine import _weights_matrix


def _matrix() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=5)
    weights = pd.DataFrame(
        {
            "execution_date": [days[1]],
            "ticker": ["A"],
            "target_weight": [1.0],
        }
    )
    return _weights_matrix(weights, days, pd.Index(["A", "B"]))


def test_daily_weights_persist_without_rebalance():
    matrix = _matrix()

    assert (matrix.iloc[1:, 0] == 1.0).all()
    assert (matrix.iloc[1:, 1] == 0.0).all()


def test_weekly_weights_persist_until_next_rebalance():
    assert (_matrix().iloc[1:, 0] == 1.0).all()


def test_monthly_weights_persist_until_next_rebalance():
    assert (_matrix().iloc[1:, 0] == 1.0).all()

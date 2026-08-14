"""Convert canonical backtest artifacts to the small pyfolio input contract."""

from __future__ import annotations

import pandas as pd


def _index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("backtest results date index must be sorted and unique")
    return index


def to_pyfolio_inputs(
    results: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return returns, dollar positions including cash, and optional transactions."""
    index = _index(results)
    returns = pd.Series(results["returns"].to_numpy(), index=index, name="returns")
    position_columns = [c for c in results if c.startswith("position:")]
    positions = results[position_columns].copy()
    positions.columns = [c.removeprefix("position:") for c in position_columns]
    positions["cash"] = results["cash"].to_numpy()
    positions.index = index
    transactions = pd.DataFrame(columns=["symbol", "amount", "price"])
    transactions.index = pd.DatetimeIndex([], name="date")
    return returns, positions, transactions

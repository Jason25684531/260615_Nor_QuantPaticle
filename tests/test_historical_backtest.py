from __future__ import annotations

import pandas as pd


def test_canonical_backtest_has_returns_positions_rebalances_and_transactions() -> None:
    root = "data/research/fundamental-production-final-v1/historical-evidence/"
    returns = pd.read_parquet(root + "canonical_backtest_returns.parquet")
    positions = pd.read_parquet(root + "canonical_backtest_positions.parquet")
    rebalances = pd.read_parquet(root + "canonical_backtest_rebalances.parquet")
    transactions = pd.read_parquet(root + "canonical_backtest_transactions.parquet")
    assert len(returns) > 750
    assert returns["date"].is_unique
    assert set(
        ["date", "ticker", "market_value", "cash", "portfolio_value", "weight"]
    ).issubset(positions.columns)
    assert len(rebalances) > 0
    assert len(transactions) > 0
    assert rebalances["strategy_fingerprint"].nunique() == 1


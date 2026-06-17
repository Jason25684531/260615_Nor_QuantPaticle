"""Tests for Top N sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.scenarios import run_topn_sensitivity


def _make_close(n_days: int = 100, n_tickers: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    data = {}
    for i in range(n_tickers):
        ticker = f"T{i:03d}.TW"
        prices = (
            100.0
            * (1 + np.random.default_rng(i + 99).normal(0, 0.01, n_days)).cumprod()
        )
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_factors(close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in close.index[::5]:
        for ticker in close.columns:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "composite_score": float(hash(ticker + str(date)) % 100),
                    "composite_type": "historical_price_volume",
                    "is_snapshot_component_used": False,
                }
            )
    return pd.DataFrame(rows)


def _make_config(top_n_grid: list[int] | None = None) -> dict:
    return {
        "backtest": {
            "factor_name": "historical_price_volume",
            "top_n": 20,
            "top_n_grid": top_n_grid or [10, 20, 30],
            "rebalance_frequency": "weekly",
            "execution_lag_days": 1,
            "initial_cash": 1_000_000,
            "fees": {
                "buy_fee_rate": 0.001425,
                "sell_fee_rate": 0.001425,
                "transaction_tax_rate": 0.003,
                "slippage_rate": 0.001,
            },
            "rules": {"hold_until_drop": False, "drop_rank_buffer": 30},
        }
    }


class TestRunTopNSensitivity:
    def test_one_row_per_top_n(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config([10, 20, 30])
        df = run_topn_sensitivity(close, factors, config)
        assert len(df) == 3
        assert set(df["top_n"]) == {10, 20, 30}

    def test_required_columns_present(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config([10, 20])
        df = run_topn_sensitivity(close, factors, config)
        required = [
            "top_n",
            "factor_name",
            "rebalance_frequency",
            "buffer_enabled",
            "total_return",
            "annualized_return",
            "sharpe",
            "max_drawdown",
            "turnover",
            "avg_exposure",
            "engine",
            "notes",
        ]
        for col in required:
            assert col in df.columns, f"Missing: {col}"

    def test_small_universe_notes(self) -> None:
        close = _make_close(n_tickers=5)
        factors = _make_factors(close)
        config = _make_config([10])  # top_n > universe
        df = run_topn_sensitivity(close, factors, config)
        assert len(df) == 1
        # notes should mention universe size or be empty string (handled gracefully)

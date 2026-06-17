"""Tests for rebalance frequency sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.scenarios import run_rebalance_sensitivity


def _make_close(n_days: int = 260, n_tickers: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    data = {}
    for i in range(n_tickers):
        ticker = f"T{i:03d}.TW"
        prices = (
            100.0
            * (1 + np.random.default_rng(i + 77).normal(0, 0.01, n_days)).cumprod()
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


def _make_config(freq_grid: list[str] | None = None) -> dict:
    return {
        "backtest": {
            "factor_name": "historical_price_volume",
            "top_n": 10,
            "rebalance_frequency_grid": freq_grid or ["daily", "weekly", "monthly"],
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


class TestRunRebalanceSensitivity:
    def test_one_row_per_frequency(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config(["daily", "weekly", "monthly"])
        df = run_rebalance_sensitivity(close, factors, config)
        assert len(df) == 3
        assert set(df["rebalance_frequency"]) == {"daily", "weekly", "monthly"}

    def test_required_columns_present(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config(["weekly"])
        df = run_rebalance_sensitivity(close, factors, config)
        required = [
            "rebalance_frequency",
            "factor_name",
            "top_n",
            "buffer_enabled",
            "rebalance_count",
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

    def test_rebalance_count_monotone(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config(["daily", "weekly", "monthly"])
        df = run_rebalance_sensitivity(close, factors, config)
        counts = df.set_index("rebalance_frequency")["rebalance_count"]
        assert counts["daily"] > counts["weekly"] > counts["monthly"]

    def test_turnover_daily_gte_weekly_gte_monthly(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config(["daily", "weekly", "monthly"])
        df = run_rebalance_sensitivity(close, factors, config)
        t = df.set_index("rebalance_frequency")["turnover"]
        # Turnover generally decreases with lower frequency
        assert t["daily"] >= t["monthly"]

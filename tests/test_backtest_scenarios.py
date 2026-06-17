"""Tests for cost sensitivity scenario runner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.scenarios import run_cost_scenarios


def _make_close(n_days: int = 100, n_tickers: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    data = {}
    for i in range(n_tickers):
        ticker = f"T{i:03d}.TW"
        prices = (
            100.0 * (1 + np.random.default_rng(i).normal(0, 0.01, n_days)).cumprod()
        )
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_factors(
    close: pd.DataFrame, factor_name: str = "historical_price_volume"
) -> pd.DataFrame:
    rows = []
    for date in close.index[::5]:  # every 5 days for speed
        for ticker in close.columns:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "composite_score": float(hash(ticker + str(date)) % 100),
                    "composite_type": factor_name,
                    "is_snapshot_component_used": False,
                }
            )
    return pd.DataFrame(rows)


def _make_config(scenarios: list[dict] | None = None) -> dict:
    base = {
        "backtest": {
            "factor_name": "historical_price_volume",
            "top_n": 10,
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
            "cost_sensitivity": {
                "enabled": True,
                "scenarios": scenarios
                or [
                    {
                        "name": "no_cost",
                        "buy_fee_rate": 0.0,
                        "sell_fee_rate": 0.0,
                        "transaction_tax_rate": 0.0,
                        "slippage_rate": 0.0,
                    },
                    {
                        "name": "base_cost",
                        "buy_fee_rate": 0.001425,
                        "sell_fee_rate": 0.001425,
                        "transaction_tax_rate": 0.003,
                        "slippage_rate": 0.001,
                    },
                ],
            },
        }
    }
    return base


class TestRunCostScenarios:
    def test_one_row_per_scenario(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config()
        df = run_cost_scenarios(close, factors, config)
        assert len(df) == 2
        assert set(df["scenario"]) == {"no_cost", "base_cost"}

    def test_no_cost_return_gte_base_cost(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config()
        df = run_cost_scenarios(close, factors, config)
        no_cost = float(df[df["scenario"] == "no_cost"]["total_return"].iloc[0])
        base = float(df[df["scenario"] == "base_cost"]["total_return"].iloc[0])
        assert no_cost >= base

    def test_cost_drag_computed(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config()
        df = run_cost_scenarios(close, factors, config)
        no_cost_return = float(df[df["scenario"] == "no_cost"]["total_return"].iloc[0])
        base_return = float(df[df["scenario"] == "base_cost"]["total_return"].iloc[0])
        base_drag = float(df[df["scenario"] == "base_cost"]["cost_drag"].iloc[0])
        assert abs(base_drag - (no_cost_return - base_return)) < 1e-6

    def test_snapshot_factor_returns_empty(self) -> None:
        close = _make_close()
        factors = _make_factors(close, factor_name="pb_inverse")
        config = _make_config()
        config["backtest"]["factor_name"] = "pb_inverse"
        df = run_cost_scenarios(close, factors, config)
        assert df.empty

    def test_required_columns_present(self) -> None:
        close = _make_close()
        factors = _make_factors(close)
        config = _make_config()
        df = run_cost_scenarios(close, factors, config)
        required = [
            "scenario",
            "factor_name",
            "top_n",
            "rebalance_frequency",
            "buffer_enabled",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "turnover",
            "avg_exposure",
            "cost_drag",
            "engine",
            "start_date",
            "end_date",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

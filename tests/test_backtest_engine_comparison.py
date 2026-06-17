"""Tests for engine cross-check (vectorbt vs fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.diagnostics import run_engine_comparison


def _make_close(n_days: int = 100, n_tickers: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    data = {}
    for i in range(n_tickers):
        ticker = f"T{i:03d}.TW"
        prices = (
            100.0
            * (1 + np.random.default_rng(i + 42).normal(0, 0.01, n_days)).cumprod()
        )
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_weights(close: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    tickers = list(close.columns[:top_n])
    rows = []
    for date in close.index[1::5]:
        for ticker in tickers:
            rows.append(
                {"execution_date": date, "ticker": ticker, "target_weight": 1.0 / top_n}
            )
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "backtest": {
            "top_n": 5,
            "initial_cash": 1_000_000,
            "fees": {
                "buy_fee_rate": 0.001425,
                "sell_fee_rate": 0.001425,
                "transaction_tax_rate": 0.003,
                "slippage_rate": 0.001,
            },
        }
    }


class TestRunEngineComparison:
    def test_fallback_always_present(self) -> None:
        close = _make_close()
        weights = _make_weights(close)
        df = run_engine_comparison(close, weights, _config())
        fb = df[df["engine"] == "fallback_weight_engine"]
        assert len(fb) == 1
        assert not np.isnan(float(fb.iloc[0]["total_return"]))

    def test_vectorbt_row_always_present(self) -> None:
        close = _make_close()
        weights = _make_weights(close)
        df = run_engine_comparison(close, weights, _config())
        vbt = df[df["engine"] == "vectorbt"]
        assert len(vbt) == 1

    def test_vectorbt_unavailable_does_not_raise(self) -> None:
        """vectorbt may or may not be installed; either way, no exception."""

        close = _make_close()
        weights = _make_weights(close)
        config = _config()
        df = run_engine_comparison(close, weights, config)
        vbt = df[df["engine"] == "vectorbt"].iloc[0]
        notes = str(vbt["notes"]).lower()
        # Either available (numeric result) or unavailable (NaN + note)
        if "unavailable" in notes:
            assert np.isnan(float(vbt["total_return"]))
        else:
            assert not np.isnan(float(vbt["total_return"]))

    def test_required_columns_present(self) -> None:
        close = _make_close()
        weights = _make_weights(close)
        df = run_engine_comparison(close, weights, _config())
        for col in [
            "engine",
            "total_return",
            "sharpe",
            "max_drawdown",
            "turnover",
            "notes",
        ]:
            assert col in df.columns

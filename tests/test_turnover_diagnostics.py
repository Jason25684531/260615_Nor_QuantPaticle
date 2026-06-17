"""Tests for turnover diagnostics computation."""

from __future__ import annotations

import pandas as pd

from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.diagnostics import compute_turnover_diagnostics


def _make_weights(
    n_dates: int = 50, n_tickers: int = 10, top_n: int = 5
) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = []
    for date in dates:
        selected = [f"T{j:02d}" for j in range(top_n)]
        for ticker in selected:
            rows.append(
                {
                    "execution_date": date,
                    "ticker": ticker,
                    "target_weight": 1.0 / top_n,
                }
            )
    return pd.DataFrame(rows)


class TestComputeTurnoverDiagnostics:
    def test_required_metrics_present(self) -> None:
        weights = _make_weights()
        df = compute_turnover_diagnostics(weights)
        metrics = set(df["metric"])
        required = {
            "average_daily_turnover",
            "median_daily_turnover",
            "max_daily_turnover",
            "annualized_turnover_estimate",
            "avg_holdings",
            "avg_buys_per_rebalance",
            "avg_sells_per_rebalance",
            "estimated_cost_drag",
        }
        assert required.issubset(metrics)

    def test_annualized_is_252x_avg(self) -> None:
        weights = _make_weights()
        df = compute_turnover_diagnostics(weights)
        avg = float(df[df["metric"] == "average_daily_turnover"]["value"].iloc[0])
        ann = float(df[df["metric"] == "annualized_turnover_estimate"]["value"].iloc[0])
        assert abs(ann - avg * 252) < 1e-6

    def test_values_non_negative(self) -> None:
        weights = _make_weights()
        df = compute_turnover_diagnostics(weights)
        assert (df["value"] >= 0).all()

    def test_empty_weights_returns_empty(self) -> None:
        empty = pd.DataFrame(columns=["execution_date", "ticker", "target_weight"])
        df = compute_turnover_diagnostics(empty)
        assert df.empty

    def test_with_cost_model(self) -> None:
        weights = _make_weights()
        cost_model = CostModel(
            buy_fee_rate=0.0,
            sell_fee_rate=0.0,
            transaction_tax_rate=0.0,
            slippage_rate=0.0,
        )
        df = compute_turnover_diagnostics(weights, cost_model=cost_model)
        drag = float(df[df["metric"] == "estimated_cost_drag"]["value"].iloc[0])
        assert drag == 0.0  # zero cost → zero drag

    def test_avg_holdings_matches_top_n(self) -> None:
        top_n = 5
        weights = _make_weights(top_n=top_n)
        df = compute_turnover_diagnostics(weights)
        avg_h = float(df[df["metric"] == "avg_holdings"]["value"].iloc[0])
        assert abs(avg_h - top_n) < 1e-6

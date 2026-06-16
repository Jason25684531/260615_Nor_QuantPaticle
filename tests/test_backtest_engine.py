import pandas as pd

from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest


def test_fallback_backtest_engine_produces_equity_curve_and_metrics():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    close_matrix = pd.DataFrame(
        {"0001": [100.0, 101.0, 102.0, 103.0], "0002": [50.0, 49.0, 51.0, 52.0]},
        index=dates,
    )
    weights = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["0001", "0002", "0001", "0002"],
            "target_weight": [0.5, 0.5, 1.0, 0.0],
            "execution_date": [dates[1], dates[1], dates[2], dates[2]],
            "execution_lag_days": [1, 1, 1, 1],
        }
    )

    results, metrics = run_weight_backtest(
        close_matrix=close_matrix,
        portfolio_weights=weights,
        cost_model=CostModel(),
        initial_cash=1_000_000,
        top_n=2,
        use_vectorbt=False,
        allow_fallback=True,
    )

    assert {"date", "equity", "returns", "drawdown"}.issubset(results.columns)
    assert "total_return" in metrics.columns
    assert "engine" in metrics.columns
    assert metrics.loc[0, "engine"] == "fallback_weight_engine"

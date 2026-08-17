import numpy as np
import pandas as pd
import pytest

from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest


@pytest.mark.parametrize(
    "prices, targets",
    [
        ([[100, 100], [101, 99], [102, 98]], [[0.5, 0.5], [1, 0]]),
        ([[100, 100], [np.nan, 101], [102, 102]], [[0.5, 0.5], [0, 1]]),
        ([[100, 100], [np.nan, 101], [np.nan, 102], [103, 103]], [[0.5, 0.5], [0, 1]]),
        ([[100, 100], [np.nan, 101], [np.nan, 102]], [[0.5, 0.5], [0, 1]]),
        ([[100, 100], [101, 101], [102, 102]], [[1, 0], [0, 1]]),
        ([[100, 100], [101, 99], [102, 98]], [[1, 0], [0.5, 0], [1, 0]]),
    ],
)
def test_economic_parity_fixtures(prices, targets):
    """Continuous, missing, removal, and breadth-exposure fixtures agree."""
    dates = pd.bdate_range("2024-01-01", periods=len(prices))
    close = pd.DataFrame(prices, index=dates, columns=["A", "B"])
    rows = []
    for index, weights in enumerate(targets):
        date = dates[min(index, len(dates) - 1)]
        rows.extend(
            {"execution_date": date, "ticker": ticker, "target_weight": weight}
            for ticker, weight in zip(["A", "B"], weights, strict=True)
        )
    targets_frame = pd.DataFrame(rows)
    kwargs = dict(
        close_matrix=close,
        portfolio_weights=targets_frame,
        cost_model=CostModel(0.0, 0.0, 0.0, 0.0),
        initial_cash=1_000.0,
        top_n=2,
        allow_fallback=False,
    )
    custom, custom_metrics = run_weight_backtest(**kwargs, use_vectorbt=False)
    vectorbt, vectorbt_metrics = run_weight_backtest(**kwargs, use_vectorbt=True)
    columns = [
        "equity", "returns", "cash", "turnover", "exposure", "position:A", "position:B"
    ]
    np.testing.assert_allclose(custom[columns], vectorbt[columns], rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        custom_metrics[["total_return", "sharpe", "max_drawdown"]],
        vectorbt_metrics[["total_return", "sharpe", "max_drawdown"]],
        rtol=0,
        atol=1e-8,
    )

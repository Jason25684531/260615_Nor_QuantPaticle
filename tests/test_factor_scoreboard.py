import pandas as pd

from twse_factor_lab.selection.scoreboard import (
    build_factor_scoreboard,
    select_backtest_factor,
)


def test_scoreboard_excludes_snapshot_factors_and_marks_default_candidate():
    ic_summary = pd.DataFrame(
        {
            "factor": ["historical_price_volume", "latest_snapshot_mixed"],
            "horizon": [20, 20],
            "ic_mean": [0.03, 0.50],
            "ic_std": [0.20, 0.10],
            "ir": [0.15, 5.0],
        }
    )
    quantile_returns = pd.DataFrame(
        {
            "factor": ["historical_price_volume", "historical_price_volume"],
            "horizon": [20, 20],
            "quantile": [1, 5],
            "mean_return": [0.01, 0.04],
        }
    )
    turnover = pd.DataFrame(
        {
            "factor": ["historical_price_volume"],
            "average_best_bucket_turnover": [0.25],
        }
    )
    monotonicity = pd.DataFrame(
        {
            "factor": ["historical_price_volume"],
            "horizon": [20],
            "adjacent_agreement_ratio": [0.75],
            "monotonicity_pass": [False],
        }
    )

    scoreboard = build_factor_scoreboard(
        ic_summary=ic_summary,
        quantile_returns=quantile_returns,
        turnover=turnover,
        monotonicity=monotonicity,
        ohlcv_ticker_count=30,
    )

    required = {
        "factor",
        "best_horizon",
        "ic_mean",
        "ic_std",
        "ir",
        "top_bottom_spread",
        "avg_turnover",
        "monotonicity_score",
        "monotonicity_pass",
        "is_historical_ready",
        "is_backtest_candidate",
        "notes",
    }
    assert required.issubset(scoreboard.columns)
    default = scoreboard[scoreboard["factor"] == "historical_price_volume"].iloc[0]
    snapshot = scoreboard[scoreboard["factor"] == "latest_snapshot_mixed"].iloc[0]
    assert bool(default["is_backtest_candidate"])
    assert not bool(snapshot["is_backtest_candidate"])
    assert "snapshot-only" in snapshot["notes"]
    assert "limited OHLCV coverage: 30 tickers" in default["notes"]
    assert select_backtest_factor(scoreboard) == "historical_price_volume"

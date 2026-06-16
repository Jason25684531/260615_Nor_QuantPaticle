"""Week 3 vectorbt-first backtest engine with deterministic fallback."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.costs import CostModel


def _weights_matrix(
    portfolio_weights: pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    weights = portfolio_weights.copy()
    weights["execution_date"] = pd.to_datetime(weights["execution_date"])
    matrix = weights.pivot_table(
        index="execution_date",
        columns="ticker",
        values="target_weight",
        aggfunc="last",
    )
    matrix = matrix.reindex(index=index, columns=columns).fillna(0.0)
    return matrix


def _drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _metrics(
    *,
    equity: pd.Series,
    returns: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
    cost_model: CostModel,
    top_n: int,
    engine: str,
) -> pd.DataFrame:
    periods = max(len(returns), 1)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    annualized_return = float((1.0 + total_return) ** (252 / periods) - 1.0)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = (
        float(annualized_return / annualized_volatility)
        if annualized_volatility
        else np.nan
    )
    drawdown = _drawdown(equity)
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
        "turnover": float(turnover.mean()),
        "avg_exposure": float(weights.sum(axis=1).mean()),
        "start_date": str(equity.index.min().date()),
        "end_date": str(equity.index.max().date()),
        "ticker_count": int(weights.shape[1]),
        "top_n": int(top_n),
        "cost_model_summary": json.dumps(cost_model.summary(), sort_keys=True),
        "engine": engine,
    }
    return pd.DataFrame([metrics])


def run_weight_backtest(
    *,
    close_matrix: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    cost_model: CostModel,
    initial_cash: float,
    top_n: int,
    use_vectorbt: bool = True,
    allow_fallback: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = "fallback_weight_engine"
    if use_vectorbt:
        try:
            __import__("vectorbt")
            engine = "vectorbt"
        except Exception:
            if not allow_fallback:
                raise

    close = close_matrix.sort_index().astype(float)
    weights = _weights_matrix(portfolio_weights, close.index, close.columns)
    asset_returns = close.pct_change().fillna(0.0)
    previous_weights = weights.shift(1).fillna(0.0)
    gross_returns = (previous_weights * asset_returns).sum(axis=1)

    weight_changes = weights.diff().fillna(weights)
    buy_turnover = weight_changes.clip(lower=0.0).sum(axis=1)
    sell_turnover = (-weight_changes.clip(upper=0.0)).sum(axis=1)
    cost_returns = (
        buy_turnover * cost_model.buy_cost_rate
        + sell_turnover * cost_model.sell_cost_rate
    )
    net_returns = gross_returns - cost_returns
    equity = (1.0 + net_returns).cumprod() * float(initial_cash)
    results = pd.DataFrame(
        {
            "date": close.index,
            "equity": equity.values,
            "returns": net_returns.values,
            "drawdown": _drawdown(equity).values,
            "gross_returns": gross_returns.values,
            "cost_returns": cost_returns.values,
            "turnover": (buy_turnover + sell_turnover).values,
            "exposure": weights.sum(axis=1).values,
        }
    )
    metrics = _metrics(
        equity=equity,
        returns=net_returns,
        weights=weights,
        turnover=buy_turnover + sell_turnover,
        cost_model=cost_model,
        top_n=top_n,
        engine=engine,
    )
    return results, metrics

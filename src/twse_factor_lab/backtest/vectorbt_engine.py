"""Weight backtests with an honest vectorbt path and a deterministic reference."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.costs import CostModel


def _weights_matrix(
    portfolio_weights: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    """Expand complete target-replacement events into held daily weights."""
    weights = portfolio_weights.copy()
    weights["execution_date"] = pd.to_datetime(weights["execution_date"])
    invalid_dates = weights.loc[
        ~weights["execution_date"].isin(index), "execution_date"
    ]
    if not invalid_dates.empty:
        raise ValueError(
            "portfolio_weights contains execution dates outside close_matrix"
        )
    events = (
        weights.pivot_table(
            index="execution_date",
            columns="ticker",
            values="target_weight",
            aggfunc="last",
        )
        .reindex(columns=columns, fill_value=0.0)
        .fillna(0.0)
    )
    return events.reindex(index=index).ffill().fillna(0.0)


def _event_matrix(
    portfolio_weights: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    """Return target percentages only on execution dates; NaN means no order."""
    held = _weights_matrix(portfolio_weights, index, columns)
    event_dates = pd.to_datetime(portfolio_weights["execution_date"]).unique()
    events = pd.DataFrame(np.nan, index=index, columns=columns)
    events.loc[events.index.isin(event_dates)] = held.loc[
        events.index.isin(event_dates)
    ]
    return events


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _metrics(
    *,
    equity: pd.Series,
    returns: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
    cost_model: CostModel,
    top_n: int,
    requested_engine: str,
    actual_engine: str,
) -> pd.DataFrame:
    periods = max(len(returns), 1)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    annualized_return = float((1.0 + total_return) ** (252 / periods) - 1.0)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(252))
    metrics: dict[str, Any] = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": annualized_return / annualized_volatility
        if annualized_volatility
        else np.nan,
        "max_drawdown": float(_drawdown(equity).min()),
        "win_rate": float((returns > 0).mean()),
        "turnover": float(turnover.mean()),
        "avg_exposure": float(weights.sum(axis=1).mean()),
        "start_date": str(equity.index.min().date()),
        "end_date": str(equity.index.max().date()),
        "ticker_count": int(weights.shape[1]),
        "top_n": int(top_n),
        "cost_model_summary": json.dumps(cost_model.summary(), sort_keys=True),
        "engine": actual_engine,
        "requested_engine": requested_engine,
        "actual_engine": actual_engine,
    }
    return pd.DataFrame([metrics])


def _custom_backtest(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    cost_model: CostModel,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Execute canonical orders: stale marks, but no order without Close."""
    mark_prices = close.ffill()
    changes = weights.diff().fillna(weights)
    buy_turnover = changes.clip(lower=0.0).sum(axis=1)
    sell_turnover = -changes.clip(upper=0.0).sum(axis=1)
    shares = pd.Series(0.0, index=close.columns)
    cash = float(initial_cash)
    rows: list[dict[str, float | pd.Timestamp]] = []
    order_sizes = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    previous_equity = float(initial_cash)
    for date, raw_prices in close.iterrows():
        prices = mark_prices.loc[date]
        pre_trade_equity = float(cash + (shares * prices).sum())
        target = weights.loc[date]
        if changes.loc[date].ne(0).any():
            tradable = raw_prices.notna()
            desired = target * pre_trade_equity
            values = shares * prices
            sells = (values - desired).clip(lower=0.0).where(tradable, 0.0)
            cash += float((sells * (1 - cost_model.sell_cost_rate)).sum())
            shares -= sells / prices
            order_sizes.loc[date] -= sells / prices
            values = shares * prices
            buys = (desired - values).clip(lower=0.0).where(tradable, 0.0)
            total_buy = float((buys * (1 + cost_model.buy_cost_rate)).sum())
            if total_buy > cash and total_buy:
                buys *= cash / total_buy
                total_buy = cash
            cash -= total_buy
            shares += buys / prices
            order_sizes.loc[date] += buys / prices
        equity = float(cash + (shares * prices).sum())
        gross_return = pre_trade_equity / previous_equity - 1.0
        net_return = equity / previous_equity - 1.0
        row: dict[str, float | pd.Timestamp] = {
            "date": date,
            "equity": equity,
            "returns": net_return,
            "drawdown": 0.0,
            "gross_returns": gross_return,
            "cost_returns": gross_return - net_return,
            "turnover": float(buy_turnover.loc[date] + sell_turnover.loc[date]),
            "exposure": float((shares * prices).sum() / equity) if equity else 0.0,
            "cash": cash,
        }
        row.update(
            {
                f"position:{ticker}": float(shares[ticker] * prices[ticker])
                for ticker in close.columns
            }
        )
        rows.append(row)
        previous_equity = equity
    results = pd.DataFrame(rows)
    results["drawdown"] = _drawdown(results["equity"])
    returns = pd.Series(results["returns"].to_numpy(), index=close.index)
    return results, returns, buy_turnover + sell_turnover, order_sizes


def _vectorbt_backtest(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    events: pd.DataFrame,
    order_sizes: pd.DataFrame,
    cost_model: CostModel,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, pd.DataFrame]]:
    import vectorbt as vbt

    # CostModel defines slippage as a deterministic cost component, so it is
    # charged through vectorbt's fee rates below.  Passing it to vectorbt's
    # price-slippage parameter would create a second price semantics and
    # break Custom/Vectorbt parity.
    fees = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    event_mask = order_sizes.ne(0)
    fees[event_mask & (order_sizes >= 0)] = (
        cost_model.buy_fee_rate + cost_model.slippage_rate
    )
    fees[event_mask & (order_sizes < 0)] = (
        cost_model.sell_fee_rate
        + cost_model.transaction_tax_rate
        + cost_model.slippage_rate
    )
    slippage = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    portfolio = vbt.Portfolio.from_orders(
        close.ffill(),
        size=order_sizes,
        size_type="amount",
        direction="longonly",
        price=close,
        call_seq="auto",
        fees=fees,
        slippage=slippage,
        init_cash=float(initial_cash),
        cash_sharing=True,
        group_by=True,
        freq="D",
    )
    equity = portfolio.value(group_by=True)
    returns = portfolio.returns(group_by=True).fillna(0.0)
    positions = portfolio.asset_value(group_by=False)
    cash = portfolio.cash(group_by=True)
    changes = weights.diff().fillna(weights)
    turnover = changes.abs().sum(axis=1)
    results = pd.DataFrame(
        {
            "date": close.index,
            "equity": equity.to_numpy(),
            "returns": returns.to_numpy(),
            "drawdown": _drawdown(equity).to_numpy(),
            "gross_returns": np.nan,
            "cost_returns": np.nan,
            "turnover": turnover.to_numpy(),
            "exposure": weights.sum(axis=1).to_numpy(),
            "cash": cash.to_numpy(),
        }
    )
    position_values = positions.copy()
    position_values.columns = [f"position:{ticker}" for ticker in position_values]
    results = pd.concat([results, position_values.reset_index(drop=True)], axis=1)
    artifacts = {
        "orders": portfolio.orders.records_readable,
        "trades": portfolio.trades.records_readable,
        "positions": positions.reset_index(names="date"),
        "returns": pd.DataFrame({"date": close.index, "returns": returns.to_numpy()}),
    }
    return results, returns, turnover, artifacts


def run_weight_backtest(
    *,
    close_matrix: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    cost_model: CostModel,
    initial_cash: float,
    top_n: int,
    use_vectorbt: bool = True,
    allow_fallback: bool = True,
    artifacts_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run vectorbt only when it really executes; otherwise label the reference path."""
    close = close_matrix.sort_index().astype(float)
    weights = _weights_matrix(portfolio_weights, close.index, close.columns)
    requested_engine = "vectorbt" if use_vectorbt else "custom"
    actual_engine = "custom"
    try:
        if not use_vectorbt:
            raise ImportError("vectorbt was not requested")
        custom_results, custom_returns, custom_turnover, order_sizes = _custom_backtest(
            close, weights, cost_model, initial_cash
        )
        results, returns, turnover, artifacts = _vectorbt_backtest(
            close,
            weights,
            _event_matrix(portfolio_weights, close.index, close.columns),
            order_sizes,
            cost_model,
            initial_cash,
        )
        actual_engine = "vectorbt"
        if artifacts_dir:
            from pathlib import Path

            output = Path(artifacts_dir)
            output.mkdir(parents=True, exist_ok=True)
            for name, frame in artifacts.items():
                frame.to_parquet(output / f"vectorbt_{name}.parquet", index=False)
    except Exception:
        if use_vectorbt and not allow_fallback:
            raise
        results, returns, turnover, _ = _custom_backtest(
            close, weights, cost_model, initial_cash
        )
        actual_engine = "fallback_custom" if use_vectorbt else "custom"
    return results, _metrics(
        equity=pd.Series(results["equity"].to_numpy(), index=close.index),
        returns=returns,
        weights=weights,
        turnover=turnover,
        cost_model=cost_model,
        top_n=top_n,
        requested_engine=requested_engine,
        actual_engine=actual_engine,
    )

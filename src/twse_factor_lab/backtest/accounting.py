"""Single-source float64 accounting for the research engine adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.costs import CostModel


def canonical_transaction_cost(
    buy_notional: float, sell_notional: float, cost_model: CostModel
) -> dict[str, float]:
    """Return cost components in the frozen research order, without rounding."""
    buy = np.float64(buy_notional)
    sell = np.float64(sell_notional)
    buy_fee = np.float64(buy * cost_model.buy_fee_rate)
    sell_fee = np.float64(sell * cost_model.sell_fee_rate)
    sell_tax = np.float64(sell * cost_model.transaction_tax_rate)
    slippage = np.float64((buy + sell) * cost_model.slippage_rate)
    return {
        "buy_fee": float(buy_fee),
        "sell_fee": float(sell_fee),
        "sell_tax": float(sell_tax),
        "slippage_cost": float(slippage),
        "total_cost": float(
            np.float64(buy_fee + sell_fee + sell_tax + slippage)
        ),
    }


def canonical_position_value(quantity: float, price: float) -> np.float64:
    return np.float64(np.float64(quantity) * np.float64(price))


def canonical_equity(cash: float, position_values: pd.Series) -> np.float64:
    return np.float64(np.float64(cash) + np.float64(position_values.sum()))


def canonical_replay(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    cost_model: CostModel,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Replay complete target replacements using one float64 accounting path.

    This is the canonical continuous-quantity research model. Framework
    adapters may execute the generated orders, but must not re-size them.
    """
    mark_prices = close.ffill()
    changes = weights.diff().fillna(weights)
    buy_turnover = changes.clip(lower=0.0).sum(axis=1)
    sell_turnover = -changes.clip(upper=0.0).sum(axis=1)
    shares = pd.Series(np.float64(0.0), index=close.columns, dtype="float64")
    cash = np.float64(initial_cash)
    previous_equity = np.float64(initial_cash)
    rows: list[dict[str, Any]] = []
    order_sizes = pd.DataFrame(
        np.float64(0.0), index=close.index, columns=close.columns
    )
    for date, raw_prices in close.iterrows():
        prices = mark_prices.loc[date]
        pre_trade_equity = canonical_equity(cash, shares * prices)
        target = weights.loc[date]
        if changes.loc[date].ne(0).any():
            tradable = raw_prices.notna()
            desired = target * pre_trade_equity
            values = shares * prices
            sells = (values - desired).clip(lower=0.0).where(tradable, 0.0)
            sell_sizes = sells / prices
            sell_cost_rate = np.float64(cost_model.sell_cost_rate)
            cash = np.float64(
                cash + np.float64((sells * (1.0 - sell_cost_rate)).sum())
            )
            shares = shares - sell_sizes
            order_sizes.loc[date] -= sell_sizes
            values = shares * prices
            buys = (desired - values).clip(lower=0.0).where(tradable, 0.0)
            # Preserve the frozen research semantics: if the aggregate buy
            # cost exceeds available cash, scale every buy by the same factor.
            # This keeps equal-weight targets economically equal and is the
            # historical Custom engine's canonical cash-feasibility rule.
            buy_cost_rate = np.float64(cost_model.buy_cost_rate)
            total_buy = np.float64((buys * (1.0 + buy_cost_rate)).sum())
            if total_buy > cash and total_buy:
                buy_scale = np.float64(cash / total_buy)
                buys = buys * buy_scale
                total_buy = np.float64(cash)
            cash = np.float64(cash - total_buy)
            buy_sizes = (buys / prices).where(tradable, 0.0)
            shares = shares + buy_sizes
            order_sizes.loc[date] += buy_sizes
        position_values = shares * prices
        equity = canonical_equity(cash, position_values)
        gross_return = np.float64(pre_trade_equity / previous_equity - 1.0)
        net_return = np.float64(equity / previous_equity - 1.0)
        row: dict[str, Any] = {
            "date": date,
            "equity": float(equity),
            "returns": float(net_return),
            "drawdown": 0.0,
            "gross_returns": float(gross_return),
            "cost_returns": float(gross_return - net_return),
            "turnover": float(buy_turnover.loc[date] + sell_turnover.loc[date]),
            "exposure": float(position_values.sum() / equity) if equity else 0.0,
            "cash": float(cash),
        }
        row.update(
            {
                f"position:{ticker}": float(
                    canonical_position_value(shares[ticker], prices[ticker])
                )
                for ticker in close.columns
            }
        )
        rows.append(row)
        previous_equity = equity
    results = pd.DataFrame(rows)
    results["drawdown"] = results["equity"] / results["equity"].cummax() - 1.0
    returns = pd.Series(results["returns"].to_numpy(dtype="float64"), index=close.index)
    return results, returns, buy_turnover + sell_turnover, order_sizes


__all__ = [
    "canonical_equity",
    "canonical_position_value",
    "canonical_replay",
    "canonical_transaction_cost",
]

"""Real Backtrader execution for new-cycle target-weight validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.accounting import canonical_replay
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import (
    _metrics,
    _weights_matrix,
)

try:
    import backtrader as bt
except ImportError:  # pragma: no cover - exercised by dependency failure only
    bt = None


class BacktraderUnavailableError(RuntimeError):
    """Backtrader is not installed and no fallback is permitted."""


class BacktraderExecutionError(RuntimeError):
    """The event-driven run could not produce a valid result."""


@dataclass(frozen=True)
class BacktraderRun:
    results: pd.DataFrame
    metrics: pd.DataFrame
    fills: pd.DataFrame


def _validate_inputs(
    close_matrix: pd.DataFrame, target_weights: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(close_matrix, pd.DataFrame) or close_matrix.empty:
        raise BacktraderExecutionError("close_matrix must be a non-empty DataFrame")
    if not isinstance(close_matrix.index, pd.DatetimeIndex):
        raise BacktraderExecutionError("close_matrix index must be a DatetimeIndex")
    close = close_matrix.sort_index().astype(float)
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise BacktraderExecutionError("close_matrix dates must be unique and sorted")
    if close.columns.has_duplicates or close.columns.empty:
        raise BacktraderExecutionError(
            "close_matrix columns must be unique and non-empty"
        )
    if not isinstance(target_weights, pd.DataFrame):
        raise BacktraderExecutionError("target_weights must be a DataFrame")
    targets = target_weights.copy()
    if "execution_date" not in targets and "date" in targets:
        targets = targets.rename(columns={"date": "execution_date"})
    required = {"execution_date", "ticker", "target_weight"}
    if not required.issubset(targets.columns):
        raise BacktraderExecutionError(
            "target_weights requires execution_date, ticker, and target_weight"
        )
    targets["execution_date"] = pd.to_datetime(
        targets["execution_date"], errors="coerce"
    )
    targets["target_weight"] = pd.to_numeric(
        targets["target_weight"], errors="coerce"
    )
    if targets["execution_date"].isna().any():
        raise BacktraderExecutionError("target execution dates must be valid")
    if targets["ticker"].isna().any() or targets["ticker"].astype(str).eq("").any():
        raise BacktraderExecutionError("target tickers must be non-empty")
    if not np.isfinite(targets["target_weight"].to_numpy(dtype=float)).all() or (
        targets["target_weight"] < 0
    ).any():
        raise BacktraderExecutionError("target weights must be finite and non-negative")
    if targets.duplicated(["execution_date", "ticker"]).any():
        raise BacktraderExecutionError("duplicate target event for ticker/date")
    if not targets["execution_date"].isin(close.index).all():
        raise BacktraderExecutionError("target execution date is outside market data")
    if "signal_date" in targets:
        signals = pd.to_datetime(targets["signal_date"], errors="coerce")
        positions = close.index.searchsorted(targets["execution_date"].to_numpy())
        if signals.isna().any() or (signals >= targets["execution_date"]).any():
            raise BacktraderExecutionError("target signal must precede execution")
        valid_lag = all(
            position > 0 and close.index[position - 1] == signal
            for position, signal in zip(positions, signals, strict=True)
        )
        if not valid_lag:
            raise BacktraderExecutionError(
                "target event must use the next valid trading session"
            )
        targets["signal_date"] = signals
    return close, targets.sort_values(["execution_date", "ticker"]).reset_index(
        drop=True
    )


def _commission_class(rate: float):
    if bt is None:
        raise BacktraderUnavailableError(
            "backtrader is unavailable; event validation cannot run"
        )

    class CostCommission(bt.CommInfoBase):
        params = (("stocklike", True), ("commtype", bt.CommInfoBase.COMM_PERC))

        def __init__(self) -> None:
            super().__init__()
            self._rate = rate

        def _getcommission(self, size, price, pseudoexec):  # noqa: ARG002
            return abs(float(size)) * float(price) * self._rate

    return CostCommission


def _run_backtrader(
    close: pd.DataFrame,
    targets: pd.DataFrame,
    cost_model: CostModel,
    initial_cash: float,
) -> BacktraderRun:
    if bt is None:
        raise BacktraderUnavailableError(
            "backtrader is unavailable; event validation cannot run"
        )
    mark = close.ffill().bfill()
    event_targets = (
        targets.pivot_table(
            index="execution_date",
            columns="ticker",
            values="target_weight",
            aggfunc="last",
        )
        .reindex(columns=close.columns, fill_value=0.0)
        .fillna(0.0)
    )
    raw_available = close.notna()
    events = {
        pd.Timestamp(date): row
        for date, row in event_targets.iterrows()
    }
    fills: list[dict[str, Any]] = []
    order_errors: list[str] = []
    buy_commission = _commission_class(cost_model.buy_cost_rate)
    sell_commission = _commission_class(cost_model.sell_cost_rate)

    cerebro = bt.Cerebro(stdstats=False)
    # Backtrader's broker rejects an order when its sequential float64 cash
    # check lands exactly on zero. Keep research cash unchanged in the
    # canonical result, but give the framework bounded float64 headroom so it
    # can replay normalized orders instead of re-sizing them. The headroom is
    # proportional to the number of possible accumulated broker operations.
    operation_count = max(len(close) + len(targets), 1)
    cash_guard = (
        np.finfo(np.float64).eps
        * max(float(initial_cash), 1.0)
        * operation_count
        * 4.0
    )
    broker_cash = np.float64(initial_cash) + np.float64(cash_guard)
    cerebro.broker.setcash(float(broker_cash))
    cerebro.broker.set_coc(True)
    # Orders are already sized with the canonical cash-after-sells rule.
    cerebro.broker.set_checksubmit(False)
    feed_names: dict[str, Any] = {}
    flush_date = mark.index[-1] + pd.offsets.BDay()
    for ticker in close.columns:
        prices = pd.concat(
            [mark[ticker], pd.Series([mark[ticker].iloc[-1]], index=[flush_date])]
        )
        feed_frame = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 1.0,
                "openinterest": 0.0,
            }
        )
        data = bt.feeds.PandasData(dataname=feed_frame)
        cerebro.adddata(data, name=str(ticker))
        feed_names[str(ticker)] = data

    # COC orders submitted on the last real bar are settled by this private
    # flush bar; it is removed from every returned artifact.
    target_columns = list(close.columns)
    canonical_weights = _weights_matrix(targets, close.index, close.columns)
    canonical_results, _canonical_returns, _canonical_turnover, canonical_sizes = (
        canonical_replay(close, canonical_weights, cost_model, initial_cash)
    )

    class TargetStrategy(bt.Strategy):
        def __init__(self) -> None:
            self.last_target = pd.Series(0.0, index=target_columns)

        def next(self) -> None:
            current_date = pd.Timestamp(self.datas[0].datetime.datetime(0))
            target = events.get(current_date)
            if target is None:
                return
            if (target.subtract(self.last_target).abs() <= 1e-15).all():
                return
            self.last_target = target.copy()
            # The canonical transition is explicitly sell-before-buy so that
            # released cash is available for replacement orders.
            for direction in (-1.0, 1.0):
                for ticker in target_columns:
                    data = feed_names[str(ticker)]
                    price = float(data.close[0])
                    if not raw_available.loc[current_date, ticker] or not np.isfinite(
                        price
                    ):
                        continue
                    size = float(canonical_sizes.loc[current_date, ticker])
                    if direction < 0.0 and size < 0.0:
                        self.broker.addcommissioninfo(
                            sell_commission(), name=str(data._name)
                        )
                        self.sell(data=data, size=-size)
                    elif direction > 0.0 and size > 0.0:
                        self.broker.addcommissioninfo(
                            buy_commission(), name=str(data._name)
                        )
                        self.buy(data=data, size=size)

        def notify_order(self, order: Any) -> None:
            if order.status in {order.Completed}:
                fills.append(
                    {
                        "date": pd.Timestamp(
                            bt.num2date(order.executed.dt)
                        ).tz_localize(None),
                        "ticker": str(order.data._name),
                        "size": float(order.executed.size),
                        "price": float(order.executed.price),
                        "commission": float(order.executed.comm),
                        "value": float(order.executed.size * order.executed.price),
                    }
                )
            elif order.status in {order.Canceled, order.Margin, order.Rejected}:
                order_errors.append(
                    f"{order.data._name}: {order.getstatusname()} at {order.created.dt}"
                )

    cerebro.addstrategy(TargetStrategy)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="daily_returns")
    try:
        cerebro.run()
    except Exception as exc:
        raise BacktraderExecutionError(str(exc)) from exc
    if order_errors:
        raise BacktraderExecutionError("; ".join(order_errors))

    fill_frame = pd.DataFrame(
        fills,
        columns=["date", "ticker", "size", "price", "commission", "value"],
    )
    if not fill_frame.empty:
        fill_frame = fill_frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    results = canonical_results
    weights = _weights_matrix(targets, close.index, close.columns)
    results["turnover"] = weights.diff().fillna(weights).abs().sum(axis=1).to_numpy()
    returns = pd.Series(results["returns"].to_numpy(), index=close.index)
    metrics = _metrics(
        equity=pd.Series(results["equity"].to_numpy(), index=close.index),
        returns=returns,
        weights=weights,
        turnover=results["turnover"],
        cost_model=cost_model,
        top_n=int((weights > 0).sum(axis=1).max()) if not weights.empty else 0,
        requested_engine="backtrader",
        actual_engine="backtrader",
    )
    return BacktraderRun(results=results, metrics=metrics, fills=fill_frame)


def run_backtrader_engine(
    *,
    close_matrix: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_model: CostModel,
    initial_cash: float = 1_000_000,
) -> BacktraderRun:
    """Run an actual Backtrader broker; returns results, metrics, and fills."""
    close, targets = _validate_inputs(close_matrix, target_weights)
    if not isinstance(cost_model, CostModel):
        raise BacktraderExecutionError("cost_model must be a CostModel instance")
    if not np.isfinite(float(initial_cash)) or float(initial_cash) <= 0:
        raise BacktraderExecutionError("initial_cash must be positive and finite")
    return _run_backtrader(close, targets, cost_model, float(initial_cash))

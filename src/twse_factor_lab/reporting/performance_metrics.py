"""Canonical-backed metrics for the comprehensive performance tear sheet."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.robustness import compute_metrics

from .performance_adapter import PerformanceData

ANNUALIZATION_SESSIONS = 252
ROLLING_WINDOWS = (63, 126, 252)
DEFAULT_ROLLING_WINDOW = 126
TAIL_WARNING_OBSERVATIONS = 60
VAR_QUANTILE = 0.05

METRIC_NAMES = (
    "total_return",
    "cagr",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "max_drawdown_start",
    "max_drawdown_trough",
    "max_drawdown_recovery",
    "max_drawdown_duration",
    "best_day",
    "best_day_date",
    "worst_day",
    "worst_day_date",
    "best_month",
    "best_month_date",
    "worst_month",
    "worst_month_date",
    "positive_day_ratio",
    "positive_month_ratio",
    "skew",
    "kurtosis",
    "var_95",
    "cvar_95",
    "alpha",
    "beta",
    "information_ratio",
    "tracking_error",
    "benchmark_return",
    "benchmark_cagr",
    "benchmark_sharpe",
    "benchmark_max_drawdown",
    "excess_return",
    "active_return",
    "turnover",
    "actual_rebalance_count",
    "commission",
    "tax",
    "slippage",
    "transaction_cost",
    "cost_to_gross_return",
    "gross_return",
    "net_return",
    "cost_drag",
    "gross_exposure",
    "net_exposure",
    "average_position_count",
    "largest_position_weight",
    "average_largest_position_weight",
    "top_3_weight",
    "top_5_weight",
    "top_holdings_contribution",
    "hhi",
    "round_trip_win_rate",
    "round_trip_loss_rate",
    "average_win",
    "average_loss",
    "profit_factor",
    "median_holding_period",
    "average_holding_period",
    "best_trade",
    "worst_trade",
)


@dataclass(frozen=True)
class MetricValue:
    value: Any
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "status": self.status, "reason": self.reason}


@dataclass
class PerformanceMetrics:
    period: str
    status: str
    metrics: dict[str, MetricValue]
    drawdowns: list[dict[str, Any]] = field(default_factory=list)
    rolling: dict[str, dict[str, Any]] = field(default_factory=dict)
    calendar: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    transaction_analysis: dict[str, Any] = field(default_factory=dict)
    series: dict[str, pd.Series | pd.DataFrame] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metrics": {
                key: value.to_dict() for key, value in self.metrics.items()
            },
            "drawdowns": self.drawdowns,
            "rolling": self.rolling,
            "calendar": self.calendar,
            "warnings": self.warnings,
            "transaction_analysis": self.transaction_analysis,
        }


def _value(value: Any, reason: str = "METRIC_UNDEFINED") -> MetricValue:
    if isinstance(value, pd.Timestamp):
        value = value.date().isoformat()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return MetricValue(int(value), "AVAILABLE")
    if isinstance(value, (np.floating, float)):
        return (
            MetricValue(float(value), "AVAILABLE")
            if math.isfinite(float(value))
            else MetricValue(None, "INSUFFICIENT_DATA", reason)
        )
    if value is None:
        return MetricValue(None, "INSUFFICIENT_DATA", reason)
    return MetricValue(value, "AVAILABLE")


def _missing(reason: str, status: str = "UNAVAILABLE") -> MetricValue:
    return MetricValue(None, status, reason)


def unavailable_metrics(data: PerformanceData) -> PerformanceMetrics:
    reason = data.reason or "RETURNS_UNAVAILABLE"
    return PerformanceMetrics(
        data.period,
        "INSUFFICIENT_DATA",
        {
            name: _missing(reason, "INSUFFICIENT_DATA")
            for name in METRIC_NAMES
        },
        transaction_analysis=data.transaction_status,
    )


def _monthly(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).resample("ME").prod().sub(1.0).rename("monthly_return")


def _annual(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).resample("YE").prod().sub(1.0).rename("annual_return")


def _drawdowns(returns: pd.Series) -> tuple[pd.Series, list[dict[str, Any]]]:
    equity = (1.0 + returns).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0).rename("drawdown")
    episodes: list[dict[str, Any]] = []
    start: pd.Timestamp | None = None
    trough: pd.Timestamp | None = None
    trough_value = 0.0
    for position, (date, value) in enumerate(drawdown.items()):
        if value < 0 and start is None:
            start = drawdown.index[max(0, position - 1)]
            trough = date
            trough_value = float(value)
        elif start is not None and value < trough_value:
            trough = date
            trough_value = float(value)
        if start is not None and value >= 0:
            episodes.append(_drawdown_row(start, trough, date, trough_value, drawdown))
            start = trough = None
            trough_value = 0.0
    if start is not None:
        episodes.append(
            _drawdown_row(start, trough, None, trough_value, drawdown)
        )
    episodes.sort(key=lambda item: item["drawdown"])
    for rank, episode in enumerate(episodes[:5], 1):
        episode["rank"] = rank
    return drawdown, episodes[:5]


def _drawdown_row(
    start: pd.Timestamp,
    trough: pd.Timestamp | None,
    recovery: pd.Timestamp | None,
    value: float,
    drawdown: pd.Series,
) -> dict[str, Any]:
    end = recovery or drawdown.index[-1]
    duration = max(0, len(drawdown.loc[start:end]) - 1)
    recovery_duration = (
        max(0, len(drawdown.loc[trough:recovery]) - 1)
        if trough is not None and recovery is not None
        else None
    )
    return {
        "start": start.date().isoformat(),
        "trough": trough.date().isoformat() if trough is not None else None,
        "recovery": (
            recovery.date().isoformat() if recovery is not None else "NOT_RECOVERED"
        ),
        "drawdown": value,
        "duration": duration,
        "recovery_duration": recovery_duration,
    }


def _rolling_summary(series: pd.Series | None, reason: str) -> dict[str, Any]:
    if series is None or series.dropna().empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": reason,
            "latest": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "negative_period_ratio": None,
        }
    values = series.dropna()
    return {
        "status": "AVAILABLE",
        "reason": None,
        "latest": float(values.iloc[-1]),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "negative_period_ratio": float((values < 0).mean()),
    }


def _rolling(
    returns: pd.Series,
    benchmark: pd.Series | None,
    window: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.Series]]:
    reason = f"REQUIRES_{window}_OBSERVATIONS"
    if len(returns) < window:
        empty = {
            name: _rolling_summary(None, reason)
            for name in ("return", "sharpe", "volatility", "beta")
        }
        for item in empty.values():
            item["window"] = window
        return empty, {}
    total = (1.0 + returns).rolling(window).apply(np.prod, raw=True).sub(1.0)
    cagr = (1.0 + total).pow(ANNUALIZATION_SESSIONS / window).sub(1.0)
    volatility = returns.rolling(window).std(ddof=0) * np.sqrt(
        ANNUALIZATION_SESSIONS
    )
    sharpe = cagr.div(volatility.where(volatility.ne(0)))
    beta = None
    if benchmark is not None:
        aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
        if len(aligned) >= window:
            strategy_column, benchmark_column = aligned.columns
            covariance = aligned[strategy_column].rolling(window).cov(
                aligned[benchmark_column], ddof=0
            )
            variance = aligned[benchmark_column].rolling(window).var(ddof=0)
            beta = covariance.div(variance.where(variance.ne(0)))
    series = {
        "rolling_return": total,
        "rolling_sharpe": sharpe,
        "rolling_volatility": volatility,
    }
    if beta is not None:
        series["rolling_beta"] = beta
    summaries = {
        "return": _rolling_summary(total, reason),
        "sharpe": _rolling_summary(sharpe, reason),
        "volatility": _rolling_summary(volatility, reason),
        "beta": _rolling_summary(beta, "BENCHMARK_OR_WINDOW_UNAVAILABLE"),
    }
    for item in summaries.values():
        item["window"] = window
    return summaries, series


def _benchmark_metrics(
    returns: pd.Series, benchmark: pd.Series | None
) -> tuple[dict[str, MetricValue], dict[str, pd.Series]]:
    names = (
        "alpha",
        "beta",
        "information_ratio",
        "tracking_error",
        "benchmark_return",
        "benchmark_cagr",
        "benchmark_sharpe",
        "benchmark_max_drawdown",
        "excess_return",
        "active_return",
    )
    if benchmark is None:
        return ({name: _missing("BENCHMARK_UNAVAILABLE") for name in names}, {})
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return (
            {name: _missing("BENCHMARK_ALIGNMENT_INSUFFICIENT") for name in names},
            {},
        )
    strategy = aligned["strategy"]
    bench = aligned["benchmark"]
    bench_metrics = compute_metrics(bench)
    active = strategy - bench
    variance = float(bench.var(ddof=0))
    beta = float(strategy.cov(bench, ddof=0) / variance) if variance else np.nan
    alpha = (
        float((strategy.mean() - beta * bench.mean()) * ANNUALIZATION_SESSIONS)
        if math.isfinite(beta)
        else np.nan
    )
    tracking = float(active.std(ddof=0) * np.sqrt(ANNUALIZATION_SESSIONS))
    information = (
        float(active.mean() * ANNUALIZATION_SESSIONS / tracking)
        if tracking
        else np.nan
    )
    strategy_total = float((1.0 + strategy).prod() - 1.0)
    benchmark_total = float((1.0 + bench).prod() - 1.0)
    return (
        {
            "alpha": _value(alpha),
            "beta": _value(beta),
            "information_ratio": _value(information),
            "tracking_error": _value(tracking),
            "benchmark_return": _value(benchmark_total),
            "benchmark_cagr": _value(bench_metrics["cagr"]),
            "benchmark_sharpe": _value(bench_metrics["sharpe"]),
            "benchmark_max_drawdown": _value(bench_metrics["max_drawdown"]),
            "excess_return": _value(strategy_total - benchmark_total),
            "active_return": _value(float(active.mean() * ANNUALIZATION_SESSIONS)),
        },
        {"benchmark": bench, "active_return": active},
    )


def _position_metrics(
    positions: pd.DataFrame | None,
) -> tuple[dict[str, MetricValue], dict[str, pd.Series]]:
    names = (
        "gross_exposure",
        "net_exposure",
        "average_position_count",
        "largest_position_weight",
        "average_largest_position_weight",
        "top_3_weight",
        "top_5_weight",
        "top_holdings_contribution",
        "hhi",
    )
    if positions is None or positions.empty:
        return ({name: _missing("POSITIONS_UNAVAILABLE") for name in names}, {})
    grouped = positions.groupby("date", sort=True)
    portfolio = grouped["portfolio_value"].first().replace(0, np.nan)
    market = grouped["market_value"]
    gross = market.apply(lambda values: values.abs().sum()).div(portfolio)
    net = market.sum().div(portfolio)
    weights = positions.assign(abs_weight=positions["weight"].abs())
    weight_group = weights.groupby("date", sort=True)["abs_weight"]
    count = weight_group.apply(lambda values: int((values > 0).sum()))
    largest = weight_group.max()
    top3 = weight_group.apply(lambda values: values.nlargest(3).sum())
    top5 = weight_group.apply(lambda values: values.nlargest(5).sum())
    hhi = weight_group.apply(lambda values: float((values**2).sum()))
    return (
        {
            "gross_exposure": _value(gross.mean()),
            "net_exposure": _value(net.mean()),
            "average_position_count": _value(count.mean()),
            "largest_position_weight": _value(largest.max()),
            "average_largest_position_weight": _value(largest.mean()),
            "top_3_weight": _value(top3.mean()),
            "top_5_weight": _value(top5.mean()),
            "top_holdings_contribution": _missing(
                "POSITION_RETURN_ATTRIBUTION_UNAVAILABLE"
            ),
            "hhi": _value(hhi.mean()),
        },
        {
            "gross_exposure": gross,
            "net_exposure": net,
            "position_concentration": hhi,
        },
    )


def _transaction_metrics(
    transactions: pd.DataFrame | None,
    positions: pd.DataFrame | None,
    net_return: float,
) -> tuple[dict[str, MetricValue], dict[str, pd.Series], pd.DataFrame]:
    names = (
        "turnover",
        "actual_rebalance_count",
        "commission",
        "tax",
        "slippage",
        "transaction_cost",
        "cost_to_gross_return",
        "gross_return",
        "net_return",
        "cost_drag",
    )
    values = {"net_return": _value(net_return)}
    if transactions is None or transactions.empty:
        values.update(
            {
                name: _missing("TRANSACTIONS_UNAVAILABLE")
                for name in names
                if name != "net_return"
            }
        )
        return values, {}, pd.DataFrame()
    daily_value = transactions.groupby("date")["value"].apply(
        lambda item: item.abs().sum()
    )
    daily_cost = transactions.groupby("date")["total_cost"].sum()
    turnover = None
    initial_value = None
    if positions is not None and not positions.empty:
        portfolio = positions.groupby("date")["portfolio_value"].first()
        aligned = pd.concat([daily_value, portfolio], axis=1, join="inner").dropna()
        if not aligned.empty:
            turnover = aligned.iloc[:, 0].div(
                aligned.iloc[:, 1].replace(0, np.nan)
            )
        initial_value = float(portfolio.iloc[0]) if len(portfolio) else None
    total_cost = float(transactions["total_cost"].sum())
    cost_drag = (
        total_cost / initial_value
        if initial_value is not None and initial_value
        else np.nan
    )
    gross_return = net_return + cost_drag if math.isfinite(cost_drag) else np.nan
    values.update(
        {
            "turnover": _value(turnover.mean() if turnover is not None else np.nan),
            "actual_rebalance_count": _value(transactions["date"].nunique()),
            "commission": _value(transactions["commission"].sum()),
            "tax": _value(transactions["tax"].sum()),
            "slippage": _value(transactions["slippage"].sum()),
            "transaction_cost": _value(total_cost),
            "cost_drag": _value(cost_drag),
            "gross_return": _value(gross_return),
            "cost_to_gross_return": _value(
                abs(cost_drag / gross_return) if gross_return else np.nan
            ),
        }
    )
    round_trips = _round_trips(transactions)
    series: dict[str, pd.Series] = {"transaction_cost": daily_cost}
    if turnover is not None:
        series["turnover"] = turnover
    return values, series, round_trips


def _round_trips(transactions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for ticker, rows in transactions.groupby("ticker", sort=True):
        lots: deque[dict[str, Any]] = deque()
        for row in rows.sort_values("date", kind="stable").itertuples():
            amount = float(row.amount)
            unit_cost = float(row.total_cost) / abs(amount) if amount else 0.0
            if amount > 0:
                lots.append(
                    {
                        "amount": amount,
                        "price": float(row.price),
                        "date": row.date,
                        "unit_cost": unit_cost,
                    }
                )
                continue
            remaining = abs(amount)
            while remaining > 0 and lots:
                lot = lots[0]
                quantity = min(remaining, lot["amount"])
                basis = quantity * lot["price"] + quantity * lot["unit_cost"]
                proceeds = quantity * float(row.price) - quantity * unit_cost
                profit = proceeds - basis
                records.append(
                    {
                        "ticker": ticker,
                        "return": profit / basis if basis else np.nan,
                        "profit": profit,
                        "duration": (row.date - lot["date"]).days,
                    }
                )
                remaining -= quantity
                lot["amount"] -= quantity
                if lot["amount"] <= 0:
                    lots.popleft()
    return pd.DataFrame(records)


def _round_trip_metrics(round_trips: pd.DataFrame) -> dict[str, MetricValue]:
    names = (
        "round_trip_win_rate",
        "round_trip_loss_rate",
        "average_win",
        "average_loss",
        "profit_factor",
        "median_holding_period",
        "average_holding_period",
        "best_trade",
        "worst_trade",
    )
    if round_trips.empty:
        return {name: _missing("CLOSED_ROUND_TRIPS_UNAVAILABLE") for name in names}
    returns = round_trips["return"].dropna()
    profits = round_trips["profit"].dropna()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = profits[profits > 0].sum()
    gross_loss = abs(profits[profits < 0].sum())
    return {
        "round_trip_win_rate": _value((returns > 0).mean()),
        "round_trip_loss_rate": _value((returns < 0).mean()),
        "average_win": _value(wins.mean() if not wins.empty else np.nan),
        "average_loss": _value(losses.mean() if not losses.empty else np.nan),
        "profit_factor": _value(gross_profit / gross_loss if gross_loss else np.nan),
        "median_holding_period": _value(round_trips["duration"].median()),
        "average_holding_period": _value(round_trips["duration"].mean()),
        "best_trade": _value(returns.max()),
        "worst_trade": _value(returns.min()),
    }


def calculate_metrics(
    data: PerformanceData,
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> PerformanceMetrics:
    """Calculate canonical metrics plus deterministic diagnostics for one period."""

    if not data.available or data.returns is None:
        return unavailable_metrics(data)
    if rolling_window not in ROLLING_WINDOWS:
        raise ValueError(f"rolling window must be one of {ROLLING_WINDOWS}")
    returns = data.returns
    canonical = compute_metrics(returns)
    monthly = _monthly(returns)
    annual = _annual(returns)
    drawdown, drawdowns = _drawdowns(returns)
    primary = drawdowns[0] if drawdowns else None
    metrics: dict[str, MetricValue] = {
        "total_return": _value(canonical["total_return"]),
        "cagr": _value(canonical["cagr"]),
        "annualized_return": _value(canonical["cagr"]),
        "annualized_volatility": _value(canonical["volatility"]),
        "sharpe": _value(canonical["sharpe"]),
        "sortino": _value(canonical["sortino"]),
        "calmar": _value(canonical["calmar"]),
        "max_drawdown": _value(canonical["max_drawdown"]),
        "max_drawdown_start": _value(
            primary["start"] if primary else None, "NO_DRAWDOWN"
        ),
        "max_drawdown_trough": _value(
            primary["trough"] if primary else None, "NO_DRAWDOWN"
        ),
        "max_drawdown_recovery": _value(
            primary["recovery"] if primary else None, "NO_DRAWDOWN"
        ),
        "max_drawdown_duration": _value(
            primary["duration"] if primary else None, "NO_DRAWDOWN"
        ),
        "best_day": _value(returns.max()),
        "best_day_date": _value(returns.idxmax()),
        "worst_day": _value(returns.min()),
        "worst_day_date": _value(returns.idxmin()),
        "best_month": _value(monthly.max()),
        "best_month_date": _value(monthly.idxmax()),
        "worst_month": _value(monthly.min()),
        "worst_month_date": _value(monthly.idxmin()),
        "positive_day_ratio": _value((returns > 0).mean()),
        "positive_month_ratio": _value((monthly > 0).mean()),
        "skew": _value(returns.skew() if len(returns) >= 3 else np.nan),
        "kurtosis": _value(returns.kurt() if len(returns) >= 4 else np.nan),
    }
    var = float(returns.quantile(VAR_QUANTILE))
    tail = returns[returns <= var]
    metrics["var_95"] = _value(var)
    metrics["cvar_95"] = _value(tail.mean() if not tail.empty else np.nan)
    benchmark_metrics, benchmark_series = _benchmark_metrics(
        returns, data.benchmark_returns
    )
    metrics.update(benchmark_metrics)
    position_metrics, position_series = _position_metrics(data.positions)
    metrics.update(position_metrics)
    transaction_metrics, transaction_series, round_trips = _transaction_metrics(
        data.transactions, data.positions, float(canonical["total_return"])
    )
    metrics.update(transaction_metrics)
    metrics.update(_round_trip_metrics(round_trips))
    rolling, rolling_series = _rolling(
        returns, data.benchmark_returns, rolling_window
    )
    warnings = []
    if len(returns) < TAIL_WARNING_OBSERVATIONS:
        warnings.append("TAIL_RISK_SAMPLE_WARNING")
    series: dict[str, pd.Series | pd.DataFrame] = {
        "returns": returns,
        "cumulative_returns": (1.0 + returns).cumprod().sub(1.0),
        "drawdown": drawdown,
        "monthly_returns": monthly,
        "annual_returns": annual,
        "round_trips": round_trips,
        **benchmark_series,
        **position_series,
        **transaction_series,
        **rolling_series,
    }
    calendar = {
        "positive_months": int((monthly > 0).sum()),
        "negative_months": int((monthly < 0).sum()),
        "monthly_returns": [
            {"date": date.date().isoformat(), "value": float(value)}
            for date, value in monthly.items()
        ],
        "annual_returns": [
            {"year": int(date.year), "value": float(value)}
            for date, value in annual.items()
        ],
    }
    return PerformanceMetrics(
        data.period,
        "AVAILABLE",
        metrics,
        drawdowns=drawdowns,
        rolling=rolling,
        calendar=calendar,
        warnings=warnings,
        transaction_analysis=data.transaction_status,
        series=series,
    )


__all__ = [
    "ANNUALIZATION_SESSIONS",
    "DEFAULT_ROLLING_WINDOW",
    "METRIC_NAMES",
    "ROLLING_WINDOWS",
    "MetricValue",
    "PerformanceMetrics",
    "calculate_metrics",
]

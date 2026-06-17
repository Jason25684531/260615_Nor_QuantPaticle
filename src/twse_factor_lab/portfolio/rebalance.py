"""Rebalance calendar builder supporting daily / weekly / monthly frequencies."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

RebalanceFrequency = Literal["daily", "weekly", "monthly"]

_SNAPSHOT_FACTORS = frozenset(
    ["pb_inverse", "pe_inverse", "dividend_yield", "latest_snapshot_mixed"]
)


def build_rebalance_calendar(
    trading_days: pd.DatetimeIndex,
    *,
    frequency: RebalanceFrequency = "daily",
    execution_lag_days: int = 1,
) -> pd.DataFrame:
    """Return a DataFrame mapping signal_date → execution_date.

    execution_date is always the next available trading day after signal_date
    based on the actual trading_days index (no synthetic date_range).
    """
    trading_days = pd.DatetimeIndex(sorted(set(trading_days)))
    trading_set = set(trading_days)

    if frequency == "daily":
        signal_dates = list(trading_days[:-execution_lag_days])
    elif frequency == "weekly":
        signal_dates = _first_trading_day_of_week(trading_days)
    elif frequency == "monthly":
        signal_dates = _first_trading_day_of_month(trading_days)
    else:
        raise ValueError(f"Unknown rebalance frequency: {frequency!r}")

    rows = []
    for sig in signal_dates:
        exec_date = _next_trading_day(sig, trading_days, execution_lag_days)
        if exec_date is not None and exec_date in trading_set:
            rows.append(
                {
                    "signal_date": sig,
                    "execution_date": exec_date,
                    "rebalance_frequency": frequency,
                    "execution_lag_days": execution_lag_days,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "execution_date",
                "rebalance_frequency",
                "execution_lag_days",
            ]
        )
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    return df.reset_index(drop=True)


def _first_trading_day_of_week(trading_days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    seen: set[tuple[int, int]] = set()
    result = []
    for day in sorted(trading_days):
        key = (day.isocalendar().year, day.isocalendar().week)
        if key not in seen:
            seen.add(key)
            result.append(day)
    return result


def _first_trading_day_of_month(trading_days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    seen: set[tuple[int, int]] = set()
    result = []
    for day in sorted(trading_days):
        key = (day.year, day.month)
        if key not in seen:
            seen.add(key)
            result.append(day)
    return result


def _next_trading_day(
    reference: pd.Timestamp,
    trading_days: pd.DatetimeIndex,
    lag: int,
) -> pd.Timestamp | None:
    future = [d for d in trading_days if d > reference]
    if len(future) >= lag:
        return future[lag - 1]
    return None


def validate_factor_eligibility(factor_name: str) -> None:
    """Raise ValueError if factor_name is a snapshot-only valuation factor."""
    if factor_name in _SNAPSHOT_FACTORS:
        raise ValueError(
            f"Factor '{factor_name}' is a snapshot valuation factor and cannot be "
            "used in historical backtest. Eligible factors: historical_price_volume "
            "or any historical composite from factor_scoreboard.parquet."
        )


def save_rebalance_calendar(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return output

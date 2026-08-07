"""Equal-weight portfolio target construction."""

from __future__ import annotations

import pandas as pd


def build_equal_weight_portfolio(
    topn_positions: pd.DataFrame,
    *,
    rebalance_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Build equal weights using the authoritative signal-to-execution map."""
    selected = topn_positions[topn_positions["selected"].astype(bool)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "target_weight",
                "execution_date",
                "execution_lag_days",
            ]
        )
    calendar = rebalance_calendar[
        ["signal_date", "execution_date", "execution_lag_days"]
    ].copy()
    calendar["signal_date"] = pd.to_datetime(calendar["signal_date"])
    calendar["execution_date"] = pd.to_datetime(calendar["execution_date"])
    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected.merge(
        calendar,
        left_on="date",
        right_on="signal_date",
        how="inner",
        validate="many_to_one",
    )
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "target_weight",
                "execution_date",
                "execution_lag_days",
            ]
        )
    selected_count = selected.groupby("date")["ticker"].transform("count")
    selected["target_weight"] = 1.0 / selected_count
    return (
        selected[
            [
                "date",
                "ticker",
                "target_weight",
                "execution_date",
                "execution_lag_days",
            ]
        ]
        .sort_values(["execution_date", "ticker"])
        .reset_index(drop=True)
    )

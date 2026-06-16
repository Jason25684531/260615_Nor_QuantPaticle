"""Equal-weight portfolio target construction."""

from __future__ import annotations

import pandas as pd


def build_equal_weight_portfolio(
    topn_positions: pd.DataFrame,
    *,
    execution_lag_days: int = 1,
) -> pd.DataFrame:
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
    selected_count = selected.groupby("date")["ticker"].transform("count")
    selected["target_weight"] = 1.0 / selected_count
    selected["execution_date"] = pd.to_datetime(selected["date"]) + pd.to_timedelta(
        execution_lag_days,
        unit="D",
    )
    selected["execution_lag_days"] = int(execution_lag_days)
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

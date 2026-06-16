"""Top N selection from historical composite factor scores."""

from __future__ import annotations

import pandas as pd


def build_topn_positions(
    factors_composite: pd.DataFrame,
    *,
    top_n: int = 20,
    factor_name: str = "historical_price_volume",
) -> pd.DataFrame:
    historical = factors_composite[
        (factors_composite["composite_type"] == factor_name)
        & (~factors_composite["is_snapshot_component_used"].astype(bool))
    ].copy()
    historical = historical.dropna(subset=["composite_score"])
    historical = historical.rename(columns={"composite_score": "factor_score"})
    historical["rank"] = historical.groupby("date")["factor_score"].rank(
        ascending=False,
        method="first",
    )
    historical["selected"] = historical["rank"] <= int(top_n)
    historical["rank"] = historical["rank"].astype(int)
    historical["top_n"] = int(top_n)
    return (
        historical[["date", "ticker", "factor_score", "rank", "selected", "top_n"]]
        .sort_values(["date", "rank", "ticker"])
        .reset_index(drop=True)
    )

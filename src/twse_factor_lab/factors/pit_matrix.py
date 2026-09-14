"""PIT fundamental factor matrices for Research Cycle v2.

Builds wide (date x ticker) EPS/ROE matrices straight from the point-in-time
fundamental matrix, masked to the liquid PIT-safe universe. Missing values stay
NaN so cross-sectional ranking excludes them (never imputed).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def build_pit_factor_matrix(
    fundamental_matrix: pd.DataFrame,
    metric: str,
    *,
    v2_universe: pd.DataFrame,
    index: pd.DatetimeIndex,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Wide PIT matrix for one metric, masked to universe_included == True."""

    columns = [str(column) for column in columns]
    frame = fundamental_matrix[fundamental_matrix["metric"] == metric].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str)
    frame = frame[frame["value"].notna()]
    if "available_date" in frame.columns:
        available = pd.to_datetime(frame["available_date"], errors="coerce")
        frame = frame[available.isna() | (available <= frame["date"])]
    wide = frame.pivot_table(
        index="date", columns="ticker", values="value", aggfunc="last"
    ).reindex(index=index, columns=columns)

    included = v2_universe[v2_universe["universe_included"].astype(bool)].copy()
    included["date"] = pd.to_datetime(included["date"], errors="coerce")
    included["ticker"] = included["ticker"].astype(str)
    mask = (
        included.assign(_flag=True)
        .pivot_table(index="date", columns="ticker", values="_flag", aggfunc="last")
        .reindex(index=index, columns=columns)
        .fillna(False)
        .astype(bool)
    )
    return wide.where(mask)

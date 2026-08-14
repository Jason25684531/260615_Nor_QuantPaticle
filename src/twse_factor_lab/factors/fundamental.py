"""PIT fundamental factor adapter; input is exclusively D2.5's matrix."""

from __future__ import annotations

import pandas as pd

FUNDAMENTAL_FACTORS = ["revenue_yoy", "eps", "roe", "pe", "pb", "dividend_yield"]


def build_fundamental_factor_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "metric", "value"}
    if missing := required - set(matrix.columns):
        raise KeyError(f"fundamental_matrix missing columns: {sorted(missing)}")
    frame = matrix[matrix["metric"].isin(FUNDAMENTAL_FACTORS)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    values = frame.pivot_table(
        index=["date", "ticker"], columns="metric", values="value", aggfunc="last"
    )
    result = values.reindex(columns=FUNDAMENTAL_FACTORS).reset_index()
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)

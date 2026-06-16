"""Turnover summaries for best-quantile memberships."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor_name, factor_frame in assignments.groupby("factor", sort=True):
        best_quantile = int(factor_frame["quantile"].max())
        best_members = factor_frame[factor_frame["quantile"] == best_quantile]
        membership_by_date = {
            pd.Timestamp(date): set(frame["ticker"])
            for date, frame in best_members.groupby("date", sort=True)
        }
        dates = sorted(membership_by_date)
        turnovers: list[float] = []
        for previous_date, current_date in zip(dates, dates[1:], strict=False):
            previous_members = membership_by_date[previous_date]
            current_members = membership_by_date[current_date]
            if not previous_members:
                continue
            overlap = len(previous_members & current_members)
            turnovers.append(1.0 - (overlap / len(previous_members)))

        rows.append(
            {
                "factor": factor_name,
                "best_quantile": best_quantile,
                "average_best_bucket_turnover": (
                    float(np.mean(turnovers)) if turnovers else np.nan
                ),
                "date_count": len(dates),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "factor",
                "best_quantile",
                "average_best_bucket_turnover",
                "date_count",
            ]
        )
    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)

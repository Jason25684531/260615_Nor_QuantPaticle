"""Cross-sectional IC helpers for historical factor analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_information_coefficients(
    *,
    factor_matrices: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor_name, factor_matrix in factor_matrices.items():
        for horizon, horizon_frame in forward_returns.groupby("horizon", sort=True):
            return_matrix = horizon_frame.pivot(
                index="date", columns="ticker", values="forward_return"
            )
            common_dates = factor_matrix.index.intersection(return_matrix.index)
            common_tickers = factor_matrix.columns.intersection(return_matrix.columns)
            if len(common_dates) == 0 or len(common_tickers) == 0:
                continue

            for date in common_dates:
                factor_row = factor_matrix.loc[date, common_tickers]
                return_row = return_matrix.loc[date, common_tickers]
                aligned = pd.concat(
                    [factor_row.rename("factor"), return_row.rename("forward_return")],
                    axis=1,
                ).dropna()
                if len(aligned) < 2:
                    continue
                ranked = aligned.rank(method="average")
                ic = ranked["factor"].corr(ranked["forward_return"], method="pearson")
                if pd.isna(ic):
                    continue
                rows.append(
                    {
                        "factor": factor_name,
                        "horizon": int(horizon),
                        "date": pd.Timestamp(date),
                        "ic": float(ic),
                        "asset_count": int(len(aligned)),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["factor", "horizon", "date", "ic", "asset_count"])
    return (
        pd.DataFrame(rows)
        .sort_values(["factor", "horizon", "date"])
        .reset_index(drop=True)
    )


def summarize_information_coefficients(ic_results: pd.DataFrame) -> pd.DataFrame:
    if ic_results.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "horizon",
                "ic_mean",
                "ic_std",
                "ir",
                "valid_date_count",
                "average_asset_count",
            ]
        )

    summary = (
        ic_results.groupby(["factor", "horizon"], as_index=False)
        .agg(
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            valid_date_count=("date", "nunique"),
            average_asset_count=("asset_count", "mean"),
        )
        .sort_values(["factor", "horizon"])
        .reset_index(drop=True)
    )
    summary["ic_std"] = summary["ic_std"].fillna(0.0)
    summary["ir"] = summary.apply(
        lambda row: (
            np.nan
            if pd.isna(row["ic_std"]) or row["ic_std"] == 0
            else row["ic_mean"] / row["ic_std"]
        ),
        axis=1,
    )
    return summary

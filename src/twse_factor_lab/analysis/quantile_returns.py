"""Quantile assignment and return aggregation helpers."""

from __future__ import annotations

import pandas as pd


def assign_factor_quantiles(
    *,
    factor_matrices: dict[str, pd.DataFrame],
    directions: dict[str, str],
    quantiles: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor_name, factor_matrix in factor_matrices.items():
        direction = directions[factor_name]
        for date, row in factor_matrix.iterrows():
            clean = row.dropna()
            if clean.empty:
                continue
            oriented = clean if direction == "higher_is_better" else -clean
            ranks = oriented.rank(method="first")
            quantile_series = (((ranks - 1) * quantiles / len(clean)).astype(int)) + 1
            for ticker, factor_value in clean.items():
                rows.append(
                    {
                        "factor": factor_name,
                        "date": pd.Timestamp(date),
                        "ticker": ticker,
                        "factor_value": float(factor_value),
                        "quantile": int(quantile_series.loc[ticker]),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["factor", "date", "ticker", "factor_value", "quantile"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["factor", "date", "ticker"])
        .reset_index(drop=True)
    )


def compute_quantile_returns(
    assignments: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.DataFrame:
    merged = assignments.merge(forward_returns, on=["date", "ticker"], how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["factor", "horizon", "quantile", "mean_return", "member_count"]
        )

    return (
        merged.groupby(["factor", "horizon", "quantile"], as_index=False)
        .agg(
            mean_return=("forward_return", "mean"),
            member_count=("ticker", "count"),
        )
        .sort_values(["factor", "horizon", "quantile"])
        .reset_index(drop=True)
    )

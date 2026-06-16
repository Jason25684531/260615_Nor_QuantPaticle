"""Prepare factor and price structures for Alphalens-style workflows."""

from __future__ import annotations

import pandas as pd


def factor_matrix_to_series(factor_matrix: pd.DataFrame) -> pd.Series:
    series = factor_matrix.stack(future_stack=True)
    series.index = series.index.set_names(["date", "asset"])
    series.name = "factor"
    return series.sort_index()


def validate_alphalens_inputs(
    factor_matrix: pd.DataFrame,
    price_matrix: pd.DataFrame,
) -> dict[str, object]:
    shared_dates = factor_matrix.index.intersection(price_matrix.index)
    shared_assets = factor_matrix.columns.intersection(price_matrix.columns)
    is_ready = (
        len(shared_dates) == len(factor_matrix.index)
        and len(shared_assets) == len(factor_matrix.columns)
        and not factor_matrix.empty
        and not price_matrix.empty
    )
    return {
        "is_ready": is_ready,
        "shared_dates": int(len(shared_dates)),
        "shared_assets": int(len(shared_assets)),
    }

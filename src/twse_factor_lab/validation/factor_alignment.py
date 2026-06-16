"""Alignment validation for factor matrices."""

from __future__ import annotations

import pandas as pd


def validate_factor_matrix(
    factor_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
) -> dict[str, object]:
    duplicate_index = bool(factor_matrix.index.duplicated().any())
    duplicate_columns = bool(factor_matrix.columns.duplicated().any())
    shared_dates = factor_matrix.index.intersection(close_matrix.index)
    shared_tickers = factor_matrix.columns.intersection(close_matrix.columns)

    is_aligned = (
        not duplicate_index
        and not duplicate_columns
        and len(shared_dates) == len(factor_matrix.index)
        and len(shared_tickers) == len(factor_matrix.columns)
    )
    missing_ratio = (
        float(factor_matrix.isna().mean().mean()) if not factor_matrix.empty else 0.0
    )

    return {
        "is_aligned": is_aligned,
        "duplicate_index": duplicate_index,
        "duplicate_columns": duplicate_columns,
        "missing_ratio": missing_ratio,
        "shared_dates": int(len(shared_dates)),
        "shared_tickers": int(len(shared_tickers)),
    }

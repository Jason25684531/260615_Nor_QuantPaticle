"""Helpers that document Week 2 no-lookahead assumptions."""

from __future__ import annotations

import pandas as pd


def uses_backward_looking_windows(
    factor_matrix: pd.DataFrame,
    *,
    lookback: int,
) -> dict[str, object]:
    return {
        "is_backward_looking": lookback > 0 and not factor_matrix.empty,
        "lookback": lookback,
        "rows": int(factor_matrix.shape[0]),
    }


def describe_snapshot_limitation(valuation_frame: pd.DataFrame) -> str:
    if "date" in valuation_frame and valuation_frame["date"].isna().all():
        return (
            "valuation inputs are latest snapshot data because the valuation date "
            "column is empty; they are not historical point-in-time factors."
        )
    return "Valuation inputs include dated observations."

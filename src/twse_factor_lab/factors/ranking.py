"""Cross-sectional ranking helpers for factor matrices."""

from __future__ import annotations

import pandas as pd


def rank_factor(
    factor_matrix: pd.DataFrame,
    *,
    direction: str = "higher_is_better",
) -> pd.DataFrame:
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError(f"Unsupported direction: {direction}")

    ascending = direction == "higher_is_better"
    return factor_matrix.rank(axis=1, pct=True, ascending=ascending)

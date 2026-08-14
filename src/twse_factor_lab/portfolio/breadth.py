"""PARTIAL-UNIVERSE market breadth and the fixed breadth-to-exposure rule.

Breadth answers "how much" (gross exposure), never "who" (stock selection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BREADTH_COLUMNS = [
    "date",
    "eligible_usable_count",
    "above_ma60_count",
    "breadth",
    "universe_status",
]


def compute_market_breadth(
    *,
    close_matrix: pd.DataFrame,
    ma_matrix: pd.DataFrame,
    eligible_matrix: pd.DataFrame,
    universe_status: str,
) -> pd.DataFrame:
    """Breadth_t = (eligible & MA-usable & Close>MA) / (eligible & MA-usable)."""
    eligible_matrix = eligible_matrix.reindex_like(close_matrix).fillna(False)
    usable = eligible_matrix & close_matrix.notna() & ma_matrix.notna()
    above = usable & (close_matrix > ma_matrix)

    denominator = usable.sum(axis=1).astype(float)
    numerator = above.sum(axis=1).astype(float)
    breadth = numerator / denominator.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "date": close_matrix.index,
            "eligible_usable_count": denominator.to_numpy(),
            "above_ma60_count": numerator.to_numpy(),
            "breadth": breadth.to_numpy(),
            "universe_status": universe_status,
        }
    )


def breadth_to_exposure(
    breadth: pd.DataFrame,
    *,
    threshold: float,
    exposure_high: float,
    exposure_low: float,
) -> pd.DataFrame:
    """Fixed step rule: breadth > threshold -> exposure_high, else exposure_low.

    Undefined breadth (empty denominator) carries the last valid observation
    forward; with no prior valid value the conservative low exposure applies.
    """
    result = breadth.sort_values("date").reset_index(drop=True).copy()
    filled = result["breadth"].ffill()
    exposure = filled.gt(threshold).map({True: exposure_high, False: exposure_low})
    result["gross_exposure"] = exposure.where(filled.notna(), exposure_low)
    return result

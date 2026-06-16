"""Forward-return generation for historical factor analysis."""

from __future__ import annotations

import pandas as pd


def build_forward_returns(
    close_matrix: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for horizon in horizons:
        forward = close_matrix.shift(-horizon).div(close_matrix) - 1.0
        long_frame = (
            forward.stack(future_stack=True).rename("forward_return").reset_index()
        )
        long_frame.columns = ["date", "ticker", "forward_return"]
        long_frame["horizon"] = int(horizon)
        frames.append(long_frame.dropna(subset=["forward_return"]))

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "horizon", "forward_return"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["horizon", "date", "ticker"])
        .reset_index(drop=True)
    )

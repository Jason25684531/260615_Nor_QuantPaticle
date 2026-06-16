"""Historical price-volume factor builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_60d(close_matrix: pd.DataFrame) -> pd.DataFrame:
    return close_matrix / close_matrix.shift(60) - 1


def _fallback_return_volatility(close_matrix: pd.DataFrame) -> pd.DataFrame:
    return close_matrix.pct_change().rolling(20, min_periods=20).std()


def low_volatility_method(
    *,
    high_matrix: pd.DataFrame | None = None,
    low_matrix: pd.DataFrame | None = None,
) -> str:
    if high_matrix is None or low_matrix is None:
        return "return_volatility_20d"
    if high_matrix.empty or low_matrix.empty:
        return "return_volatility_20d"
    if high_matrix.isna().all().all() or low_matrix.isna().all().all():
        return "return_volatility_20d"
    return "atr_20d_over_close"


def low_volatility_20d(
    *,
    close_matrix: pd.DataFrame,
    high_matrix: pd.DataFrame | None = None,
    low_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if low_volatility_method(high_matrix=high_matrix, low_matrix=low_matrix) == (
        "return_volatility_20d"
    ):
        return _fallback_return_volatility(close_matrix)

    previous_close = close_matrix.shift(1)
    tr_components = [
        (high_matrix - low_matrix),
        (high_matrix - previous_close).abs(),
        (low_matrix - previous_close).abs(),
    ]
    true_range = pd.DataFrame(
        np.maximum.reduce([component.to_numpy() for component in tr_components]),
        index=close_matrix.index,
        columns=close_matrix.columns,
    )
    atr = true_range.rolling(20, min_periods=20).mean()
    return atr / close_matrix


def volume_ratio_5d_60d(volume_matrix: pd.DataFrame) -> pd.DataFrame:
    short = volume_matrix.rolling(5, min_periods=5).mean()
    long = volume_matrix.rolling(60, min_periods=60).mean()
    return short / long


def build_price_volume_factor_frame(
    *,
    close_matrix: pd.DataFrame,
    high_matrix: pd.DataFrame | None,
    low_matrix: pd.DataFrame | None,
    volume_matrix: pd.DataFrame,
) -> pd.DataFrame:
    factors = {
        "momentum_60d": momentum_60d(close_matrix),
        "low_volatility_20d": low_volatility_20d(
            close_matrix=close_matrix,
            high_matrix=high_matrix,
            low_matrix=low_matrix,
        ),
        "volume_ratio_5d_60d": volume_ratio_5d_60d(volume_matrix),
    }

    stacked_frames: list[pd.DataFrame] = []
    for name, matrix in factors.items():
        long_frame = matrix.stack(future_stack=True).rename(name).reset_index()
        long_frame.columns = ["date", "ticker", name]
        stacked_frames.append(long_frame)

    merged = stacked_frames[0]
    for frame in stacked_frames[1:]:
        merged = merged.merge(frame, on=["date", "ticker"], how="outer")
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)

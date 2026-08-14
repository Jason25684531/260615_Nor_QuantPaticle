"""Historical price-volume factor builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=window).mean()
    std = frame.rolling(window, min_periods=window).std()
    return (
        ((frame - mean) / std).where(std.ne(0), 0.0).replace([np.inf, -np.inf], np.nan)
    )


def rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    result = 100 - 100 / (1 + avg_gain / avg_loss)
    # A flat price has neither gains nor losses: neutral, not an arbitrary extreme.
    return (
        result.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
        .mask(avg_gain.eq(0) & avg_loss.gt(0), 0.0)
        .mask(avg_gain.eq(0) & avg_loss.eq(0), 50.0)
    )


def atr(
    close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, period: int
) -> pd.DataFrame:
    previous = close.shift(1)
    tr = pd.DataFrame(
        np.maximum.reduce(
            [
                (high - low).to_numpy(),
                (high - previous).abs().to_numpy(),
                (low - previous).abs().to_numpy(),
            ]
        ),
        index=close.index,
        columns=close.columns,
    )
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    direction = close.diff().apply(np.sign).fillna(0.0)
    return (direction * volume).cumsum()


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
    config: dict[str, int] | None = None,
) -> pd.DataFrame:
    config = config or {}
    rsi_period, rsi_z_window = (
        int(config.get("rsi_period", 28)),
        int(config.get("rsi_z_window", 90)),
    )
    momentum_period, atr_period = (
        int(config.get("momentum_period", 40)),
        int(config.get("atr_period", 14)),
    )
    obv_z_window, ma_short, ma_long = (
        int(config.get("obv_z_window", 90)),
        int(config.get("ma_short", 60)),
        int(config.get("ma_long", 120)),
    )
    rsi_28 = rsi(close_matrix, rsi_period)
    atr_14 = (
        atr(close_matrix, high_matrix, low_matrix, atr_period)
        if high_matrix is not None and low_matrix is not None
        else pd.DataFrame(
            np.nan, index=close_matrix.index, columns=close_matrix.columns
        )
    )
    natr_14 = (atr_14 / close_matrix).replace([np.inf, -np.inf], np.nan)
    momentum_40d = close_matrix / close_matrix.shift(momentum_period) - 1
    obv_values = obv(close_matrix, volume_matrix)
    factors = {
        "momentum_60d": momentum_60d(close_matrix),
        "low_volatility_20d": low_volatility_20d(
            close_matrix=close_matrix,
            high_matrix=high_matrix,
            low_matrix=low_matrix,
        ),
        "volume_ratio_5d_60d": volume_ratio_5d_60d(volume_matrix),
        "rsi_28": rsi_28,
        "rsi_z": _zscore(rsi_28, rsi_z_window),
        "momentum_40d": momentum_40d,
        "atr_14": atr_14,
        "natr_14": natr_14,
        "risk_adjusted_momentum": (momentum_40d / natr_14)
        .where(natr_14.ne(0))
        .replace([np.inf, -np.inf], np.nan),
        "obv": obv_values,
        "obv_z90": _zscore(obv_values, obv_z_window),
        "ma_60": close_matrix.rolling(ma_short, min_periods=ma_short).mean(),
        "ma_120": close_matrix.rolling(ma_long, min_periods=ma_long).mean(),
    }
    factors["trend_signal"] = (close_matrix > factors["ma_60"]) & (
        close_matrix > factors["ma_120"]
    )

    stacked_frames: list[pd.DataFrame] = []
    for name, matrix in factors.items():
        long_frame = matrix.stack(future_stack=True).rename(name).reset_index()
        long_frame.columns = ["date", "ticker", name]
        stacked_frames.append(long_frame)

    merged = stacked_frames[0]
    for frame in stacked_frames[1:]:
        merged = merged.merge(frame, on=["date", "ticker"], how="outer")
    return (
        merged.sort_values(["date", "ticker"])
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )

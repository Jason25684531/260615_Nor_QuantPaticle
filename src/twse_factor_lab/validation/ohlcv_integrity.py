"""Validation for the long-format research OHLCV contract."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}
FLOAT_TOLERANCE = 1e-10


def validate_ohlcv(frame: pd.DataFrame) -> None:
    """Raise when complete OHLC rows violate their market-data relationships."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise KeyError(f"Missing required OHLCV columns: {sorted(missing)}")
    if frame.duplicated(["date", "ticker"], keep=False).any():
        raise ValueError("Found duplicate date+ticker rows in OHLCV input")

    valid = frame.dropna(subset=["open", "high", "low", "close"])
    checks = {
        "close <= 0": valid["close"] <= 0,
        "high < low": valid["high"] + FLOAT_TOLERANCE < valid["low"],
        "high < open": valid["high"] + FLOAT_TOLERANCE < valid["open"],
        "high < close": valid["high"] + FLOAT_TOLERANCE < valid["close"],
        "low > close": valid["low"] - FLOAT_TOLERANCE > valid["close"],
        "low > open": valid["low"] - FLOAT_TOLERANCE > valid["open"],
    }
    for description, failed in checks.items():
        if failed.any():
            raise ValueError(f"Invalid OHLCV row: {description}")


def sort_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the deterministic date/ticker order without changing values."""

    return frame.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

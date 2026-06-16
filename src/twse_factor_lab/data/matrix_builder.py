"""Helpers for building wide OHLCV matrices from Week 1 long-format data."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def _validate_input_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")


def _ensure_no_duplicates(frame: pd.DataFrame) -> None:
    duplicated = frame.duplicated(subset=["date", "ticker"], keep=False)
    if duplicated.any():
        raise ValueError("Found duplicate date+ticker rows in OHLCV input")


def build_field_matrix(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot a single OHLCV field into a date-by-ticker matrix."""

    _validate_input_columns(frame, {"date", "ticker", field})
    prepared = frame[["date", "ticker", field]].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["ticker"] = prepared["ticker"].astype(str)
    _ensure_no_duplicates(prepared)

    matrix = prepared.pivot(index="date", columns="ticker", values=field)
    matrix = matrix.sort_index().sort_index(axis=1)
    matrix.index.name = "date"
    matrix.columns.name = "ticker"
    return matrix


def build_ohlcv_matrices(frame: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Build all supported OHLCV matrices from Week 1 long-format data."""

    fields = {
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }
    return {name: build_field_matrix(frame, source) for name, source in fields.items()}

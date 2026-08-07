"""Validation for configured in-sample and out-of-sample periods."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _period(config: dict[str, Any], name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        period = config[name]
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid research.{name} period") from error
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError(f"Invalid research.{name} period")
    return start, end


def validate_research_periods(
    research_config: dict[str, Any],
    data_start: str | pd.Timestamp,
    data_end: str | pd.Timestamp,
) -> None:
    """Require ordered, non-overlapping IS/OOS periods within available data."""

    in_start, in_end = _period(research_config, "in_sample")
    out_start, out_end = _period(research_config, "out_of_sample")
    available_start, available_end = pd.Timestamp(data_start), pd.Timestamp(data_end)
    if available_start > available_end:
        raise ValueError("Invalid available data range")
    if in_end >= out_start:
        raise ValueError("In-sample and out-of-sample periods overlap")
    if in_start < available_start or out_end > available_end:
        raise ValueError("Research periods fall outside the available data range")

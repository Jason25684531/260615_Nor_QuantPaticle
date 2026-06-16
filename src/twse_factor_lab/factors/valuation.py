"""Snapshot-safe valuation factor builders."""

from __future__ import annotations

import pandas as pd


def build_snapshot_valuation_factors(
    valuation: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    factors = valuation.copy()
    factors["as_of_date"] = pd.Timestamp(as_of_date)
    factors["pb_inverse"] = factors["pb"].where(factors["pb"] > 0).rdiv(1.0)
    factors["pe_inverse"] = factors["pe"].where(factors["pe"] > 0).rdiv(1.0)
    factors["is_snapshot"] = True
    return factors[
        [
            "date",
            "as_of_date",
            "ticker",
            "pb_inverse",
            "pe_inverse",
            "dividend_yield",
            "is_snapshot",
        ]
    ].reset_index(drop=True)

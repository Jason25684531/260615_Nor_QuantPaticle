from __future__ import annotations

import pandas as pd


def test_attribution_contains_security_year_and_rebalance_rows() -> None:
    frame = pd.read_csv(
        "data/research/fundamental-production-final-v1/historical-evidence/attribution.csv"
    )
    assert {"SECURITY", "YEAR", "REBALANCE"}.issubset(
        set(frame["attribution_type"])
    )
    assert frame["contribution"].notna().all()


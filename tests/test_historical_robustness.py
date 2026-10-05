from __future__ import annotations

import pandas as pd


def test_annual_and_sensitivity_outputs_are_present_and_diagnostic() -> None:
    root = "data/research/fundamental-production-final-v1/historical-evidence/"
    annual = pd.read_csv(root + "annual_performance.csv")
    sensitivity = pd.read_csv(root + "sensitivity.csv")
    assert len(annual) >= 4
    assert {2021, 2022, 2023, 2024, 2025}.issubset(set(annual["year"]))
    assert sensitivity["diagnostic_only"].astype(bool).all()
    assert "FROZEN_TOP5_REB60_SCORE" in set(sensitivity["scenario"])


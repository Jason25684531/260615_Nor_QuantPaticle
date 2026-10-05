from __future__ import annotations

import pandas as pd


def test_quantile_analysis_has_all_factors_and_groups() -> None:
    frame = pd.read_csv(
        "data/research/fundamental-production-final-v1/historical-evidence/factor_quantile_validation.csv"
    )
    assert set(frame["factor"]) == {"G2", "G3", "COMPOSITE"}
    assert set(frame["quantile"]) == {1, 2, 3, 4, 5}
    assert frame["mean_forward_return"].notna().any()


"""Pairwise correlations of already normalized factor exposures."""

from __future__ import annotations

import pandas as pd


def compute_factor_correlation(
    matrices: dict[str, pd.DataFrame], minimum_samples: int
) -> pd.DataFrame:
    rows = []
    names = sorted(matrices)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            paired = pd.concat(
                [matrices[first].stack(), matrices[second].stack()], axis=1
            ).dropna()
            count = len(paired)
            rows.append(
                {
                    "factor_a": first,
                    "factor_b": second,
                    "correlation": paired.iloc[:, 0].corr(paired.iloc[:, 1])
                    if count >= minimum_samples
                    else float("nan"),
                    "sample_count": count,
                    "status": "KNOWN" if count >= minimum_samples else "UNKNOWN",
                }
            )
    return pd.DataFrame(
        rows, columns=["factor_a", "factor_b", "correlation", "sample_count", "status"]
    )

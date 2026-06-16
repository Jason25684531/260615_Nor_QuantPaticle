"""Monotonicity evaluation for quantile-return summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_monotonicity(quantile_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = quantile_returns.groupby(["factor", "horizon"], sort=True)
    for (factor_name, horizon), frame in grouped:
        ordered = frame.sort_values("quantile")
        returns = ordered["mean_return"].tolist()
        if len(returns) < 2:
            ratio = np.nan
            monotonicity_pass = False
            notes = "insufficient quantile buckets"
            spread = np.nan
        else:
            comparisons = [
                current >= previous
                for previous, current in zip(returns, returns[1:], strict=False)
            ]
            ratio = float(sum(comparisons) / len(comparisons))
            monotonicity_pass = ratio == 1.0
            notes = (
                "monotonic increasing from Q1 to Q5"
                if monotonicity_pass
                else "non-monotonic quantile-return ordering"
            )
            spread = float(returns[-1] - returns[0])

        rows.append(
            {
                "factor": factor_name,
                "horizon": int(horizon),
                "monotonicity_pass": monotonicity_pass,
                "adjacent_agreement_ratio": ratio,
                "top_bottom_spread": spread,
                "notes": notes,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "factor",
                "horizon",
                "monotonicity_pass",
                "adjacent_agreement_ratio",
                "top_bottom_spread",
                "notes",
            ]
        )
    return pd.DataFrame(rows).sort_values(["factor", "horizon"]).reset_index(drop=True)

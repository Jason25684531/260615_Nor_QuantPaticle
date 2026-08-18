from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from scipy.stats import kurtosis, skew


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    nobs: int,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    if nobs <= 1:
        raise ValueError("nobs must exceed one")
    denominator = math.sqrt(
        1
        - skewness * observed_sharpe
        + ((kurtosis_excess + 3) - 1) * observed_sharpe**2 / 4
    )
    return NormalDist().cdf(
        (observed_sharpe - benchmark_sharpe) * math.sqrt(nobs - 1) / denominator
    )


def daily_sharpe(returns) -> float:
    values = np.asarray(returns, dtype=float)
    return float(values.mean() / values.std(ddof=1)) if values.std(ddof=1) else 0.0


def deflated_sharpe_ratio(
    observed_sharpe: float, nobs: int, trial_sharpes
) -> tuple[float, float]:
    values = np.asarray(list(trial_sharpes), dtype=float)
    if not len(values):
        raise ValueError("trial inventory is required for DSR")
    n = len(values)
    variance = float(np.var(values, ddof=1)) if n > 1 else 0.0
    gamma = 0.5772156649015329
    sr0 = math.sqrt(variance) * (
        (1 - gamma) * NormalDist().inv_cdf(1 - 1 / n)
        + gamma * NormalDist().inv_cdf(1 - 1 / (n * math.e))
    )
    return probabilistic_sharpe_ratio(observed_sharpe, nobs, sr0), sr0


def moments(returns) -> tuple[float, float]:
    values = np.asarray(returns, dtype=float)
    return float(skew(values, bias=False)), float(
        kurtosis(values, fisher=False, bias=False)
    )

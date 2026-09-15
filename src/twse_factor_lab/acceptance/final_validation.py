"""Small, deterministic final-validation policy primitives."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

LOCKED_FINGERPRINT = "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"
BLOCK_LENGTH = 20
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
STATISTICAL_THRESHOLD = 0.95


class FinalValidationError(ValueError):
    """A frozen final-validation contract was violated."""


def sha256_json(value: Mapping[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode()).hexdigest()


def validate_locked_candidate(candidate: Mapping[str, Any]) -> None:
    if candidate.get("candidate_fingerprint") != LOCKED_FINGERPRINT:
        raise FinalValidationError("FINAL_VALIDATION_CONTAMINATED")
    locked = candidate.get("candidate", candidate)
    expected = {
        "strategy_id": "S3",
        "components": ["L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D"],
        "weights": {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5},
        "top_n": 5,
        "rebalance": "monthly",
        "buffer": False,
        "weighting": "equal_weight",
        "cost_model": "base_cost",
    }
    if any(locked.get(key) != value for key, value in expected.items()):
        raise FinalValidationError("FINAL_VALIDATION_CONTAMINATED")


def freeze_three_folds(
    index: pd.DatetimeIndex, warmup_sessions: int = 252
) -> list[dict[str, Any]]:
    dates = pd.DatetimeIndex(index).sort_values().unique()
    eligible = dates[warmup_sessions:]
    if len(eligible) < 3:
        raise FinalValidationError("insufficient validation sessions")
    blocks = np.array_split(eligible, 3)
    folds = []
    for number, block in enumerate(blocks, start=1):
        train = dates[dates < block[0]]
        folds.append(
            {
                "fold": number,
                "training_start": train[0].date().isoformat() if len(train) else None,
                "training_end": train[-1].date().isoformat() if len(train) else None,
                "validation_start": block[0].date().isoformat(),
                "validation_end": block[-1].date().isoformat(),
                "validation_sessions": int(len(block)),
                "expanding_window": True,
            }
        )
    return folds


def _drawdown(values: np.ndarray) -> float:
    equity = np.cumprod(1.0 + values)
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def moving_block_bootstrap(
    returns: pd.Series,
    *,
    block_length: int = BLOCK_LENGTH,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fixed moving-block bootstrap; no parameter search."""
    if (
        block_length != BLOCK_LENGTH
        or samples != BOOTSTRAP_SAMPLES
        or seed != BOOTSTRAP_SEED
    ):
        raise FinalValidationError("FINAL_VALIDATION_CONTAMINATED:bootstrap_parameters")
    values = pd.Series(returns, dtype=float).dropna().to_numpy()
    if len(values) <= block_length:
        raise FinalValidationError("bootstrap requires more than one block")
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0,
        len(values) - block_length + 1,
        size=(samples, math.ceil(len(values) / block_length)),
    )
    paths = np.concatenate(
        [values[start : start + block_length] for start in starts.ravel()]
    ).reshape(samples, -1)[:, : len(values)]
    sharpes = np.mean(paths, axis=1) / np.std(paths, axis=1, ddof=0) * math.sqrt(252)
    cagr = np.power(np.prod(1.0 + paths, axis=1), 252 / len(values)) - 1.0
    mdd = np.array([_drawdown(path) for path in paths])
    return {
        "method": "moving_block_bootstrap",
        "samples": samples,
        "block_length": block_length,
        "random_seed": seed,
        "sharpe_distribution": sharpes.tolist(),
        "cagr_distribution": cagr.tolist(),
        "mdd_distribution": mdd.tolist(),
        "median_sharpe": float(np.median(sharpes)),
        "p05_sharpe": float(np.quantile(sharpes, 0.05)),
        "p95_sharpe": float(np.quantile(sharpes, 0.95)),
        "p_sharpe_positive": float(np.mean(sharpes > 0)),
        "p_cagr_positive": float(np.mean(cagr > 0)),
        "status": (
            "PASS"
            if np.mean(sharpes > 0) >= 0.80 and np.mean(cagr > 0) >= 0.80
            else "FAIL"
            if np.mean(sharpes > 0) < 0.60 or np.mean(cagr > 0) < 0.60
            else "MIXED"
        ),
    }


def temporal_status(rows: Sequence[Mapping[str, Any]], aggregate_return: float) -> str:
    positive = sum(
        float(row["total_return"]) > 0 and float(row["sharpe"]) > 0 for row in rows
    )
    return "PASS" if positive >= 2 and aggregate_return > 0 else "FAIL"


def final_verdict(gates: Mapping[str, str], *, lock_success: bool = True) -> str:
    critical = (
        "temporal_validation",
        "cost_stress",
        "statistical",
        "fingerprint",
        "reproducibility",
    )
    if (
        not lock_success
        or any(gates.get(name) == "FAIL" for name in critical)
        or gates.get("engine_parity") == "FAIL"
    ):
        return "REJECT"
    if (
        all(
            gates.get(name) == "PASS"
            for name in (
                "temporal_validation",
                "bootstrap",
                "cost_stress",
                "statistical",
                "reproducibility",
            )
        )
        and gates.get("engine_parity") == "PASS"
    ):
        return "ACCEPT"
    return "CANDIDATE"


__all__ = [
    "BLOCK_LENGTH",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "FinalValidationError",
    "LOCKED_FINGERPRINT",
    "STATISTICAL_THRESHOLD",
    "final_verdict",
    "freeze_three_folds",
    "moving_block_bootstrap",
    "sha256_json",
    "temporal_status",
    "validate_locked_candidate",
]

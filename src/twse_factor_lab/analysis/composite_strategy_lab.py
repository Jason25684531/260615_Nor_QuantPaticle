"""Governed primitives for Composite Strategy Lab v1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

COMPONENTS = ("L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D")
WEIGHTS = {component: 0.5 for component in COMPONENTS}
TOP_N = 5
BUFFER_HOLD_RANK = 7
BASE_COST = {
    "buy_fee_rate": 0.001425,
    "sell_fee_rate": 0.001425,
    "transaction_tax_rate": 0.003,
    "slippage_rate": 0.001,
}


class CompositeStrategyLabError(ValueError):
    """A contract or candidate-lock invariant was violated."""


def sha256_json(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible content."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def strategy_configs() -> list[dict[str, Any]]:
    """The complete, immutable four-member selection population."""
    return [
        {
            "strategy_id": f"S{number}",
            "rebalance": rebalance,
            "buffer": buffer_on,
            "top_n": TOP_N,
            "weighting": "equal_weight",
            "cost_model": "base_cost",
            "selection_relevant": True,
            "terminal_trial": True,
            "trial_stage": "strategy_selection",
        }
        for number, (rebalance, buffer_on) in enumerate(
            (
                ("weekly", False),
                ("weekly", True),
                ("monthly", False),
                ("monthly", True),
            ),
            start=1,
        )
    ]


def validate_selection_config(config: Mapping[str, Any]) -> None:
    """Reject values outside the predeclared four-cell selection space."""
    allowed = {
        (
            row["rebalance"],
            row["buffer"],
            row["top_n"],
            row["weighting"],
            row["cost_model"],
        )
        for row in strategy_configs()
    }
    observed = (
        config.get("rebalance"),
        config.get("buffer"),
        config.get("top_n"),
        config.get("weighting"),
        config.get("cost_model"),
    )
    if observed not in allowed:
        raise CompositeStrategyLabError(
            f"undeclared strategy configuration: {observed}"
        )


def candidate_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return precisely the values protected by candidate lock."""
    return {
        "strategy_id": row["strategy_id"],
        "components": list(COMPONENTS),
        "weights": dict(WEIGHTS),
        "top_n": TOP_N,
        "rebalance": row["rebalance"],
        "buffer": bool(row["buffer"]),
        "buffer_hold_rank": BUFFER_HOLD_RANK if row["buffer"] else 0,
        "weighting": "equal_weight",
        "cost_model": "base_cost",
        "cost_parameters": dict(BASE_COST),
    }


def _number(row: Mapping[str, Any], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompositeStrategyLabError(f"missing numeric metric: {name}") from exc
    if not math.isfinite(value):
        raise CompositeStrategyLabError(f"non-finite metric: {name}")
    return value


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reuse v2 lock ordering and reject a non-positive best candidate."""
    if len(rows) != 4:
        raise CompositeStrategyLabError(
            "candidate selection requires exactly four trials"
        )
    for row in rows:
        validate_selection_config(row)
        if not row.get("selection_relevant") or not row.get("terminal_trial"):
            raise CompositeStrategyLabError(
                "all selection trials must be terminal/relevant"
            )
    ordered = sorted(
        rows,
        key=lambda row: (
            -_number(row, "net_sharpe"),
            _number(row, "turnover"),
            _number(row, "max_drawdown"),
            str(row["strategy_id"]),
        ),
    )
    best = dict(ordered[0])
    if _number(best, "net_sharpe") <= 0 or _number(best, "net_cagr") <= 0:
        return {
            "status": "NO_POSITIVE_CANDIDATE",
            "strategy_id": None,
            "candidate": None,
            "candidate_fingerprint": None,
            "selection_rule": "base_cost_net_sharpe_then_turnover_mdd_lexical",
        }
    candidate = candidate_payload(best)
    return {
        "status": "SUCCESS",
        "strategy_id": best["strategy_id"],
        "candidate": candidate,
        "candidate_fingerprint": sha256_json(candidate),
        "selection_rule": "base_cost_net_sharpe_then_turnover_mdd_lexical",
    }


def verify_locked_candidate(
    lock: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    """Fail if any candidate-lock field has changed after the lock."""
    if lock.get("status") != "SUCCESS":
        raise CompositeStrategyLabError("a non-successful lock cannot be mutated")
    if sha256_json(candidate) != lock.get("candidate_fingerprint"):
        raise CompositeStrategyLabError("POST_LOCK_MUTATION")


def robust_plateau(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the required positive neighbour condition over the 2×2 surface."""
    if len(rows) != 4:
        raise CompositeStrategyLabError("robust plateau requires exactly four trials")
    best = max(rows, key=lambda row: _number(row, "net_sharpe"))
    neighbours = [row for row in rows if row["strategy_id"] != best["strategy_id"]]
    qualifying = [
        row
        for row in neighbours
        if _number(row, "net_sharpe") > 0
        and _number(row, "net_cagr") > 0
        and math.isfinite(_number(row, "max_drawdown"))
    ]
    status = "PASS" if qualifying else "FAIL"
    return {
        "best_config": best["strategy_id"],
        "neighbor_configs": [row["strategy_id"] for row in neighbours],
        "qualifying_neighbors": [row["strategy_id"] for row in qualifying],
        "robust_plateau_status": status,
        "rule": "one_non_best_positive_sharpe_positive_cagr_finite_mdd",
    }


def readiness(lock: Mapping[str, Any], plateau: Mapping[str, Any]) -> tuple[bool, str]:
    """Derive Change 3 eligibility; it is never a strategy acceptance verdict."""
    if lock.get("status") != "SUCCESS":
        return False, "no positive locked candidate"
    if plateau.get("robust_plateau_status") == "FAIL":
        return False, "robust plateau failed"
    return True, "positive locked candidate and non-failing robust plateau"


__all__ = [
    "BASE_COST",
    "BUFFER_HOLD_RANK",
    "COMPONENTS",
    "CompositeStrategyLabError",
    "TOP_N",
    "WEIGHTS",
    "candidate_payload",
    "readiness",
    "robust_plateau",
    "select_candidate",
    "sha256_json",
    "strategy_configs",
    "validate_selection_config",
    "verify_locked_candidate",
]

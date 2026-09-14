"""Expanding-window walk-forward validation for Research Cycle v2.

Processed data ends 2025-12-31 and 2024-2025 was already observed by the v1
cycle, so no untouched 2026+ holdout exists. This is therefore honest
WALK-FORWARD VALIDATION, never a fresh untouched OOS. Each fold re-executes the
locked candidate from fresh cash / empty positions through run_fresh_oos;
pre-test history only feeds lookback / PIT factor formation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from twse_factor_lab.acceptance.research_cycle import _write_json, run_fresh_oos
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.factors.registry import FactorRegistry
from twse_factor_lab.strategy.lab import StrategyDefinition

VALIDATION_LABEL = "WALK-FORWARD VALIDATION"
WALK_FORWARD_METHOD = "walk-forward-expanding-v1"

# Pre-declared expanding folds; 2018-2021 warmup feeds only lookback/PIT.
DEFAULT_FOLDS: tuple[tuple[str, str], ...] = (
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
)


def run_walk_forward(
    *,
    root: str | Path,
    research_id: str,
    base_strategy_id: str,
    definition: StrategyDefinition,
    factor_matrices: Mapping[str, pd.DataFrame],
    close_matrix: pd.DataFrame,
    admission_results: Mapping[str, Any],
    cost_model: CostModel,
    folds: Sequence[tuple[str, str]] = DEFAULT_FOLDS,
    registry: FactorRegistry | None = None,
) -> dict[str, Any]:
    """Run one fresh-state fold per test window and label results honestly."""
    root = Path(root)
    fold_reports: list[dict[str, Any]] = []
    returns_parts: list[pd.Series] = []
    for start, end in folds:
        year = pd.Timestamp(start).year
        strategy_id = f"{base_strategy_id}-wf-{year}"
        result = run_fresh_oos(
            root=root,
            research_id=research_id,
            strategy_id=strategy_id,
            experiment_id=f"walk-forward-{strategy_id}",
            definition=definition,
            factor_matrices=factor_matrices,
            close_matrix=close_matrix,
            admission_results=admission_results,
            cost_model=cost_model,
            oos_start=start,
            oos_end=end,
            registry=registry,
        )
        fold_reports.append(
            {
                "fold": year,
                "train_end": start,
                "test_start": result["oos_start"],
                "test_end": result["oos_end"],
                "warmup_start": result["warmup_start"],
                "fresh_state": result["fresh_state"],
                "inherited_is_state": result["inherited_is_state"],
                "metrics": result["oos_metrics"],
            }
        )
        part = pd.Series(
            {row["date"]: row["value"] for row in result["oos_returns"]}, dtype=float
        )
        returns_parts.append(part)

    combined = (
        pd.concat(returns_parts) if returns_parts else pd.Series(dtype=float)
    )
    if not combined.empty:
        combined.index = pd.to_datetime(combined.index)
        combined = combined[~combined.index.duplicated()].sort_index()

    payload = {
        "label": VALIDATION_LABEL,
        "method": WALK_FORWARD_METHOD,
        "reason": (
            "processed data ends 2025-12-31 and 2024-2025 was already observed "
            "by multifactor-validation-historical-v1; no untouched 2026+ holdout "
            "exists, so this is walk-forward validation, not a fresh OOS"
        ),
        "warmup_note": (
            "pre-test history feeds only lookback / PIT factor formation; every "
            "fold runs fresh cash, empty positions, and zero inherited P&L"
        ),
        "folds": fold_reports,
    }
    path = _write_json(
        root,
        root / "data" / "research" / research_id / "validation" / "walk_forward.json",
        payload,
    )
    payload["artifact_path"] = path
    return {"payload": payload, "combined_returns": combined}


__all__ = [
    "DEFAULT_FOLDS",
    "VALIDATION_LABEL",
    "WALK_FORWARD_METHOD",
    "run_walk_forward",
]

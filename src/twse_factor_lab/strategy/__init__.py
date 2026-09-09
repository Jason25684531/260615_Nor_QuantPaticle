"""Governed strategy-composition research for new research cycles."""

from twse_factor_lab.strategy.handoff import (
    HandoffValidationError,
    StrategyHandoff,
    load_strategy_handoff,
)
from twse_factor_lab.strategy.lab import (
    StrategyDefinition,
    StrategyLabError,
    StrategyRegistry,
    StrategyTrialResult,
    compare_incremental_contribution,
    run_strategy_trial,
)

__all__ = [
    "StrategyDefinition",
    "StrategyLabError",
    "StrategyRegistry",
    "StrategyTrialResult",
    "compare_incremental_contribution",
    "run_strategy_trial",
    "HandoffValidationError",
    "StrategyHandoff",
    "load_strategy_handoff",
]

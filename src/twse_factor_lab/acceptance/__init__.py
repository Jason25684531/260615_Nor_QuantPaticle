"""Acceptance statistics plus isolated new-cycle OOS/freeze primitives."""

from .frozen import load_inputs
from .psr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from .research_cycle import (
    ACCEPTANCE_SECTIONS,
    ACCEPTANCE_VERSION,
    FREEZE_VERSION,
    METHOD_VERSION,
    ResearchCycleError,
    assert_no_lookahead,
    build_acceptance_matrix,
    build_research_trial_inventory,
    compute_statistical_acceptance,
    evaluate_oos,
    freeze_research_cycle,
    run_oos_evaluation,
    verify_research_freeze,
)

__all__ = [
    "ACCEPTANCE_SECTIONS",
    "ACCEPTANCE_VERSION",
    "FREEZE_VERSION",
    "METHOD_VERSION",
    "ResearchCycleError",
    "assert_no_lookahead",
    "build_acceptance_matrix",
    "build_research_trial_inventory",
    "compute_statistical_acceptance",
    "deflated_sharpe_ratio",
    "evaluate_oos",
    "freeze_research_cycle",
    "load_inputs",
    "probabilistic_sharpe_ratio",
    "run_oos_evaluation",
    "verify_research_freeze",
]

"""Backtest engines and diagnostic comparison helpers."""

from twse_factor_lab.backtest.backtrader_engine import (
    BacktraderExecutionError,
    BacktraderRun,
    BacktraderUnavailableError,
    run_backtrader_engine,
)
from twse_factor_lab.backtest.tri_engine import (
    PARITY_TOLERANCE,
    TriEngineValidationError,
    compare_engine_results,
    run_tri_engine_validation,
)

__all__ = [
    "BacktraderExecutionError",
    "BacktraderRun",
    "BacktraderUnavailableError",
    "PARITY_TOLERANCE",
    "TriEngineValidationError",
    "compare_engine_results",
    "run_backtrader_engine",
    "run_tri_engine_validation",
]

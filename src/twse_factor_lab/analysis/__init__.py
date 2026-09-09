"""Historical factor analysis utilities."""

from twse_factor_lab.analysis.attribution import (
    AttributionError,
    AttributionResult,
    DescriptorBundle,
    ExposureResult,
    NeutralizationResult,
    attribute_portfolio_returns,
    compare_factor_evidence,
    compute_portfolio_exposures,
    neutralize_factor,
    run_attribution_diagnostic,
    standardize_descriptor,
    standardize_descriptors,
)
from twse_factor_lab.analysis.factor_gate import (
    FactorDiagnostics,
    FactorGateConfig,
    FactorGateError,
    evaluate_factor,
)
from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.analysis.information_coefficient import (
    compute_information_coefficients,
    summarize_information_coefficients,
)
from twse_factor_lab.analysis.monotonicity import evaluate_monotonicity
from twse_factor_lab.analysis.performance import (
    PerformanceDiagnosticError,
    evaluate_performance,
    run_performance_diagnostic,
    write_performance_report,
)
from twse_factor_lab.analysis.preparation import (
    FACTOR_DIRECTIONS,
    select_historical_factor_matrices,
)
from twse_factor_lab.analysis.quantile_returns import (
    assign_factor_quantiles,
    compute_quantile_returns,
)
from twse_factor_lab.analysis.research_robustness import (
    RobustnessError,
    ablate_factor,
    build_predeclared_grid,
    classify_robustness,
    config_fingerprint,
    run_factor_ablation,
    run_robustness_sweep,
)
from twse_factor_lab.analysis.turnover import compute_turnover_summary

__all__ = [
    "FACTOR_DIRECTIONS",
    "AttributionError",
    "AttributionResult",
    "DescriptorBundle",
    "ExposureResult",
    "FactorDiagnostics",
    "FactorGateConfig",
    "FactorGateError",
    "NeutralizationResult",
    "assign_factor_quantiles",
    "build_forward_returns",
    "compute_information_coefficients",
    "compute_quantile_returns",
    "compute_turnover_summary",
    "evaluate_monotonicity",
    "evaluate_factor",
    "attribute_portfolio_returns",
    "compare_factor_evidence",
    "compute_portfolio_exposures",
    "neutralize_factor",
    "PerformanceDiagnosticError",
    "evaluate_performance",
    "run_performance_diagnostic",
    "run_attribution_diagnostic",
    "standardize_descriptor",
    "standardize_descriptors",
    "RobustnessError",
    "ablate_factor",
    "build_predeclared_grid",
    "classify_robustness",
    "config_fingerprint",
    "run_factor_ablation",
    "run_robustness_sweep",
    "select_historical_factor_matrices",
    "summarize_information_coefficients",
    "write_performance_report",
]

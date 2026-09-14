"""Phase 7 tests: performance metric audit trail and reconstructability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twse_factor_lab.analysis.performance import (
    CANONICAL_METRIC_METADATA,
    PerformanceDiagnosticError,
    canonical_metric_metadata,
    evaluate_performance,
)
from twse_factor_lab.backtest.robustness import compute_metrics

TOL = 1e-9


def _fixture_returns() -> pd.Series:
    index = pd.bdate_range("2021-01-01", periods=8)
    values = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.03, -0.015]
    return pd.Series(values, index=index, dtype=float)


def test_downside_deviation_matches_manual_and_is_deterministic() -> None:
    returns = _fixture_returns()
    first = compute_metrics(returns)
    second = compute_metrics(returns)
    assert first["downside_deviation"] == second["downside_deviation"]

    manual = float(returns.clip(upper=0).std(ddof=0) * np.sqrt(252))
    assert abs(first["downside_deviation"] - manual) <= TOL


def test_sortino_reconstructable_from_components() -> None:
    metrics = compute_metrics(_fixture_returns())
    rebuilt = metrics["cagr"] / metrics["downside_deviation"]
    assert abs(rebuilt - metrics["sortino"]) <= 1e-9


def test_sharpe_deterministic_fixture() -> None:
    returns = _fixture_returns()
    metrics = compute_metrics(returns)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = (1 + total_return) ** (252 / len(returns)) - 1
    vol = float(returns.std(ddof=0) * np.sqrt(252))
    assert abs(metrics["sharpe"] - cagr / vol) <= 1e-9


def test_mdd_and_calmar_reconstructable() -> None:
    returns = _fixture_returns()
    metrics = compute_metrics(returns)
    equity = (1.0 + returns).cumprod()
    manual_mdd = float(equity.div(equity.cummax()).sub(1.0).min())
    assert abs(metrics["max_drawdown"] - manual_mdd) <= 1e-9
    rebuilt_calmar = metrics["cagr"] / abs(metrics["max_drawdown"])
    assert abs(rebuilt_calmar - metrics["calmar"]) <= 1e-9


def test_ddof_zero_behaviour() -> None:
    returns = _fixture_returns()
    metrics = compute_metrics(returns)
    ddof0 = float(returns.std(ddof=0) * np.sqrt(252))
    ddof1 = float(returns.std(ddof=1) * np.sqrt(252))
    assert abs(metrics["volatility"] - ddof0) <= 1e-12
    assert abs(metrics["volatility"] - ddof1) > 1e-9


def test_metadata_matches_implementation() -> None:
    meta = canonical_metric_metadata()
    assert meta is not CANONICAL_METRIC_METADATA  # defensive copy
    assert meta["annualization_factor"] == 252
    assert meta["volatility_ddof"] == 0
    assert meta["risk_free_rate"] == 0.0
    assert meta["sortino_target"] == 0.0
    assert meta["return_convention"] == "simple"
    assert meta["metric_schema_version"] == "canonical-metrics-v2"


def test_nan_policy_rejects_nonfinite_input() -> None:
    index = pd.bdate_range("2021-01-01", periods=3)
    bad = pd.Series([0.01, np.nan, 0.02], index=index)
    with pytest.raises(PerformanceDiagnosticError):
        evaluate_performance(bad)


def test_evaluate_performance_exposes_metric_definition() -> None:
    report = evaluate_performance(_fixture_returns())
    definition = report["metric_definition"]
    assert definition["metric_schema_version"] == "canonical-metrics-v2"
    assert definition["canonical_metric_source"] == (
        "backtest.robustness.compute_metrics"
    )
    # downside_deviation is carried verbatim from canonical_metrics.
    assert (
        definition["downside_deviation"]
        == report["canonical_metrics"]["downside_deviation"]
    )
    # Pyfolio Sharpe/Sortino stay definition-aware (never FAIL on definition gap).
    for check in report["cross_check"]["checks"]:
        if check["metric"] in {"sharpe", "sortino", "volatility"}:
            assert check["status"] in {"DEFINITION_DIFFERENCE", "UNKNOWN"}

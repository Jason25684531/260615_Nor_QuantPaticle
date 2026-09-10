"""Canonical performance metrics and independent Pyfolio diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.backtest.vectorbt_engine import _weights_matrix
from twse_factor_lab.governance import (
    ExperimentRecord,
    load_research_manifest,
    register_experiment,
    update_experiment_status,
)
from twse_factor_lab.governance.isolation import assert_write_allowed
from twse_factor_lab.strategy.handoff import StrategyHandoff, load_strategy_handoff

PERFORMANCE_TOLERANCE = 1e-8


class PerformanceDiagnosticError(ValueError):
    """Performance inputs or the installed Pyfolio layer are invalid."""


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _validate_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    values = pd.to_numeric(returns, errors="coerce").astype(float)
    if not isinstance(values.index, pd.DatetimeIndex):
        values.index = pd.DatetimeIndex(pd.to_datetime(values.index, errors="coerce"))
    if values.empty:
        raise PerformanceDiagnosticError("returns must not be empty")
    if values.index.isna().any() or values.index.has_duplicates:
        raise PerformanceDiagnosticError("returns index must be valid and unique")
    if not values.index.is_monotonic_increasing:
        raise PerformanceDiagnosticError("returns index must be sorted")
    if not np.isfinite(values.to_numpy()).all():
        raise PerformanceDiagnosticError("returns must contain finite values")
    values.name = "returns"
    return values


def _validated_positions(
    positions: pd.DataFrame | None, index: pd.DatetimeIndex
) -> pd.DataFrame | None:
    if positions is None:
        return None
    if not isinstance(positions, pd.DataFrame):
        raise PerformanceDiagnosticError("positions must be a DataFrame")
    result = positions.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="coerce"))
    if result.index.isna().any() or result.index.has_duplicates:
        raise PerformanceDiagnosticError("positions index must be valid and unique")
    if not result.index.equals(index):
        raise PerformanceDiagnosticError("positions dates must align with returns")
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise PerformanceDiagnosticError("positions must contain finite values")
    return result.astype(float)


def _normalise_pyfolio_metrics(raw: pd.Series) -> dict[str, Any]:
    mapping = {
        "Annual return": "cagr",
        "Cumulative returns": "total_return",
        "Annual volatility": "volatility",
        "Sharpe ratio": "sharpe",
        "Sortino ratio": "sortino",
        "Max drawdown": "max_drawdown",
        "Calmar ratio": "calmar",
        "Stability": "stability",
        "Omega ratio": "omega_ratio",
        "Gross leverage": "gross_leverage",
    }
    result: dict[str, Any] = {}
    for source, target in mapping.items():
        if source in raw:
            value = raw[source]
            result[target] = float(value) if pd.notna(value) else None
    result["raw"] = {
        str(key): (float(value) if pd.notna(value) else None)
        for key, value in raw.items()
    }
    return result


CANONICAL_DEFINITION = {
    "annualization": "252 trading sessions/year, geometric compounding",
    "risk_free_rate": 0.0,
    "nan_handling": "NaN filled with 0.0 before compounding",
    "return_convention": "simple daily returns",
}

# Per-metric comparability: metrics with `comparable=True` use the same
# formula in both layers (modulo floating-point noise), so a numeric
# tolerance genuinely detects parity breaks. Metrics with `comparable=False`
# are computed by fundamentally different formulas (documented in `reason`),
# so a numeric gap there is an expected definitional difference, not a
# parity failure, and must never be reported as FAIL.
METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "total_return": {
        "comparable": True,
        "tolerance": 1e-8,
        "reason": "identical cumulative-compounding formula",
    },
    "cagr": {
        "comparable": True,
        "tolerance": 1e-8,
        "reason": "identical 252-session geometric annualization formula",
    },
    "max_drawdown": {
        "comparable": True,
        "tolerance": 1e-8,
        "reason": "identical peak-to-trough equity-curve formula",
    },
    "volatility": {
        "comparable": False,
        "tolerance": None,
        "reason": (
            "canonical uses population std (ddof=0); pyfolio's ddof "
            "convention is package-defined and may differ"
        ),
    },
    "sharpe": {
        "comparable": False,
        "tolerance": None,
        "reason": (
            "canonical: CAGR / annualized volatility (geometric ratio); "
            "pyfolio: mean daily return / daily std * sqrt(252) "
            "(classic arithmetic ratio) — different formulas, both rf=0"
        ),
    },
    "sortino": {
        "comparable": False,
        "tolerance": None,
        "reason": (
            "canonical: CAGR / annualized downside semi-deviation "
            "(geometric ratio); pyfolio: mean daily return / downside "
            "deviation * sqrt(252) (classic arithmetic ratio)"
        ),
    },
}

PYFOLIO_DEFINITION = {
    "annualization": "pyfolio-reloaded perf_stats defaults (package-defined)",
    "risk_free_rate": 0.0,
    "nan_handling": "package-defined",
    "return_convention": "simple daily returns",
}


def _cross_check(
    canonical: dict[str, Any], pyfolio: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for metric, definition in METRIC_DEFINITIONS.items():
        left = canonical.get(metric)
        right = pyfolio.get(metric)
        if left is None or right is None:
            checks.append(
                {
                    "metric": metric,
                    "status": "UNKNOWN",
                    "canonical": left,
                    "pyfolio": right,
                    "comparable": definition["comparable"],
                    "reason": definition["reason"],
                }
            )
            continue
        difference = abs(float(left) - float(right))
        metric_tolerance = definition["tolerance"] if definition["comparable"] else None
        status = (
            ("PASS" if difference <= (metric_tolerance or tolerance) else "FAIL")
            if definition["comparable"]
            else "DEFINITION_DIFFERENCE"
        )
        checks.append(
            {
                "metric": metric,
                "canonical": float(left),
                "pyfolio": float(right),
                "difference": difference,
                "tolerance": metric_tolerance,
                "comparable": definition["comparable"],
                "reason": definition["reason"],
                "status": status,
            }
        )
    if any(row["status"] == "FAIL" for row in checks):
        overall_status = "FAIL"
    elif any(row["status"] == "DEFINITION_DIFFERENCE" for row in checks):
        overall_status = "PASS_WITH_DEFINITION_DIFFERENCE"
    else:
        overall_status = "PASS"
    return {
        "tolerance": tolerance,
        "status": overall_status,
        "checks": checks,
        "definition_notes": {
            "canonical": CANONICAL_DEFINITION,
            "pyfolio": PYFOLIO_DEFINITION,
        },
    }


def evaluate_performance(
    returns: pd.Series,
    *,
    positions: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
    tolerance: float = PERFORMANCE_TOLERANCE,
) -> dict[str, Any]:
    """Evaluate a validated return series using canonical metrics and Pyfolio."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise PerformanceDiagnosticError("tolerance must be finite and non-negative")
    values = _validate_returns(returns)
    positions = _validated_positions(positions, values.index)
    if turnover is not None:
        turnover = pd.Series(turnover, index=values.index, dtype=float)
    if exposure is not None:
        exposure = pd.Series(exposure, index=values.index, dtype=float)
    canonical = compute_metrics(values, turnover=turnover, exposure=exposure)
    canonical.update(
        {
            "observation_count": int(values.size),
            "start_date": values.index.min().date().isoformat(),
            "end_date": values.index.max().date().isoformat(),
            "annualized_return": canonical["cagr"],
            "annualized_volatility": canonical["volatility"],
        }
    )
    try:
        import pyfolio.timeseries as timeseries

        raw = timeseries.perf_stats(
            values,
            positions=positions,
            transactions=None,
        )
    except Exception as exc:
        raise PerformanceDiagnosticError(str(exc)) from exc
    pyfolio_metrics = _normalise_pyfolio_metrics(raw)
    return {
        "status": "completed",
        "canonical_metrics": _json_value(canonical),
        "pyfolio_metrics": _json_value(pyfolio_metrics),
        "cross_check": _json_value(
            _cross_check(canonical, pyfolio_metrics, tolerance)
        ),
        "transactions": {
            "available": False,
            "status": "UNAVAILABLE",
            "reason": "Day 4 handoff contains no true transaction stream",
        },
        "transaction_dependent_diagnostics": {
            "status": "UNAVAILABLE",
            "reason": "true transactions are unavailable; no position-diff fabrication",
        },
        "positions": {
            "available": positions is not None,
            "contract": (
                "dollar_positions_including_cash" if positions is not None else None
            ),
        },
    }


def _handoff_turnover_and_exposure(
    handoff: StrategyHandoff,
) -> tuple[pd.Series | None, pd.Series | None]:
    positions = handoff.positions.copy()
    position_columns = [column for column in positions.columns if column != "cash"]
    exposure = (
        positions[position_columns].sum(axis=1).div(handoff.nav).rename("exposure")
        if position_columns
        else pd.Series(0.0, index=handoff.nav.index, name="exposure")
    )
    if handoff.target_weights.empty:
        return None, exposure
    pivot = _weights_matrix(
        handoff.target_weights,
        handoff.returns.index,
        pd.Index(position_columns),
    )
    return pivot.diff().fillna(pivot).abs().sum(axis=1), exposure


def _write_json(root: Path, path: Path, payload: dict[str, Any]) -> None:
    target = assert_write_allowed(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_performance_report(
    *,
    root: str | Path,
    handoff: StrategyHandoff,
    experiment_id: str,
    report: dict[str, Any],
) -> dict[str, str]:
    root = Path(root)
    directory = (
        root
        / "data"
        / "research"
        / handoff.research_id
        / "performance"
        / handoff.strategy_id
    )
    _write_json(
        root, directory / "canonical_metrics.json", report["canonical_metrics"]
    )
    _write_json(root, directory / "pyfolio_metrics.json", report["pyfolio_metrics"])
    metadata = {
        "research_id": handoff.research_id,
        "strategy_id": handoff.strategy_id,
        "source_experiment_id": handoff.experiment_id,
        "diagnostic_experiment_id": experiment_id,
        "dataset_version": handoff.dataset_version,
        "handoff_dir": str(handoff.path.relative_to(root)),
        "status": report["status"],
        "transactions": report["transactions"],
        "transaction_dependent_diagnostics": report[
            "transaction_dependent_diagnostics"
        ],
        "positions": report["positions"],
        "cross_check": report["cross_check"],
        "risk_free_rate": 0.0,
        "annualization_sessions": 252,
        "limitation": (
            "INFRASTRUCTURE VALIDATION ONLY; no real Day 3 admission artifacts"
        ),
    }
    _write_json(root, directory / "pyfolio_metadata.json", metadata)
    return {
        "directory": str(directory.relative_to(root)),
        "canonical_metrics": str(
            (directory / "canonical_metrics.json").relative_to(root)
        ),
        "pyfolio_metrics": str(
            (directory / "pyfolio_metrics.json").relative_to(root)
        ),
        "pyfolio_metadata": str(
            (directory / "pyfolio_metadata.json").relative_to(root)
        ),
    }


def run_performance_diagnostic(
    *,
    root: str | Path,
    research_id: str,
    strategy_id: str,
    handoff_dir: str | Path,
    experiment_id: str,
    tolerance: float = PERFORMANCE_TOLERANCE,
) -> dict[str, Any]:
    """Register one diagnostic, then validate and report a Day 4 handoff."""
    root = Path(root)
    research = load_research_manifest(root, research_id)
    register_experiment(
        ExperimentRecord(
            experiment_id=experiment_id,
            research_id=research_id,
            config={
                "diagnostic": "canonical_performance_pyfolio",
                "strategy_id": strategy_id,
                "handoff_dir": str(handoff_dir),
                "tolerance": tolerance,
                "selection_relevant": False,
            },
            dataset_version=research.dataset_version,
            experiment_type="diagnostic",
            status="running",
            selection_relevant=False,
        ),
        root,
    )
    try:
        handoff = load_strategy_handoff(
            root,
            handoff_dir,
            expected_research_id=research_id,
            expected_dataset_version=research.dataset_version,
        )
        if handoff.strategy_id != strategy_id:
            raise PerformanceDiagnosticError("handoff strategy_id mismatch")
        turnover, exposure = _handoff_turnover_and_exposure(handoff)
        report = evaluate_performance(
            handoff.returns,
            positions=handoff.positions,
            turnover=turnover,
            exposure=exposure,
            tolerance=tolerance,
        )
        report["research_id"] = research_id
        report["strategy_id"] = strategy_id
        report["dataset_version"] = research.dataset_version
        report["artifact_paths"] = write_performance_report(
            root=root,
            handoff=handoff,
            experiment_id=experiment_id,
            report=report,
        )
        update_experiment_status(root, research_id, experiment_id, "completed", report)
        return report
    except Exception as exc:
        update_experiment_status(
            root,
            research_id,
            experiment_id,
            "failed",
            {"status": "failed", "error": str(exc)},
        )
        raise

"""Governed Custom/Vectorbt/Backtrader parity for new-cycle research."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.backtrader_engine import run_backtrader_engine
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.governance import (
    ExperimentRecord,
    load_research_manifest,
    register_experiment,
    update_experiment_status,
)
from twse_factor_lab.governance.isolation import assert_write_allowed
from twse_factor_lab.strategy.handoff import load_strategy_handoff

PARITY_TOLERANCE = 1e-8


class TriEngineValidationError(RuntimeError):
    """A tri-engine validation input or execution failed."""


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _first_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    tolerance: float,
) -> dict[str, Any] | None:
    left_dates = pd.DatetimeIndex(pd.to_datetime(left["date"]))
    right_dates = pd.DatetimeIndex(pd.to_datetime(right["date"]))
    if not left_dates.equals(right_dates):
        length = min(len(left_dates), len(right_dates))
        date = (
            left_dates[length]
            if len(left_dates) > length
            else right_dates[length]
            if len(right_dates) > length
            else next(
                (
                    left_dates[index]
                    for index in range(length)
                    if left_dates[index] != right_dates[index]
                ),
                None,
            )
        )
        return {
            "date": date.date().isoformat() if date is not None else None,
            "metric": "date_index",
            "left": len(left_dates),
            "right": len(right_dates),
            "absolute_difference": None,
        }
    if metric not in left or metric not in right:
        return None
    left_values = pd.to_numeric(left[metric], errors="coerce").to_numpy(float)
    right_values = pd.to_numeric(right[metric], errors="coerce").to_numpy(float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    differences = np.abs(left_values - right_values)
    bad = valid & (differences > tolerance)
    if not bad.any():
        return None
    index = int(np.flatnonzero(bad)[0])
    return {
        "date": left_dates[index].date().isoformat(),
        "metric": metric,
        "left": float(left_values[index]),
        "right": float(right_values[index]),
        "absolute_difference": float(differences[index]),
    }


def compare_engine_results(
    engine_results: Mapping[str, pd.DataFrame],
    *,
    actual_engines: Mapping[str, str] | None = None,
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Compare all available result fields against the actual Custom leg."""
    expected = ("custom", "vectorbt", "backtrader")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise TriEngineValidationError("tolerance must be finite and non-negative")
    missing = [name for name in expected if name not in engine_results]
    if missing:
        raise TriEngineValidationError(
            f"missing tri-engine result: {', '.join(missing)}"
        )
    engines = dict(actual_engines or {name: name for name in expected})
    result: dict[str, Any] = {
        "status": "PASS",
        "tolerance": tolerance,
        "engines": engines,
        "comparisons": [],
        "first_divergence": None,
    }
    if engines.get("custom") != "custom":
        result["status"] = "FAIL"
        result["first_divergence"] = {
            "date": None,
            "metric": "custom_engine",
            "left": engines.get("custom"),
            "right": "custom",
            "absolute_difference": None,
        }
        return result
    for name in ("vectorbt", "backtrader"):
        if engines.get(name) != name:
            result["status"] = "FAIL"
            result["first_divergence"] = {
                "date": None,
                "metric": f"{name}_engine",
                "left": engines.get(name),
                "right": name,
                "absolute_difference": None,
            }
            return result
    metrics = sorted(
        set(engine_results["custom"].columns)
        .intersection(engine_results["vectorbt"].columns)
        .intersection(engine_results["backtrader"].columns)
        .difference({"date"})
    )
    for name in ("vectorbt", "backtrader"):
        for metric in metrics:
            divergence = _first_difference(
                engine_results["custom"],
                engine_results[name],
                metric,
                tolerance,
            )
            result["comparisons"].append(
                {
                    "engine": name,
                    "metric": metric,
                    "status": "FAIL" if divergence else "PASS",
                }
            )
            if divergence and result["first_divergence"] is None:
                divergence["engine"] = name
                result["first_divergence"] = divergence
    if result["first_divergence"] is not None:
        result["status"] = "FAIL"
    return result


def _write_validation_artifacts(
    root: Path,
    research_id: str,
    strategy_id: str,
    experiment_id: str,
    result: dict[str, Any],
    fills: pd.DataFrame,
) -> str:
    directory = (
        root
        / "data"
        / "research"
        / research_id
        / "execution_validation"
        / strategy_id
        / experiment_id
    )
    json_path = assert_write_allowed(directory / "parity.json", root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_finite_json(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fills_path = assert_write_allowed(directory / "backtrader_fills.csv", root)
    fills.to_csv(fills_path, index=False, lineterminator="\n")
    return str(directory.relative_to(root))


def run_tri_engine_validation(
    *,
    root: str | Path,
    research_id: str,
    strategy_id: str,
    handoff_dir: str | Path,
    experiment_id: str,
    close_matrix: pd.DataFrame,
    cost_model: CostModel,
    initial_cash: float = 1_000_000,
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Run exactly one diagnostic experiment over three actual engines."""
    root = Path(root)
    research = load_research_manifest(root, research_id)
    record = ExperimentRecord(
        experiment_id=experiment_id,
        research_id=research_id,
        config={
            "diagnostic": "tri_engine_execution_validation",
            "strategy_id": strategy_id,
            "handoff_dir": str(handoff_dir),
            "engines": ["custom", "vectorbt", "backtrader"],
            "tolerance": tolerance,
            "selection_relevant": False,
        },
        dataset_version=research.dataset_version,
        experiment_type="diagnostic",
        status="running",
        selection_relevant=False,
    )
    register_experiment(record, root)
    try:
        handoff = load_strategy_handoff(
            root,
            handoff_dir,
            expected_research_id=research_id,
            expected_dataset_version=research.dataset_version,
        )
        if handoff.strategy_id != strategy_id:
            raise TriEngineValidationError("handoff strategy_id mismatch")
        if handoff.target_weights.empty:
            raise TriEngineValidationError(
                "handoff lacks explicit target-weight events for Backtrader"
            )
        targets = handoff.target_weights
        custom_results, custom_metrics = run_weight_backtest(
            close_matrix=close_matrix,
            portfolio_weights=targets,
            cost_model=cost_model,
            initial_cash=initial_cash,
            top_n=int(handoff.metadata.get("top_n", 1)),
            use_vectorbt=False,
            allow_fallback=False,
        )
        vectorbt_results, vectorbt_metrics = run_weight_backtest(
            close_matrix=close_matrix,
            portfolio_weights=targets,
            cost_model=cost_model,
            initial_cash=initial_cash,
            top_n=int(handoff.metadata.get("top_n", 1)),
            use_vectorbt=True,
            allow_fallback=False,
        )
        backtrader_run = run_backtrader_engine(
            close_matrix=close_matrix,
            target_weights=targets,
            cost_model=cost_model,
            initial_cash=initial_cash,
        )
        actual_engines = {
            "custom": str(custom_metrics.iloc[0]["actual_engine"]),
            "vectorbt": str(vectorbt_metrics.iloc[0]["actual_engine"]),
            "backtrader": str(backtrader_run.metrics.iloc[0]["actual_engine"]),
        }
        result = compare_engine_results(
            {
                "custom": custom_results,
                "vectorbt": vectorbt_results,
                "backtrader": backtrader_run.results,
            },
            actual_engines=actual_engines,
            tolerance=tolerance,
        )
        result.update(
            {
                "research_id": research_id,
                "strategy_id": strategy_id,
                "dataset_version": research.dataset_version,
                "handoff_dir": str(handoff.path.relative_to(root)),
                "execution_semantics_version": handoff.metadata[
                    "execution_semantics_version"
                ],
                "execution_events": [
                    {
                        "signal_date": row.signal_date.date().isoformat(),
                        "execution_date": row.execution_date.date().isoformat(),
                        "ticker": str(row.ticker),
                        "target_weight": float(row.target_weight),
                    }
                    for row in handoff.target_weights.itertuples(index=False)
                ],
                "execution_artifact_dir": _write_validation_artifacts(
                    root,
                    research_id,
                    strategy_id,
                    experiment_id,
                    result,
                    backtrader_run.fills,
                ),
            }
        )
        registry_status = "completed" if result["status"] == "PASS" else "failed"
        update_experiment_status(
            root, research_id, experiment_id, registry_status, result
        )
        return result
    except Exception as exc:
        update_experiment_status(
            root,
            research_id,
            experiment_id,
            "failed",
            {"status": "failed", "error": str(exc)},
        )
        raise

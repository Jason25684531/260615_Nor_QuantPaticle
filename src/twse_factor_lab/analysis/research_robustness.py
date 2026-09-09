"""Manifest-bounded robustness and ablation diagnostics for new research."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from twse_factor_lab.governance import (
    ExperimentRecord,
    load_research_manifest,
    register_experiment,
    update_experiment_status,
)
from twse_factor_lab.governance.isolation import assert_write_allowed

ROBUSTNESS_RESULTS = frozenset({"ROBUST", "FRAGILE", "MIXED", "INSUFFICIENT"})
CANONICAL_METRICS = ("total_return", "sharpe", "max_drawdown", "turnover", "exposure")
GRID_KEYS = ("top_n", "rebalance_frequency", "cost_scenario")
METHOD_VERSION = "research-robustness-mvp-v1"


class RobustnessError(ValueError):
    """Invalid or undeclared robustness input."""


def _json_value(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


def config_fingerprint(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(config).encode()).hexdigest()


def _declared_values(research: Any, key: str) -> tuple[Any, ...]:
    attribute = {
        "top_n": "top_n_search_space",
        "rebalance_frequency": "rebalance_search_space",
        "cost_scenario": "cost_scenarios",
    }[key]
    return tuple(getattr(research, attribute))


def build_predeclared_grid(
    research: Any,
    grid: Mapping[str, Sequence[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expand only values declared by a Research Manifest, in stable order."""
    source = grid or {key: _declared_values(research, key) for key in GRID_KEYS}
    unknown = set(source) - set(GRID_KEYS)
    if unknown:
        raise RobustnessError(f"unknown robustness grid keys: {sorted(unknown)}")
    normalized: dict[str, tuple[Any, ...]] = {}
    for key in GRID_KEYS:
        if key not in source:
            continue
        values = tuple(source[key])
        if not values:
            raise RobustnessError(f"robustness grid {key!r} must not be empty")
        declared = _declared_values(research, key)
        undeclared = [value for value in values if value not in declared]
        if undeclared:
            raise RobustnessError(
                f"robustness grid {key!r} contains undeclared values: {undeclared!r}"
            )
        normalized[key] = tuple(dict.fromkeys(values))
    keys = tuple(key for key in GRID_KEYS if key in normalized)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(normalized[key] for key in keys))
    ]


def _metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = result.get("canonical_metrics")
    return nested if isinstance(nested, Mapping) else result


def classify_robustness(
    scenarios: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None,
    *,
    stable_sharpe_ratio: float = 0.70,
    fragile_sharpe_ratio: float = 0.30,
    stable_drawdown_multiple: float = 1.50,
) -> str:
    """Classify stability; this function never selects a new candidate."""
    completed = [row for row in scenarios if row.get("status") == "completed"]
    if not completed or baseline is None:
        return "INSUFFICIENT"
    base = _metrics(baseline)
    try:
        base_sharpe = float(base["sharpe"])
        base_drawdown = abs(float(base["max_drawdown"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RobustnessError("baseline requires sharpe and max_drawdown") from exc
    stable = 0
    fragile = 0
    for row in completed:
        metrics = _metrics(row.get("metrics", row))
        try:
            sharpe = float(metrics["sharpe"])
            drawdown = abs(float(metrics["max_drawdown"]))
        except (KeyError, TypeError, ValueError):
            continue
        same_sign = math.copysign(1.0, sharpe) == math.copysign(1.0, base_sharpe)
        ratio = sharpe / base_sharpe if base_sharpe else (1.0 if sharpe >= 0 else -1.0)
        dd_ok = (
            drawdown <= stable_drawdown_multiple * base_drawdown
            if base_drawdown
            else drawdown == 0
        )
        if same_sign and ratio >= stable_sharpe_ratio and dd_ok:
            stable += 1
        elif (not same_sign) or ratio < fragile_sharpe_ratio:
            fragile += 1
    if fragile:
        return "FRAGILE" if stable == 0 else "MIXED"
    return "ROBUST" if stable == len(completed) else "MIXED"


def _write_json(root: Path, path: Path, payload: Any) -> str:
    root = root.resolve()
    target = assert_write_allowed(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return str(target.relative_to(root))


def run_robustness_sweep(
    *,
    root: str | Path,
    research_id: str,
    strategy_id: str,
    experiment_id: str,
    candidate_config: Mapping[str, Any],
    runner: Callable[[dict[str, Any]], Mapping[str, Any]],
    grid: Mapping[str, Sequence[Any]] | None = None,
    baseline_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a pre-declared grid around a fixed candidate and retain failures."""
    root = Path(root)
    research = load_research_manifest(root, research_id)
    declared_grid = build_predeclared_grid(research, grid)
    candidate = deepcopy(dict(candidate_config))
    register_experiment(
        ExperimentRecord(
            experiment_id=experiment_id,
            research_id=research_id,
            config={
                "experiment_type": "robustness_diagnostic",
                "strategy_id": strategy_id,
                "candidate_config": candidate,
                "candidate_fingerprint": config_fingerprint(candidate),
                "grid": declared_grid,
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
        scenarios: list[dict[str, Any]] = []
        candidate_keys = {key: candidate[key] for key in GRID_KEYS if key in candidate}
        for number, perturbation in enumerate(declared_grid, start=1):
            run_config = deepcopy(candidate)
            run_config.update(perturbation)
            row: dict[str, Any] = {
                "scenario_id": f"scenario-{number:03d}",
                "config": run_config,
                "config_fingerprint": config_fingerprint(run_config),
                "status": "completed",
            }
            try:
                row["metrics"] = dict(runner(deepcopy(run_config)))
            except Exception as exc:
                row["status"] = "failed"
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc)
            scenarios.append(row)
        if baseline_result is None:
            for row in scenarios:
                if all(
                    row["config"].get(key) == value
                    for key, value in candidate_keys.items()
                ):
                    baseline_result = row.get("metrics")
                    break
        result = {
            "strategy_id": strategy_id,
            "research_id": research_id,
            "dataset_version": research.dataset_version,
            "candidate_config": candidate,
            "candidate_fingerprint": config_fingerprint(candidate),
            "grid": declared_grid,
            "scenarios": scenarios,
            "robustness_result": classify_robustness(scenarios, baseline_result),
            "selection_relevant": False,
            "method": METHOD_VERSION,
        }
        directory = (
            root / "data" / "research" / research_id / "robustness" / strategy_id
        )
        result["artifact_path"] = _write_json(
            root, directory / f"{experiment_id}.json", result
        )
        update_experiment_status(root, research_id, experiment_id, "completed", result)
        return result
    except Exception as exc:
        update_experiment_status(
            root,
            research_id,
            experiment_id,
            "failed",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def ablate_factor(
    candidate_config: Mapping[str, Any], factor_id: str
) -> dict[str, Any]:
    """Remove exactly one factor and renormalize the remaining declared weights."""
    candidate = deepcopy(dict(candidate_config))
    factor_ids = list(candidate.get("factor_ids", []))
    if factor_id not in factor_ids:
        raise RobustnessError(f"factor_id is not in candidate: {factor_id!r}")
    factor_ids.remove(factor_id)
    if not factor_ids:
        raise RobustnessError("ablation cannot remove the only factor")
    candidate["factor_ids"] = factor_ids
    weights = dict(candidate.get("factor_weights", {}))
    weights.pop(factor_id, None)
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise RobustnessError("remaining factor weights must have positive sum")
    candidate["factor_weights"] = {
        key: float(weights[key]) / total for key in sorted(weights)
    }
    for key in ("factor_directions", "factor_families", "factor_pit_required"):
        if isinstance(candidate.get(key), Mapping):
            candidate[key] = {
                key_: value
                for key_, value in candidate[key].items()
                if key_ != factor_id
            }
    return candidate


def _metric_deltas(
    baseline: Mapping[str, Any], ablated: Mapping[str, Any]
) -> dict[str, float | None]:
    left, right = _metrics(baseline), _metrics(ablated)
    return {
        metric: (
            None
            if left.get(metric) is None or right.get(metric) is None
            else float(right[metric]) - float(left[metric])
        )
        for metric in CANONICAL_METRICS
    }


def run_factor_ablation(
    *,
    root: str | Path,
    research_id: str,
    strategy_id: str,
    experiment_id: str,
    candidate_config: Mapping[str, Any],
    factor_id: str,
    baseline_result: Mapping[str, Any],
    runner: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one diagnostic ablation; the original candidate is never mutated."""
    root = Path(root)
    research = load_research_manifest(root, research_id)
    ablated_config = ablate_factor(candidate_config, factor_id)
    register_experiment(
        ExperimentRecord(
            experiment_id=experiment_id,
            research_id=research_id,
            config={
                "experiment_type": "ablation_diagnostic",
                "strategy_id": strategy_id,
                "removed_factor": factor_id,
                "candidate_fingerprint": config_fingerprint(candidate_config),
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
        ablated_result = dict(runner(deepcopy(ablated_config)))
        result = {
            "strategy_id": strategy_id,
            "research_id": research_id,
            "dataset_version": research.dataset_version,
            "candidate_config": deepcopy(dict(candidate_config)),
            "ablated_config": ablated_config,
            "removed_factor": factor_id,
            "metrics": ablated_result,
            "deltas": _metric_deltas(baseline_result, ablated_result),
            "selection_relevant": False,
            "method": METHOD_VERSION,
        }
        directory = root / "data" / "research" / research_id / "ablation" / strategy_id
        result["artifact_path"] = _write_json(
            root, directory / f"{experiment_id}.json", result
        )
        update_experiment_status(root, research_id, experiment_id, "completed", result)
        return result
    except Exception as exc:
        update_experiment_status(
            root,
            research_id,
            experiment_id,
            "failed",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


__all__ = [
    "CANONICAL_METRICS",
    "GRID_KEYS",
    "METHOD_VERSION",
    "ROBUSTNESS_RESULTS",
    "RobustnessError",
    "ablate_factor",
    "build_predeclared_grid",
    "classify_robustness",
    "config_fingerprint",
    "run_factor_ablation",
    "run_robustness_sweep",
]

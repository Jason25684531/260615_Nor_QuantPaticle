"""Minimal, governed multi-factor Strategy Lab.

This module owns composition orchestration only.  Ranking, portfolio
construction, execution, costs, and the Pyfolio input contract remain in the
existing repository modules.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.analysis.pyfolio_adapter import to_pyfolio_inputs
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.factors.composer import compose_factor_scores
from twse_factor_lab.factors.registry import FactorRegistry, build_default_registry
from twse_factor_lab.governance import (
    ExperimentRecord,
    load_research_manifest,
    register_experiment,
    update_experiment_status,
)
from twse_factor_lab.governance.isolation import assert_write_allowed
from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar
from twse_factor_lab.portfolio.selection import build_topn_positions
from twse_factor_lab.portfolio.weights import build_equal_weight_portfolio

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ENGINES = frozenset({"custom", "vectorbt"})
_NORMALIZATIONS = frozenset({"percentile_rank"})
_CANONICAL_METRICS = (
    "total_return",
    "sharpe",
    "max_drawdown",
    "turnover",
    "exposure",
)


class StrategyLabError(ValueError):
    """Invalid strategy definition, admission input, or trial execution."""


def _require_slug(value: str, name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise StrategyLabError(f"{name} must be a non-empty filesystem-safe slug")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyLabError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise StrategyLabError(f"{name} must be finite")
    return number


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _admission_verdict(value: Any) -> str:
    if isinstance(value, str):
        verdict = value
    elif isinstance(value, Mapping):
        verdict = value.get("verdict")
    else:
        verdict = getattr(value, "verdict", None)
    if not isinstance(verdict, str) or verdict.upper() not in {
        "ACCEPT",
        "CANDIDATE",
        "REJECT",
    }:
        raise StrategyLabError(
            "each factor needs an explicit ACCEPT/CANDIDATE/REJECT admission"
        )
    return verdict.upper()


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    research_id: str
    factor_ids: tuple[str, ...]
    factor_weights: dict[str, float]
    top_n: int
    rebalance_frequency: str
    buffer_enabled: bool
    drop_rank_buffer: int
    cost_scenario: str
    universe: str
    dataset_version: str
    execution_engine: str = "custom"
    normalization_method: str = "percentile_rank"
    selection_relevant: bool = True
    execution_semantics_version: str = "signal_t_next_trading_day_v1"
    factor_directions: dict[str, str] | None = None
    allow_engine_fallback: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_ids", tuple(self.factor_ids))
        object.__setattr__(self, "factor_weights", dict(self.factor_weights))
        if self.factor_directions is not None:
            object.__setattr__(self, "factor_directions", dict(self.factor_directions))

    def validate_basic(self) -> None:
        _require_slug(self.strategy_id, "strategy_id")
        _require_slug(self.research_id, "research_id")
        if not self.factor_ids or len(set(self.factor_ids)) != len(self.factor_ids):
            raise StrategyLabError("factor_ids must be non-empty and unique")
        if any(
            not isinstance(factor_id, str) or not factor_id
            for factor_id in self.factor_ids
        ):
            raise StrategyLabError("factor_ids must be non-empty strings")
        if set(self.factor_weights) != set(self.factor_ids):
            raise StrategyLabError("factor_weights must exactly match factor_ids")
        values = {
            factor_id: _finite(weight, f"factor_weights[{factor_id!r}]")
            for factor_id, weight in self.factor_weights.items()
        }
        if any(weight < 0 for weight in values.values()):
            raise StrategyLabError("factor weights must be non-negative")
        if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise StrategyLabError("factor weights must sum to one")
        if (
            isinstance(self.top_n, bool)
            or not isinstance(self.top_n, int)
            or self.top_n <= 0
        ):
            raise StrategyLabError("top_n must be a positive integer")
        if not isinstance(self.buffer_enabled, bool):
            raise StrategyLabError("buffer_enabled must be boolean")
        if (
            isinstance(self.drop_rank_buffer, bool)
            or not isinstance(self.drop_rank_buffer, int)
            or self.drop_rank_buffer < 0
        ):
            raise StrategyLabError("drop_rank_buffer must be a non-negative integer")
        if self.buffer_enabled and self.drop_rank_buffer < self.top_n:
            raise StrategyLabError(
                "drop_rank_buffer must be >= top_n when buffer is enabled"
            )
        if not isinstance(self.selection_relevant, bool):
            raise StrategyLabError("selection_relevant must be boolean")
        if self.execution_engine not in _ENGINES:
            raise StrategyLabError(
                f"unknown execution engine: {self.execution_engine!r}"
            )
        if self.normalization_method not in _NORMALIZATIONS:
            raise StrategyLabError(
                f"unsupported normalization method: {self.normalization_method!r}"
            )
        if not isinstance(self.allow_engine_fallback, bool):
            raise StrategyLabError("allow_engine_fallback must be boolean")
        if (
            not self.universe
            or not self.dataset_version
            or not self.cost_scenario
            or not self.execution_semantics_version
        ):
            raise StrategyLabError(
                "universe, dataset_version, and cost_scenario are required"
            )

    def validate(
        self,
        *,
        research: Any,
        registry: FactorRegistry,
        admission_results: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.validate_basic()
        if self.research_id != research.research_id:
            raise StrategyLabError("research_id does not match Research Manifest")
        if self.dataset_version != research.dataset_version:
            raise StrategyLabError("dataset_version does not match Research Manifest")
        if self.universe != research.universe:
            raise StrategyLabError("universe does not match Research Manifest")
        if self.top_n not in research.top_n_search_space:
            raise StrategyLabError(
                "top_n is outside the Research Manifest search space"
            )
        if self.rebalance_frequency not in research.rebalance_search_space:
            raise StrategyLabError(
                "rebalance_frequency is outside the Research Manifest search space"
            )
        if self.cost_scenario not in research.cost_scenarios:
            raise StrategyLabError(
                "cost_scenario is outside the Research Manifest search space"
            )
        candidate_ids = set(research.factor_candidates)
        directions: dict[str, str] = {}
        families: dict[str, str] = {}
        pit_required: dict[str, bool] = {}
        verdicts: dict[str, str] = {}
        if not isinstance(admission_results, Mapping):
            raise StrategyLabError("admission_results must be supplied as a mapping")
        for factor_id in self.factor_ids:
            try:
                definition = registry.get(factor_id)
            except ValueError as exc:
                raise StrategyLabError(str(exc)) from exc
            if factor_id not in candidate_ids:
                raise StrategyLabError(
                    f"factor_id is not declared by Research Manifest: {factor_id!r}"
                )
            directions[factor_id] = definition.direction
            families[factor_id] = definition.family
            pit_required[factor_id] = definition.pit_required
            if factor_id not in admission_results:
                raise StrategyLabError(
                    f"missing explicit admission result for factor_id: {factor_id!r}"
                )
            verdicts[factor_id] = _admission_verdict(admission_results[factor_id])
        if self.selection_relevant and "REJECT" in verdicts.values():
            rejected = [
                factor_id
                for factor_id, verdict in verdicts.items()
                if verdict == "REJECT"
            ]
            raise StrategyLabError(
                "REJECT factors cannot enter selection-relevant strategy trials: "
                + ", ".join(rejected)
            )
        if self.factor_directions is not None:
            if set(self.factor_directions) != set(self.factor_ids):
                raise StrategyLabError(
                    "factor_directions must exactly match factor_ids"
                )
            if self.factor_directions != directions:
                raise StrategyLabError("factor_directions must match FactorRegistry")
        return {
            "directions": directions,
            "families": families,
            "pit_required": pit_required,
            "verdicts": verdicts,
        }

    def to_config(
        self,
        *,
        directions: Mapping[str, str],
        families: Mapping[str, str],
        pit_required: Mapping[str, bool],
        verdicts: Mapping[str, str],
        cost_model: CostModel,
    ) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "research_id": self.research_id,
            "factor_ids": list(self.factor_ids),
            "factor_weights": {
                key: float(self.factor_weights[key])
                for key in sorted(self.factor_weights)
            },
            "factor_directions": dict(sorted(directions.items())),
            "factor_families": dict(sorted(families.items())),
            "factor_pit_required": dict(sorted(pit_required.items())),
            "admission_verdicts": dict(sorted(verdicts.items())),
            "normalization_method": self.normalization_method,
            "top_n": self.top_n,
            "rebalance_frequency": self.rebalance_frequency,
            "buffer_enabled": self.buffer_enabled,
            "drop_rank_buffer": self.drop_rank_buffer,
            "cost_scenario": self.cost_scenario,
            "cost_model": cost_model.summary(),
            "universe": self.universe,
            "dataset_version": self.dataset_version,
            "execution_engine": self.execution_engine,
            "allow_engine_fallback": self.allow_engine_fallback,
            "execution_semantics_version": self.execution_semantics_version,
            "selection_relevant": self.selection_relevant,
        }


class StrategyRegistry:
    """In-memory definition registry with deterministic duplicate protection."""

    def __init__(self) -> None:
        self._definitions: dict[str, StrategyDefinition] = {}

    def register(self, definition: StrategyDefinition) -> None:
        definition.validate_basic()
        if definition.strategy_id in self._definitions:
            raise StrategyLabError(f"duplicate strategy_id: {definition.strategy_id!r}")
        self._definitions[definition.strategy_id] = definition

    def get(self, strategy_id: str) -> StrategyDefinition:
        try:
            return self._definitions[strategy_id]
        except KeyError as exc:
            raise StrategyLabError(f"unknown strategy_id: {strategy_id!r}") from exc

    def list_definitions(self) -> list[StrategyDefinition]:
        return [self._definitions[key] for key in sorted(self._definitions)]


@dataclass(frozen=True)
class StrategyTrialResult:
    strategy_id: str
    experiment_id: str
    research_id: str
    factor_ids: tuple[str, ...]
    factor_weights: dict[str, float]
    factor_directions: dict[str, str]
    factor_families: dict[str, str]
    dataset_version: str
    top_n: int
    rebalance_frequency: str
    buffer_enabled: bool
    drop_rank_buffer: int
    cost_scenario: str
    requested_engine: str
    actual_engine: str
    execution_semantics_version: str
    cost_model: dict[str, float]
    period: dict[str, str | None]
    total_return: float | None
    sharpe: float | None
    max_drawdown: float | None
    turnover: float | None
    exposure: float | None
    status: str
    handoff_dir: str | None
    transactions_available: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "experiment_id": self.experiment_id,
            "research_id": self.research_id,
            "factor_ids": list(self.factor_ids),
            "factor_weights": dict(sorted(self.factor_weights.items())),
            "factor_directions": dict(sorted(self.factor_directions.items())),
            "factor_families": dict(sorted(self.factor_families.items())),
            "dataset_version": self.dataset_version,
            "top_n": self.top_n,
            "rebalance_frequency": self.rebalance_frequency,
            "buffer_enabled": self.buffer_enabled,
            "drop_rank_buffer": self.drop_rank_buffer,
            "cost_scenario": self.cost_scenario,
            "requested_engine": self.requested_engine,
            "actual_engine": self.actual_engine,
            "execution_semantics_version": self.execution_semantics_version,
            "cost_model": self.cost_model,
            "period": self.period,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "exposure": self.exposure,
            "status": self.status,
            "handoff_dir": self.handoff_dir,
            "transactions_available": self.transactions_available,
            "notes": list(self.notes),
        }


def _validate_matrix(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise StrategyLabError(f"{name} must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise StrategyLabError(f"{name} index must be a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise StrategyLabError(f"{name} index must be unique and sorted")
    if frame.columns.has_duplicates:
        raise StrategyLabError(f"{name} columns must be unique")
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise StrategyLabError(f"{name} must contain numeric values") from exc
    return numeric.replace([np.inf, -np.inf], np.nan)


def _resolve_cost_model(
    definition: StrategyDefinition,
    *,
    cost_model: CostModel | None,
    cost_models: Mapping[str, CostModel] | None,
) -> CostModel:
    if cost_model is not None and cost_models is not None:
        raise StrategyLabError("provide cost_model or cost_models, not both")
    if cost_models is not None:
        try:
            selected = cost_models[definition.cost_scenario]
        except KeyError as exc:
            raise StrategyLabError(
                f"no CostModel supplied for cost scenario: {definition.cost_scenario!r}"
            ) from exc
        if not isinstance(selected, CostModel):
            raise StrategyLabError("cost_models values must be CostModel instances")
        return selected
    if cost_model is not None:
        if not isinstance(cost_model, CostModel):
            raise StrategyLabError("cost_model must be a CostModel instance")
        return cost_model
    if definition.cost_scenario in {"no_cost", "none"}:
        return CostModel(0.0, 0.0, 0.0, 0.0)
    return CostModel()


def _write_frame(frame: pd.DataFrame, path: Path, root: Path) -> None:
    target = assert_write_allowed(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=True, index_label="date", lineterminator="\n")


def _write_handoff(
    *,
    root: Path,
    definition: StrategyDefinition,
    experiment_id: str,
    results: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    trial_payload: dict[str, Any],
) -> str:
    handoff_dir = (
        root
        / "data"
        / "research"
        / definition.research_id
        / "strategy_handoffs"
        / f"{definition.strategy_id}__{experiment_id}"
    )
    handoff_dir = assert_write_allowed(handoff_dir / "metadata.json", root).parent
    returns, positions, _transactions = to_pyfolio_inputs(results)
    _write_frame(returns.to_frame(), handoff_dir / "returns.csv", root)
    _write_frame(
        results.set_index("date")[["equity"]].rename(columns={"equity": "nav"}),
        handoff_dir / "nav.csv",
        root,
    )
    _write_frame(positions, handoff_dir / "positions.csv", root)
    target_weights = portfolio_weights.copy()
    target_weights["date"] = pd.to_datetime(target_weights["date"])
    target_weights["execution_date"] = pd.to_datetime(
        target_weights["execution_date"]
    )
    target_records = [
        {
            "signal_date": row.date.date().isoformat(),
            "execution_date": row.execution_date.date().isoformat(),
            "ticker": str(row.ticker),
            "target_weight": float(row.target_weight),
            "execution_lag_days": int(row.execution_lag_days),
        }
        for row in target_weights.sort_values(
            ["execution_date", "ticker"]
        ).itertuples(index=False)
    ]
    metadata = {
        **trial_payload,
        "handoff_dir": str(handoff_dir.relative_to(root)),
        "positions_contract": "dollar_positions_including_cash",
        "target_weights_contract": "long_target_events",
        "target_weights": target_records,
        "transactions": {
            "available": False,
            "status": "unavailable",
            "reason": "existing pyfolio adapter does not reconstruct transactions",
        },
    }
    metadata_path = assert_write_allowed(handoff_dir / "metadata.json", root)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(handoff_dir.relative_to(root))


def _build_trial_payload(
    *,
    definition: StrategyDefinition,
    experiment_id: str,
    resolved: Mapping[str, Any],
    cost_model: CostModel,
    metrics: pd.Series,
    handoff_dir: str | None,
    notes: list[str],
) -> StrategyTrialResult:
    return StrategyTrialResult(
        strategy_id=definition.strategy_id,
        experiment_id=experiment_id,
        research_id=definition.research_id,
        factor_ids=definition.factor_ids,
        factor_weights=dict(definition.factor_weights),
        factor_directions=dict(resolved["directions"]),
        factor_families=dict(resolved["families"]),
        dataset_version=definition.dataset_version,
        top_n=definition.top_n,
        rebalance_frequency=definition.rebalance_frequency,
        buffer_enabled=definition.buffer_enabled,
        drop_rank_buffer=definition.drop_rank_buffer,
        cost_scenario=definition.cost_scenario,
        requested_engine=definition.execution_engine,
        actual_engine=str(
            metrics.get("actual_engine", metrics.get("engine", "unknown"))
        ),
        execution_semantics_version=definition.execution_semantics_version,
        cost_model=cost_model.summary(),
        period={
            "start_date": str(metrics.get("start_date"))
            if pd.notna(metrics.get("start_date"))
            else None,
            "end_date": str(metrics.get("end_date"))
            if pd.notna(metrics.get("end_date"))
            else None,
        },
        total_return=_optional_float(metrics.get("total_return")),
        sharpe=_optional_float(metrics.get("sharpe")),
        max_drawdown=_optional_float(metrics.get("max_drawdown")),
        turnover=_optional_float(metrics.get("turnover")),
        exposure=_optional_float(metrics.get("avg_exposure")),
        status="completed",
        handoff_dir=handoff_dir,
        transactions_available=False,
        notes=tuple(notes),
    )


def build_strategy_targets(
    *,
    definition: StrategyDefinition,
    root: str | Path,
    factor_matrices: Mapping[str, pd.DataFrame],
    close_matrix: pd.DataFrame,
    admission_results: Mapping[str, Any],
    registry: FactorRegistry | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build canonical rebalance targets shared by IS trials and fresh-state OOS."""
    root = Path(root)
    registry = registry or build_default_registry()
    research = load_research_manifest(root, definition.research_id)
    resolved = definition.validate(
        research=research,
        registry=registry,
        admission_results=admission_results,
    )
    close = _validate_matrix(close_matrix, "close_matrix").sort_index(axis=1)
    if close.empty:
        raise StrategyLabError("close_matrix must not be empty")
    matrices: dict[str, pd.DataFrame] = {}
    for factor_id in definition.factor_ids:
        if factor_id not in factor_matrices:
            raise StrategyLabError(f"missing factor matrix: {factor_id!r}")
        matrix = _validate_matrix(
            factor_matrices[factor_id], f"factor_matrices[{factor_id!r}]"
        )
        matrices[factor_id] = matrix.reindex(index=close.index, columns=close.columns)
    composed = compose_factor_scores(
        matrices=matrices,
        directions=resolved["directions"],
        weights=definition.factor_weights,
    )
    composed = composed.reindex(index=close.index, columns=close.columns)
    factor_frame = (
        composed.stack(future_stack=True)
        .rename("composite_score")
        .rename_axis(["date", "ticker"])
        .reset_index()
    )
    factor_frame["composite_type"] = definition.strategy_id
    factor_frame["is_snapshot_component_used"] = False
    calendar = build_rebalance_calendar(
        close.index,
        frequency=definition.rebalance_frequency,
        execution_lag_days=1,
    )
    signal_dates = (
        pd.DatetimeIndex(calendar["signal_date"]) if not calendar.empty else None
    )
    positions = build_topn_positions(
        factor_frame,
        top_n=definition.top_n,
        factor_name=definition.strategy_id,
        rebalance_dates=signal_dates,
        hold_until_drop=definition.buffer_enabled,
        drop_rank_buffer=definition.drop_rank_buffer,
        rebalance_frequency=definition.rebalance_frequency,
    )
    portfolio_weights = build_equal_weight_portfolio(
        positions,
        rebalance_calendar=calendar,
    )
    return close, portfolio_weights, resolved


def run_strategy_trial(
    *,
    definition: StrategyDefinition,
    experiment_id: str,
    root: str | Path,
    factor_matrices: Mapping[str, pd.DataFrame],
    close_matrix: pd.DataFrame,
    admission_results: Mapping[str, Any],
    registry: FactorRegistry | None = None,
    strategy_registry: StrategyRegistry | None = None,
    cost_model: CostModel | None = None,
    cost_models: Mapping[str, CostModel] | None = None,
    initial_cash: float = 1_000_000,
) -> StrategyTrialResult:
    """Run one explicit strategy config and persist one governed experiment."""
    root = Path(root)
    registry = registry or build_default_registry()
    strategy_registry = strategy_registry or StrategyRegistry()
    strategy_registry.register(definition)
    selected_cost_model = _resolve_cost_model(
        definition, cost_model=cost_model, cost_models=cost_models
    )
    initial_cash = _finite(initial_cash, "initial_cash")
    if initial_cash <= 0:
        raise StrategyLabError("initial_cash must be positive")
    _require_slug(experiment_id, "experiment_id")
    close, portfolio_weights, resolved = build_strategy_targets(
        definition=definition,
        root=root,
        factor_matrices=factor_matrices,
        close_matrix=close_matrix,
        admission_results=admission_results,
        registry=registry,
    )
    config = definition.to_config(
        directions=resolved["directions"],
        families=resolved["families"],
        pit_required=resolved["pit_required"],
        verdicts=resolved["verdicts"],
        cost_model=selected_cost_model,
    )
    record = ExperimentRecord(
        experiment_id=experiment_id,
        research_id=definition.research_id,
        config=config,
        dataset_version=definition.dataset_version,
        experiment_type="strategy_backtest",
        status="running",
        selection_relevant=definition.selection_relevant,
    )
    register_experiment(record, root)
    try:
        results, metrics_frame = run_weight_backtest(
            close_matrix=close,
            portfolio_weights=portfolio_weights,
            cost_model=selected_cost_model,
            initial_cash=initial_cash,
            top_n=definition.top_n,
            use_vectorbt=definition.execution_engine == "vectorbt",
            allow_fallback=definition.allow_engine_fallback,
        )
        metrics = metrics_frame.iloc[0]
        actual_engine = str(
            metrics.get("actual_engine", metrics.get("engine", "unknown"))
        )
        notes: list[str] = []
        if definition.execution_engine == "vectorbt" and actual_engine != "vectorbt":
            notes.append("vectorbt unavailable; result uses labelled fallback engine")
        trial = _build_trial_payload(
            definition=definition,
            experiment_id=experiment_id,
            resolved=resolved,
            cost_model=selected_cost_model,
            metrics=metrics,
            handoff_dir=None,
            notes=notes,
        )
        handoff_dir = _write_handoff(
            root=root,
            definition=definition,
            experiment_id=experiment_id,
            results=results,
            portfolio_weights=portfolio_weights,
            trial_payload=trial.to_dict(),
        )
        trial = replace(trial, handoff_dir=handoff_dir)
        update_experiment_status(
            root,
            definition.research_id,
            experiment_id,
            "completed",
            result=trial.to_dict(),
        )
        return trial
    except Exception as exc:
        update_experiment_status(
            root,
            definition.research_id,
            experiment_id,
            "failed",
            result={"error": str(exc)},
        )
        raise


def _result_metric(result: Any, metric: str) -> float | None:
    if isinstance(result, Mapping):
        value = result.get(metric)
        if metric == "exposure" and value is None:
            value = result.get("avg_exposure")
    else:
        value = getattr(result, metric, None)
        if metric == "exposure" and value is None:
            value = getattr(result, "avg_exposure", None)
    return _optional_float(value)


def compare_incremental_contribution(
    baseline: StrategyTrialResult | Mapping[str, Any],
    enriched: StrategyTrialResult | Mapping[str, Any],
) -> dict[str, Any]:
    """Return enriched-minus-baseline canonical metric deltas only."""
    for result in (baseline, enriched):
        status = (
            result.get("status", "completed")
            if isinstance(result, Mapping)
            else getattr(result, "status", "completed")
        )
        if status != "completed":
            raise StrategyLabError("incremental comparison requires completed trials")
    output: dict[str, Any] = {
        "baseline_strategy_id": (
            baseline.get("strategy_id")
            if isinstance(baseline, Mapping)
            else baseline.strategy_id
        ),
        "enriched_strategy_id": (
            enriched.get("strategy_id")
            if isinstance(enriched, Mapping)
            else enriched.strategy_id
        ),
    }
    for metric in _CANONICAL_METRICS:
        base_value = _result_metric(baseline, metric)
        enriched_value = _result_metric(enriched, metric)
        output[metric] = (
            None
            if base_value is None or enriched_value is None
            else enriched_value - base_value
        )
    return output

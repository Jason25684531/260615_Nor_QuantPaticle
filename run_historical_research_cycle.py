"""Phase 3 real-historical E2E for harden-research-validity-and-oos-v1.

Runs the full research cycle (governance -> Factor Gate -> Strategy Lab ->
tri-engine -> performance -> attribution -> robustness -> fresh-state OOS ->
statistics -> acceptance -> freeze) against real TWSE PIT parquet data under
data/processed/. No synthetic data, no post-hoc reselection.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from twse_factor_lab.acceptance.research_cycle import (
    build_acceptance_matrix,
    build_research_trial_inventory,
    compute_statistical_acceptance,
    freeze_research_cycle,
    run_fresh_oos,
    verify_research_freeze,
)
from twse_factor_lab.analysis.attribution import run_attribution_diagnostic
from twse_factor_lab.analysis.factor_gate import FactorGateConfig, evaluate_factor
from twse_factor_lab.analysis.performance import run_performance_diagnostic
from twse_factor_lab.analysis.research_robustness import (
    run_factor_ablation,
    run_robustness_sweep,
)
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.tri_engine import run_tri_engine_validation
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.factors.composer import compose_factor_scores
from twse_factor_lab.factors.registry import build_default_registry
from twse_factor_lab.governance import (
    DatasetManifest,
    ExperimentRecord,
    ResearchManifest,
    add_dataset_manifest,
    register_experiment,
    save_research_manifest,
)
from twse_factor_lab.strategy import StrategyDefinition, run_strategy_trial
from twse_factor_lab.strategy.handoff import load_strategy_handoff
from twse_factor_lab.strategy.lab import build_strategy_targets

ROOT = Path(__file__).resolve().parent
RESEARCH_ID = "multifactor-validation-historical-v1"
DATASET_ID = "historical-pit-v1"
UNIVERSE_ID = "twse-13-ticker-value-quality-momentum-v1"
UNIVERSE_TICKERS = [
    "1301", "1402", "1476", "1326", "1101",
    "1303", "1216", "1102", "1314", "1210", "1319", "1229", "1312",
]
IS_START, IS_END = "2019-01-01", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2025-12-31"
PRIMARY_HORIZON = 20
FACTOR_CANDIDATES = [
    "pe", "pb", "dividend_yield", "roe", "eps", "revenue_yoy",
    "momentum_40d", "momentum_60d", "risk_adjusted_momentum",
]
TOP_N_SEARCH = [3, 5]
REBALANCE_SEARCH = ["daily", "weekly"]
COST_SCENARIOS = ["no_cost", "base_cost"]


def _cost_for_scenario(scenario: str) -> CostModel:
    if scenario in ("no_cost", "none"):
        return CostModel(0.0, 0.0, 0.0, 0.0)
    return CostModel()


def load_real_matrices() -> dict[str, Any]:
    """Load close/factor matrices for the 13-ticker universe. Read-only."""
    processed = ROOT / "data" / "processed"
    paths = {
        "close_matrix": processed / "close_matrix.parquet",
        "factors_price_volume": processed / "factors_price_volume.parquet",
        "factors_fundamental": processed / "factors_fundamental.parquet",
        "research_universe": processed / "research_universe.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Phase 3 BLOCKED: required real PIT files missing: {missing}"
        )

    close = pd.read_parquet(paths["close_matrix"])
    close.columns = close.columns.astype(str)
    close = close[[c for c in UNIVERSE_TICKERS if c in close.columns]].sort_index()

    price_volume = pd.read_parquet(paths["factors_price_volume"])
    fundamental = pd.read_parquet(paths["factors_fundamental"])
    factor_matrices: dict[str, pd.DataFrame] = {}
    for source in (price_volume, fundamental):
        source = source[source["ticker"].astype(str).isin(UNIVERSE_TICKERS)]
        for column in FACTOR_CANDIDATES:
            if column in source.columns:
                pivot = source.pivot_table(
                    index="date", columns="ticker", values=column, aggfunc="last"
                )
                pivot.index = pd.to_datetime(pivot.index)
                pivot.columns = pivot.columns.astype(str)
                factor_matrices[column] = pivot.sort_index()

    missing_factors = set(FACTOR_CANDIDATES) - set(factor_matrices)
    if missing_factors:
        raise FileNotFoundError(
            f"Phase 3 BLOCKED: candidate factors absent from real data: "
            f"{missing_factors}"
        )
    return {"close": close, "factors": factor_matrices, "paths": paths}


def build_dataset_manifest(
    paths: dict[str, Path], close: pd.DataFrame
) -> DatasetManifest:
    hasher = hashlib.sha256()
    row_count = 0
    for name in sorted(paths):
        data = paths[name].read_bytes()
        hasher.update(data)
        row_count += len(pd.read_parquet(paths[name]))
    coverage = float(close.notna().mean().mean())
    return DatasetManifest(
        dataset_id=DATASET_ID,
        source="data/processed (TWSE historical PIT parquet)",
        retrieved_at="2026-09-10",
        schema_version="processed-v1",
        processing_version="factor-pipeline-v1",
        pit_rule=(
            "point-in-time as produced by "
            "run_data_pipeline/run_fundamental_pipeline"
        ),
        row_count=row_count,
        coverage=coverage,
        artifact_sha256=hasher.hexdigest(),
    )


def run_factor_gate_stage(
    close_is: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    dataset_manifest: DatasetManifest,
) -> dict[str, Any]:
    registry = build_default_registry()
    config = FactorGateConfig(primary_horizon=PRIMARY_HORIZON)
    results = {}
    for factor_id in FACTOR_CANDIDATES:
        matrix = factors[factor_id].loc[IS_START:IS_END]
        diagnostics = evaluate_factor(
            factor_id=factor_id,
            research_id=RESEARCH_ID,
            experiment_id=f"factor-gate-{factor_id}",
            root=ROOT,
            factor_matrix=matrix,
            close_matrix=close_is,
            dataset_manifest=dataset_manifest,
            registry=registry,
            config=config,
        )
        results[factor_id] = diagnostics
    return results


def robustness_runner(
    research: ResearchManifest,
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    admission_results: dict[str, Any],
):
    def runner(config: dict[str, Any]) -> dict[str, float]:
        factor_ids = config["factor_ids"]
        definition = StrategyDefinition(
            strategy_id="robustness-scenario",
            research_id=RESEARCH_ID,
            factor_ids=factor_ids,
            factor_weights=config["factor_weights"],
            top_n=config["top_n"],
            rebalance_frequency=config["rebalance_frequency"],
            buffer_enabled=False,
            drop_rank_buffer=0,
            cost_scenario=config["cost_scenario"],
            universe=UNIVERSE_ID,
            dataset_version=research.dataset_version,
            selection_relevant=False,
        )
        close_frame, weights, _ = build_strategy_targets(
            definition=definition,
            root=ROOT,
            factor_matrices={fid: factors[fid] for fid in factor_ids},
            close_matrix=close,
            admission_results=admission_results,
        )
        _, metrics = run_weight_backtest(
            close_matrix=close_frame,
            portfolio_weights=weights,
            cost_model=_cost_for_scenario(config["cost_scenario"]),
            initial_cash=1_000_000.0,
            top_n=config["top_n"],
            use_vectorbt=False,
        )
        row = metrics.iloc[0]
        return {
            "total_return": float(row["total_return"]),
            "sharpe": float(row["sharpe"]),
            "max_drawdown": float(row["max_drawdown"]),
            "turnover": float(row["turnover"]),
            "exposure": float(row["avg_exposure"]),
        }

    return runner


def build_evidence(
    factor_results: dict[str, Any],
    admitted: list[str],
    ablation: dict[str, Any] | None,
    tri_engine: dict[str, Any],
    performance: dict[str, Any],
    attribution: dict[str, Any],
    robustness: dict[str, Any],
    oos_result: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    performance_status = performance["cross_check"]["status"]
    performance_status = "PASS" if performance_status in (
        "PASS", "PASS_WITH_DEFINITION_DIFFERENCE"
    ) else "FAIL"

    if ablation is None:
        incremental_status = "UNAVAILABLE"
    else:
        incremental_status = (
            "MIXED"
            if any(d is not None and d > 0 for d in ablation["sharpe_deltas"].values())
            else "PASS"
        )

    oos_status = (
        "PASS"
        if oos_result.get("lookahead", {}).get("status") == "PASS"
        and oos_result.get("fresh_state")
        else "FAIL"
    )

    return {
        "factor_evidence": {
            "status": "PASS" if admitted else "REJECT",
            "admitted_factors": admitted,
            "results": {fid: diag.to_dict() for fid, diag in factor_results.items()},
        },
        "strategy_incremental_value": {
            "status": incremental_status,
            "ablation": ablation,
        },
        "execution": tri_engine,
        "performance": {"status": performance_status, "report": performance},
        "attribution": {"status": attribution["status"], "report": attribution},
        "robustness": {
            "status": robustness["robustness_result"],
            "report": robustness,
        },
        "oos": {"status": oos_status, "report": oos_result},
        "statistics": statistics,
    }


def main() -> None:
    matrices = load_real_matrices()
    close, factors, paths = matrices["close"], matrices["factors"], matrices["paths"]

    research = ResearchManifest(
        research_id=RESEARCH_ID,
        hypothesis=(
            "Value/Quality/Momentum factors on a 13-ticker TWSE universe carry "
            "positive, admissible signal at a pre-declared 20-day horizon."
        ),
        factor_candidates=FACTOR_CANDIDATES,
        universe=UNIVERSE_ID,
        dataset_version=DATASET_ID,
        is_start=IS_START,
        is_end=IS_END,
        oos_start=OOS_START,
        oos_end=OOS_END,
        rebalance_search_space=REBALANCE_SEARCH,
        top_n_search_space=TOP_N_SEARCH,
        cost_scenarios=COST_SCENARIOS,
        selection_relevant=True,
        status="active",
    )
    save_research_manifest(research, ROOT)
    dataset_manifest = build_dataset_manifest(paths, close)
    add_dataset_manifest(dataset_manifest, ROOT, RESEARCH_ID)

    close_is = close.loc[IS_START:IS_END]
    factor_results = run_factor_gate_stage(close_is, factors, dataset_manifest)
    admitted = sorted(
        fid for fid, diag in factor_results.items()
        if diag.verdict in ("ACCEPT", "CANDIDATE")
    )
    print(f"Factor Gate admitted: {admitted}")
    if not admitted:
        print("No factors admitted; Strategy Research = NOT_EVALUATED.")
        return

    admitted_factors_is = {fid: factors[fid].loc[IS_START:IS_END] for fid in admitted}
    weights = {fid: 1.0 / len(admitted) for fid in admitted}

    is_trials = []
    for top_n, rebalance, cost in product(
        TOP_N_SEARCH, REBALANCE_SEARCH, COST_SCENARIOS
    ):
        strategy_id = f"is-trial-top{top_n}-{rebalance}-{cost}"
        definition = StrategyDefinition(
            strategy_id=strategy_id,
            research_id=RESEARCH_ID,
            factor_ids=tuple(admitted),
            factor_weights=weights,
            top_n=top_n,
            rebalance_frequency=rebalance,
            buffer_enabled=False,
            drop_rank_buffer=0,
            cost_scenario=cost,
            universe=UNIVERSE_ID,
            dataset_version=DATASET_ID,
            selection_relevant=True,
        )
        result = run_strategy_trial(
            definition=definition,
            experiment_id=f"strategy-{strategy_id}",
            root=ROOT,
            factor_matrices=admitted_factors_is,
            close_matrix=close_is,
            admission_results=factor_results,
        )
        is_trials.append(result)

    locked = max(is_trials, key=lambda r: r.sharpe)
    print(f"Locked candidate: {locked.strategy_id} (IS Sharpe={locked.sharpe:.4f})")

    locked_definition = StrategyDefinition(
        strategy_id=locked.strategy_id,
        research_id=RESEARCH_ID,
        factor_ids=tuple(admitted),
        factor_weights=weights,
        top_n=locked.top_n,
        rebalance_frequency=locked.rebalance_frequency,
        buffer_enabled=False,
        drop_rank_buffer=0,
        cost_scenario=locked.cost_scenario,
        universe=UNIVERSE_ID,
        dataset_version=DATASET_ID,
        selection_relevant=True,
    )

    tri_engine = run_tri_engine_validation(
        root=ROOT,
        research_id=RESEARCH_ID,
        strategy_id=locked.strategy_id,
        handoff_dir=locked.handoff_dir,
        experiment_id=f"tri-engine-{locked.strategy_id}",
        close_matrix=close_is,
        cost_model=_cost_for_scenario(locked.cost_scenario),
    )
    print(f"Tri-engine parity: {tri_engine['status']}")

    performance = run_performance_diagnostic(
        root=ROOT,
        research_id=RESEARCH_ID,
        strategy_id=locked.strategy_id,
        handoff_dir=locked.handoff_dir,
        experiment_id=f"performance-{locked.strategy_id}",
    )
    print(f"Performance cross-check: {performance['cross_check']['status']}")

    handoff = load_strategy_handoff(ROOT, locked.handoff_dir)
    close_for_targets, portfolio_weights, _ = build_strategy_targets(
        definition=locked_definition,
        root=ROOT,
        factor_matrices=admitted_factors_is,
        close_matrix=close_is,
        admission_results=factor_results,
    )
    raw_factor = compose_factor_scores(
        matrices=admitted_factors_is,
        directions={fid: factor_results[fid].direction for fid in admitted},
        weights=weights,
    )
    attribution = run_attribution_diagnostic(
        root=ROOT,
        research_id=RESEARCH_ID,
        strategy_id=locked.strategy_id,
        experiment_id=f"attribution-{locked.strategy_id}",
        raw_factor=raw_factor,
        close_matrix=close_for_targets,
        portfolio_weights=portfolio_weights,
        descriptors=None,
        actual_returns=handoff.returns,
    )
    print(f"Attribution status: {attribution['status']}")

    candidate_config = {
        "strategy_id": locked.strategy_id,
        "factor_ids": list(admitted),
        "factor_weights": weights,
        "top_n": locked.top_n,
        "rebalance_frequency": locked.rebalance_frequency,
        "cost_scenario": locked.cost_scenario,
    }
    runner = robustness_runner(research, factors, close_is, factor_results)
    robustness = run_robustness_sweep(
        root=ROOT,
        research_id=RESEARCH_ID,
        strategy_id=locked.strategy_id,
        experiment_id=f"robustness-{locked.strategy_id}",
        candidate_config=candidate_config,
        runner=runner,
        baseline_result={
            "total_return": locked.total_return,
            "sharpe": locked.sharpe,
            "max_drawdown": locked.max_drawdown,
            "turnover": locked.turnover,
            "exposure": locked.exposure,
        },
    )
    print(f"Robustness classification: {robustness['robustness_result']}")

    ablation = None
    if len(admitted) > 1:
        sharpe_deltas = {}
        for factor_id in admitted:
            result = run_factor_ablation(
                root=ROOT,
                research_id=RESEARCH_ID,
                strategy_id=locked.strategy_id,
                experiment_id=f"ablation-{locked.strategy_id}-{factor_id}",
                candidate_config=candidate_config,
                factor_id=factor_id,
                baseline_result={
                    "total_return": locked.total_return,
                    "sharpe": locked.sharpe,
                    "max_drawdown": locked.max_drawdown,
                    "turnover": locked.turnover,
                    "exposure": locked.exposure,
                },
                runner=runner,
            )
            sharpe_deltas[factor_id] = result["deltas"]["sharpe"]
        ablation = {"sharpe_deltas": sharpe_deltas}

    full_factors_admitted = {fid: factors[fid] for fid in admitted}
    oos_result = run_fresh_oos(
        root=ROOT,
        research_id=RESEARCH_ID,
        strategy_id=locked.strategy_id,
        experiment_id=f"fresh-oos-{locked.strategy_id}",
        definition=locked_definition,
        factor_matrices=full_factors_admitted,
        close_matrix=close,
        admission_results=factor_results,
        cost_model=_cost_for_scenario(locked.cost_scenario),
        oos_start=OOS_START,
        oos_end=OOS_END,
    )
    print(f"Fresh-state OOS lookahead: {oos_result['lookahead']['status']}")

    oos_returns = pd.Series(
        {row["date"]: row["value"] for row in oos_result["oos_returns"]}
    )
    oos_returns.index = pd.to_datetime(oos_returns.index)
    inventory, effective_trials = build_research_trial_inventory(ROOT, RESEARCH_ID)
    statistics = compute_statistical_acceptance(oos_returns.sort_index(), inventory)
    print(
        f"Statistics: status={statistics['status']} psr={statistics['psr']:.4f} "
        f"dsr={statistics['dsr']:.4f} strategy_trials={statistics['effective_trials']} "
        f"effective_selection_relevant={effective_trials}"
    )

    evidence = build_evidence(
        factor_results, admitted, ablation, tri_engine, performance,
        attribution, robustness, oos_result, statistics,
    )
    acceptance = build_acceptance_matrix(evidence, real_candidate=True)
    print(
        f"Acceptance: platform={acceptance['research_platform_verdict']} "
        f"strategy={acceptance['strategy_verdict']}"
    )

    register_experiment(
        ExperimentRecord(
            experiment_id="final-acceptance-matrix",
            research_id=RESEARCH_ID,
            config={
                "experiment_type": "acceptance_matrix",
                "selection_relevant": False,
            },
            dataset_version=DATASET_ID,
            experiment_type="diagnostic",
            status="completed",
            selection_relevant=False,
            result=acceptance,
        ),
        ROOT,
    )

    freeze_path = ROOT / "data" / "research" / RESEARCH_ID / "acceptance_matrix.json"
    freeze_path.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    frozen = freeze_research_cycle(
        root=ROOT,
        research_id=RESEARCH_ID,
        candidate_config=candidate_config,
        acceptance=acceptance,
        require_clean_tree=True,
    )
    print(f"Freeze verify status: {frozen['status']}")

    reverify = verify_research_freeze(ROOT, RESEARCH_ID)
    print(f"Post-freeze re-verify: {reverify['status']}")


if __name__ == "__main__":
    main()

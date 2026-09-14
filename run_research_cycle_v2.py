"""Research Cycle v2 real-historical E2E: quality-cost-breadth-v2.

Liquid PIT-safe TWSE universe -> EPS/ROE factor gate + 50/50 composite ablation
-> low-turnover staged strategy search (weekly/monthly x buffer off/on, base
cost) -> candidate lock -> MA60 breadth overlay -> performance metric audit ->
walk-forward validation -> PSR/DSR acceptance -> report -> freeze.

Reuses every canonical primitive; introduces no second backtester and does not
touch the frozen multifactor-validation-historical-v1 cycle. Results are
reported honestly: a Strategy REJECT is a legal outcome.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from twse_factor_lab.acceptance.research_cycle import (
    build_acceptance_matrix,
    build_research_trial_inventory,
    compute_statistical_acceptance,
    freeze_research_cycle,
    verify_research_freeze,
)
from twse_factor_lab.acceptance.walk_forward import DEFAULT_FOLDS, run_walk_forward
from twse_factor_lab.analysis.factor_gate import FactorGateConfig, evaluate_factor
from twse_factor_lab.analysis.performance import (
    canonical_metric_metadata,
    run_performance_diagnostic,
)
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.factors.pit_matrix import build_pit_factor_matrix
from twse_factor_lab.factors.registry import build_default_registry
from twse_factor_lab.governance import (
    DatasetManifest,
    ExperimentRecord,
    ResearchManifest,
    add_dataset_manifest,
    register_experiment,
    save_research_manifest,
)
from twse_factor_lab.governance.isolation import assert_write_allowed
from twse_factor_lab.portfolio.breadth import (
    breadth_to_exposure,
    compute_market_breadth,
)
from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar
from twse_factor_lab.portfolio.weights import apply_gross_exposure
from twse_factor_lab.reporting.model import ReportSection, ResearchReportModel
from twse_factor_lab.reporting.render_html import render_html
from twse_factor_lab.reporting.render_md import render_markdown
from twse_factor_lab.strategy.lab import (
    StrategyDefinition,
    build_strategy_targets,
    run_strategy_trial,
)

ROOT = Path(__file__).resolve().parent
RESEARCH_ID = "quality-cost-breadth-v2"
DATASET_ID = "historical-pit-v2"
UNIVERSE_ID = "twse-liquid-pit-v2"
FACTORS = ("eps", "roe")
PRIMARY_HORIZON = 20
IS_START, IS_END = "2019-01-01", "2021-12-31"
OOS_START, OOS_END = "2022-01-01", "2025-12-31"
WARMUP_START = "2018-01-01"
LIQUIDITY_WINDOW = 20
LIQUIDITY_THRESHOLD = 50_000_000
BREADTH_THRESHOLD, BREADTH_HIGH, BREADTH_LOW = 0.40, 1.0, 0.5
INITIAL_CASH = 1_000_000.0
TOP_N = 3
DROP_RANK_BUFFER = 5

_ADMISSIBLE = {"ACCEPT", "CANDIDATE"}


def _write(path: Path, payload: Any) -> str:
    target = assert_write_allowed(path, ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    return target.relative_to(ROOT).as_posix()


def _cost(scenario: str) -> CostModel:
    return CostModel(0.0, 0.0, 0.0, 0.0) if scenario in {"no_cost", "none"} else CostModel()


def load_processed() -> dict[str, Any]:
    processed = ROOT / "data" / "processed"
    paths = {
        "universe": processed / "universe.parquet",
        "ohlcv": processed / "ohlcv.parquet",
        "fundamental_matrix": processed / "fundamental_matrix.parquet",
        "close_matrix": processed / "close_matrix.parquet",
        "factors_price_volume": processed / "factors_price_volume.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"v2 BLOCKED: required real PIT files missing: {missing}")
    return {name: pd.read_parquet(path) for name, path in paths.items()} | {"paths": paths}


def build_universe(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from twse_factor_lab.data.universe import (
        build_liquid_pit_coverage,
        build_liquid_pit_universe,
    )

    close = data["close_matrix"]
    close.index = pd.to_datetime(close.index)
    dates = close.index
    v2 = build_liquid_pit_universe(
        data["universe"],
        data["ohlcv"],
        data["fundamental_matrix"],
        dates=dates,
        window=LIQUIDITY_WINDOW,
        threshold=LIQUIDITY_THRESHOLD,
        fundamental_metrics=FACTORS,
    )
    per_date, summary = build_liquid_pit_coverage(v2, fundamental_metrics=FACTORS)
    return v2, per_date, summary


def _factor_matrices(
    data: dict[str, Any], v2: pd.DataFrame, tickers: list[str], index: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    return {
        metric: build_pit_factor_matrix(
            data["fundamental_matrix"], metric,
            v2_universe=v2, index=index, columns=tickers,
        )
        for metric in FACTORS
    }


def _dataset_manifest(paths: dict[str, Path], coverage: float) -> DatasetManifest:
    hasher = hashlib.sha256()
    row_count = 0
    for name in sorted(paths):
        hasher.update(paths[name].read_bytes())
        row_count += len(pd.read_parquet(paths[name]))
    return DatasetManifest(
        dataset_id=DATASET_ID,
        source="data/processed (TWSE historical PIT parquet)",
        retrieved_at="2026-09-10",
        schema_version="processed-v1",
        processing_version="factor-pipeline-v1",
        pit_rule="liquid PIT-safe universe with merge_asof fundamental availability",
        row_count=row_count,
        coverage=coverage,
        artifact_sha256=hasher.hexdigest(),
    )


def _definition(strategy_id: str, factor_ids: tuple[str, ...], rebalance: str,
                buffer_on: bool, cost: str, selection: bool) -> StrategyDefinition:
    weights = {fid: 1.0 / len(factor_ids) for fid in factor_ids}
    return StrategyDefinition(
        strategy_id=strategy_id,
        research_id=RESEARCH_ID,
        factor_ids=factor_ids,
        factor_weights=weights,
        top_n=TOP_N,
        rebalance_frequency=rebalance,
        buffer_enabled=buffer_on,
        drop_rank_buffer=DROP_RANK_BUFFER if buffer_on else 0,
        cost_scenario=cost,
        universe=UNIVERSE_ID,
        dataset_version=DATASET_ID,
        selection_relevant=selection,
    )


def _run(definition: StrategyDefinition, experiment_id: str,
         factors_is: dict[str, pd.DataFrame], close_is: pd.DataFrame,
         admission: dict[str, Any]) -> Any:
    return run_strategy_trial(
        definition=definition,
        experiment_id=experiment_id,
        root=ROOT,
        factor_matrices={fid: factors_is[fid] for fid in definition.factor_ids},
        close_matrix=close_is,
        admission_results=admission,
        cost_model=_cost(definition.cost_scenario),
        initial_cash=INITIAL_CASH,
    )


def _gross_net(arm_id: str, factor_ids: tuple[str, ...], rebalance: str, buffer_on: bool,
               factors_is: dict[str, pd.DataFrame], close_is: pd.DataFrame,
               admission: dict[str, Any], selection: bool) -> dict[str, Any]:
    """Run base_cost (net) selection trial + no_cost (gross) diagnostic; reconcile."""
    net_def = _definition(arm_id, factor_ids, rebalance, buffer_on, "base_cost", selection)
    net = _run(net_def, f"strategy-{arm_id}", factors_is, close_is, admission)
    gross_def = _definition(f"{arm_id}-gross", factor_ids, rebalance, buffer_on,
                            "no_cost", False)
    gross = _run(gross_def, f"strategy-{arm_id}-gross", factors_is, close_is, admission)
    net_ret = net.total_return if net.total_return is not None else float("nan")
    gross_ret = gross.total_return if gross.total_return is not None else float("nan")
    return {
        "configuration": arm_id,
        "factor_ids": list(factor_ids),
        "rebalance": rebalance,
        "buffer": buffer_on,
        "selection_relevant": selection,
        "gross_return": gross_ret,
        "net_return": net_ret,
        "cost_drag": gross_ret - net_ret,
        "turnover": net.turnover,
        "cagr": None,
        "sharpe": net.sharpe,
        "sortino": None,
        "mdd": net.max_drawdown,
        "calmar": None,
        "net_trial": net,
        "net_definition": net_def,
    }


def main() -> None:
    data = load_processed()
    v2, per_date, summary = build_universe(data)
    included_tickers = sorted(
        v2.loc[v2["universe_included"], "ticker"].astype(str).unique()
    )
    print(f"v2 universe: {len(included_tickers)} fundamental-eligible tickers")

    close = data["close_matrix"]
    close.columns = close.columns.astype(str)
    factor_tickers = [t for t in included_tickers if t in close.columns]
    close_factor = close[factor_tickers].sort_index()
    full_index = close_factor.index
    factors_full = _factor_matrices(data, v2, factor_tickers, full_index)

    research = ResearchManifest(
        research_id=RESEARCH_ID,
        hypothesis=(
            "EPS/ROE quality factors on a liquid PIT-safe TWSE universe survive "
            "cost and a single MA60 breadth overlay under walk-forward validation."
        ),
        factor_candidates=list(FACTORS),
        universe=UNIVERSE_ID,
        dataset_version=DATASET_ID,
        is_start=IS_START,
        is_end=IS_END,
        oos_start=OOS_START,
        oos_end=OOS_END,
        rebalance_search_space=["weekly", "monthly"],
        top_n_search_space=[TOP_N],
        cost_scenarios=["no_cost", "base_cost"],
        selection_relevant=True,
        status="active",
    )
    save_research_manifest(research, ROOT)
    coverage_value = float(close_factor.notna().mean().mean())
    add_dataset_manifest(_dataset_manifest(data["paths"], coverage_value), ROOT, RESEARCH_ID)

    universe_manifest = {
        "universe_id": UNIVERSE_ID,
        "methodology": "listed -> tradable -> liquid -> fundamental_available (PIT)",
        "liquidity_rule": f"20D median traded value > TWD {LIQUIDITY_THRESHOLD:,}",
        "liquidity_measure_source": summary["liquidity_measure_source"],
        "survivorship": summary["survivorship"],
        "fundamental_metrics": list(FACTORS),
        "included_ticker_count": len(included_tickers),
        "included_tickers": included_tickers,
    }
    _write(ROOT / "data" / "research" / RESEARCH_ID / "universe" / "universe_manifest.json",
           universe_manifest)
    coverage_report = {
        "summary": summary,
        "per_date": per_date.assign(date=per_date["date"].astype(str)).to_dict("records"),
    }
    _write(ROOT / "data" / "research" / RESEARCH_ID / "universe" / "coverage_report.json",
           coverage_report)

    # --- Factor Gate (EPS, ROE) ---
    registry = build_default_registry()
    gate_config = FactorGateConfig(primary_horizon=PRIMARY_HORIZON)
    close_is = close_factor.loc[IS_START:IS_END]
    factors_is = {fid: factors_full[fid].loc[IS_START:IS_END] for fid in FACTORS}
    factor_results: dict[str, Any] = {}
    for fid in FACTORS:
        factor_results[fid] = evaluate_factor(
            factor_id=fid,
            research_id=RESEARCH_ID,
            experiment_id=f"factor-gate-{fid}",
            root=ROOT,
            factor_matrix=factors_is[fid],
            close_matrix=close_is,
            dataset_manifest=_dataset_manifest(data["paths"], coverage_value),
            registry=registry,
            config=gate_config,
        )
        print(f"Factor Gate {fid}: {factor_results[fid].verdict}")

    admissible = {
        fid for fid in FACTORS if factor_results[fid].verdict in _ADMISSIBLE
    }

    # --- Stage 1: factor ablation (fixed structure top3/monthly/buffer-off) ---
    ablation_arms = {
        "eps-only": ("eps",),
        "roe-only": ("roe",),
        "eps-roe-5050": ("eps", "roe"),
    }
    ablation_rows: list[dict[str, Any]] = []
    stage1: dict[str, dict[str, Any]] = {}
    for arm_id, factor_ids in ablation_arms.items():
        selection = all(fid in admissible for fid in factor_ids)
        row = _gross_net(f"s1-{arm_id}", factor_ids, "monthly", False,
                         factors_is, close_is, factor_results, selection)
        stage1[arm_id] = row
        ablation_rows.append({
            "configuration": arm_id,
            "factor_gate": "/".join(factor_results[f].verdict for f in factor_ids),
            "is_net_return": row["net_return"],
            "sharpe": row["sharpe"],
            "sortino": row["sortino"],
            "mdd": row["mdd"],
            "turnover": row["turnover"],
            "selection_relevant": selection,
        })
    _write(ROOT / "data" / "research" / RESEARCH_ID / "factor_ablation.json",
           {"rows": ablation_rows})

    # Choose the factor structure: best net IS Sharpe among admissible arms.
    admissible_arms = {
        arm: row for arm, row in stage1.items() if row["selection_relevant"]
    }
    strategy_evaluable = bool(admissible_arms)
    lock_row = None
    breadth_report: dict[str, Any] = {
        "status": "UNAVAILABLE",
        "reason": "no factor admitted by the Factor Gate; no locked candidate to overlay",
    }
    cost_rows = [
        {k: v for k, v in row.items() if k not in {"net_trial", "net_definition"}}
        for row in stage1.values()
    ]
    stage2_rows: list[dict[str, Any]] = []
    validation_payload: dict[str, Any] = {
        "label": "WALK-FORWARD VALIDATION",
        "folds": [],
        "reason": "no factor admitted by the Factor Gate; no locked candidate to validate",
    }
    combined_returns = pd.Series(dtype=float)
    performance: dict[str, Any] | None = None

    # Performance metric audit is a platform capability: demonstrate it on a real
    # returns series (the composite ablation handoff) regardless of the strategy
    # verdict, so the metadata contract and downside_deviation are auditable.
    audit_trial = stage1.get("eps-roe-5050", {}).get("net_trial")
    if audit_trial is not None and audit_trial.handoff_dir:
        performance = _audit_performance(audit_trial)

    if strategy_evaluable:
        chosen_arm = max(
            admissible_arms,
            key=lambda a: (admissible_arms[a]["sharpe"] or float("-inf"),
                           -(admissible_arms[a]["turnover"] or 0.0)),
        )
        chosen_factors = ablation_arms[chosen_arm]
        print(f"Chosen factor structure: {chosen_arm} {chosen_factors}")

        # --- Stage 2: structure search (weekly/monthly x buffer off/on) ---
        stage2: dict[str, dict[str, Any]] = {}
        for rebalance in ("weekly", "monthly"):
            for buffer_on in (False, True):
                arm_id = f"s2-{chosen_arm}-{rebalance}-{'buf' if buffer_on else 'nobuf'}"
                row = _gross_net(arm_id, chosen_factors, rebalance, buffer_on,
                                 factors_is, close_is, factor_results, True)
                stage2[arm_id] = row
                stage2_rows.append({
                    "configuration": arm_id,
                    "gross_return": row["gross_return"],
                    "net_return": row["net_return"],
                    "cost_drag": row["cost_drag"],
                    "turnover": row["turnover"],
                    "cagr": row["cagr"],
                    "sharpe": row["sharpe"],
                    "sortino": row["sortino"],
                    "mdd": row["mdd"],
                    "calmar": row["calmar"],
                })
        cost_rows += stage2_rows

        # --- Stage 3: lock on best net IS Sharpe (tie -> lower turnover) ---
        lock_id = max(
            stage2,
            key=lambda a: (stage2[a]["sharpe"] or float("-inf"),
                           -(stage2[a]["turnover"] or 0.0)),
        )
        lock_row = stage2[lock_id]
        locked_definition = lock_row["net_definition"]
        locked_trial = lock_row["net_trial"]
        candidate_config = {
            "strategy_id": locked_trial.strategy_id,
            "factor_ids": list(chosen_factors),
            "factor_weights": dict(locked_definition.factor_weights),
            "top_n": TOP_N,
            "rebalance_frequency": locked_definition.rebalance_frequency,
            "buffer_enabled": locked_definition.buffer_enabled,
            "drop_rank_buffer": locked_definition.drop_rank_buffer,
            "cost_scenario": "base_cost",
        }
        candidate_lock = {
            "config": candidate_config,
            "reason": "highest base_cost net IS Sharpe; ties broken by lower turnover",
            "fingerprint": hashlib.sha256(
                json.dumps(candidate_config, sort_keys=True).encode()
            ).hexdigest(),
        }
        _write(ROOT / "data" / "research" / RESEARCH_ID / "candidate_lock.json",
               candidate_lock)
        print(f"Locked candidate: {locked_trial.strategy_id}")

        # --- Performance metric audit on the locked candidate ---
        performance = run_performance_diagnostic(
            root=ROOT, research_id=RESEARCH_ID,
            strategy_id=locked_trial.strategy_id,
            handoff_dir=locked_trial.handoff_dir,
            experiment_id=f"performance-{locked_trial.strategy_id}",
        )
        _write(ROOT / "data" / "research" / RESEARCH_ID / "performance_metric_definition.json",
               performance["metric_definition"])

        # --- Stage 4: MA60 breadth overlay on the locked candidate ---
        breadth_report = _breadth_overlay(
            data, v2, close, locked_definition, factors_full, factor_results, lock_row
        )

        # --- Walk-forward validation (fresh-state folds) ---
        wf = run_walk_forward(
            root=ROOT, research_id=RESEARCH_ID,
            base_strategy_id=locked_trial.strategy_id,
            definition=locked_definition,
            factor_matrices={fid: factors_full[fid] for fid in chosen_factors},
            close_matrix=close_factor,
            admission_results=factor_results,
            cost_model=_cost("base_cost"),
            folds=DEFAULT_FOLDS,
        )
        validation_payload = wf["payload"]
        combined_returns = wf["combined_returns"]
    else:
        candidate_config = None
        print("No admissible factor; Strategy Research not evaluable.")

    _write(ROOT / "data" / "research" / RESEARCH_ID / "cost_turnover_report.json",
           {"rows": cost_rows})

    # --- Statistical acceptance (PSR/DSR) ---
    inventory, effective = build_research_trial_inventory(ROOT, RESEARCH_ID)
    if strategy_evaluable and not combined_returns.empty and len(combined_returns) > 1:
        statistics = compute_statistical_acceptance(combined_returns, inventory)
    else:
        statistics = {
            "status": "INSUFFICIENT",
            "psr": None, "dsr": None,
            "reason": "no evaluable strategy / walk-forward returns",
            "strategy_selection_trial_count": 0,
        }
    _write(ROOT / "data" / "research" / RESEARCH_ID / "statistical_acceptance.json",
           statistics)

    # --- Final acceptance matrix ---
    evidence = _build_evidence(
        factor_results, admissible, performance, breadth_report,
        validation_payload, statistics, strategy_evaluable
    )
    # This is a real research run with real factor evidence, so acceptance is
    # evaluated honestly: a factor-gate REJECT yields Strategy REJECT (not
    # "not evaluated"). Platform can still PASS.
    acceptance = build_acceptance_matrix(evidence, real_candidate=True)
    _write(ROOT / "data" / "research" / RESEARCH_ID / "acceptance_matrix.json", acceptance)
    register_experiment(
        ExperimentRecord(
            experiment_id="final-acceptance-matrix",
            research_id=RESEARCH_ID,
            config={"experiment_type": "acceptance_matrix", "selection_relevant": False},
            dataset_version=DATASET_ID,
            experiment_type="diagnostic",
            status="completed",
            selection_relevant=False,
            result=acceptance,
        ),
        ROOT,
    )
    print(f"Acceptance: platform={acceptance['research_platform_verdict']} "
          f"strategy={acceptance['strategy_verdict']}")

    # --- Report ---
    _write_report(research, summary, per_date, ablation_rows, stage2_rows, cost_rows,
                  lock_row, performance, breadth_report, validation_payload, statistics,
                  acceptance, included_tickers, factor_results)

    # --- Freeze + verify ---
    frozen = freeze_research_cycle(
        root=ROOT, research_id=RESEARCH_ID,
        candidate_config=candidate_config, acceptance=acceptance,
        require_clean_tree=False,
    )
    print(f"Freeze verify: {frozen['status']} ({frozen['artifact_count']} artifacts)")
    reverify = verify_research_freeze(ROOT, RESEARCH_ID)
    print(f"Post-freeze re-verify: {reverify['status']}")


def _audit_performance(trial: Any) -> dict[str, Any]:
    """Run the canonical performance metric audit on a trial's return series.

    Positions can carry NaN on days the thin liquid+fundamental universe is
    empty, so the audit uses returns only (positions unavailable, stated
    honestly). It still produces the full canonical/pyfolio/metric-definition
    contract and registers one diagnostic experiment.
    """
    from twse_factor_lab.analysis.performance import evaluate_performance

    returns_csv = ROOT / trial.handoff_dir / "returns.csv"
    frame = pd.read_csv(returns_csv, index_col="date", parse_dates=True)
    returns = frame.iloc[:, 0].astype(float)
    report = evaluate_performance(returns)
    report["research_id"] = RESEARCH_ID
    report["strategy_id"] = trial.strategy_id
    report["positions"] = {
        "available": False,
        "reason": "thin liquid+fundamental universe yields NaN positions on empty days",
    }
    experiment_id = f"performance-{trial.strategy_id}"
    register_experiment(
        ExperimentRecord(
            experiment_id=experiment_id, research_id=RESEARCH_ID,
            config={"experiment_type": "performance_audit",
                    "strategy_id": trial.strategy_id, "selection_relevant": False},
            dataset_version=DATASET_ID, experiment_type="diagnostic",
            status="completed", selection_relevant=False,
            result={"status": report["status"],
                    "cross_check": report["cross_check"]["status"]},
        ),
        ROOT,
    )
    directory = ROOT / "data" / "research" / RESEARCH_ID / "performance" / trial.strategy_id
    _write(directory / "canonical_metrics.json", report["canonical_metrics"])
    _write(directory / "pyfolio_metrics.json", report["pyfolio_metrics"])
    _write(ROOT / "data" / "research" / RESEARCH_ID
           / "performance_metric_definition.json", report["metric_definition"])
    return report


def _breadth_overlay(data, v2, close, locked_definition, factors_full,
                     factor_results, lock_row) -> dict[str, Any]:
    """Breadth OFF (locked) vs ON (exposure-scaled) over the broad liquid universe."""
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    pv = data["factors_price_volume"].copy()
    pv["ticker"] = pv["ticker"].astype(str)
    ma = pv.pivot_table(index="date", columns="ticker", values="ma_60", aggfunc="last")
    ma.index = pd.to_datetime(ma.index)
    ma = ma.reindex(index=close.index, columns=close.columns)
    eligible = (
        v2.assign(date=pd.to_datetime(v2["date"]), ticker=v2["ticker"].astype(str))
        .pivot_table(index="date", columns="ticker", values="liquidity_pass", aggfunc="last")
        .reindex(index=close.index, columns=close.columns)
        .fillna(False)
        .astype(bool)
    )
    breadth = compute_market_breadth(
        close_matrix=close, ma_matrix=ma, eligible_matrix=eligible,
        universe_status="PARTIAL",
    )
    exposed = breadth_to_exposure(
        breadth, threshold=BREADTH_THRESHOLD,
        exposure_high=BREADTH_HIGH, exposure_low=BREADTH_LOW,
    )
    risk_off_days = int((exposed["gross_exposure"] <= BREADTH_LOW).sum())
    total_days = int(len(exposed))

    chosen_factors = locked_definition.factor_ids
    close_factor = close[
        [t for t in close.columns if t in factors_full[chosen_factors[0]].columns]
    ]
    _close, weights_off, _r = build_strategy_targets(
        definition=locked_definition, root=ROOT,
        factor_matrices={fid: factors_full[fid] for fid in chosen_factors},
        close_matrix=close_factor, admission_results=factor_results,
    )
    calendar = build_rebalance_calendar(
        _close.index, frequency=locked_definition.rebalance_frequency,
        execution_lag_days=1,
    )
    weights_on = apply_gross_exposure(
        weights_off, market_breadth=exposed, rebalance_calendar=calendar
    )

    def _metrics(weights: pd.DataFrame) -> dict[str, Any]:
        _results, frame = run_weight_backtest(
            close_matrix=_close, portfolio_weights=weights,
            cost_model=_cost("base_cost"), initial_cash=INITIAL_CASH,
            top_n=TOP_N, use_vectorbt=False, allow_fallback=True,
        )
        m = frame.iloc[0]
        return {
            "cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
            "sortino": float(m["sortino"]), "mdd": float(m["max_drawdown"]),
            "calmar": float(m["calmar"]), "net_return": float(m["total_return"]),
            "turnover": float(m["turnover"]),
        }

    off = {"variant": "breadth_off", **_metrics(weights_off)}
    on = {"variant": "breadth_on", **_metrics(weights_on)}
    for record in (off, on):
        register_experiment(
            ExperimentRecord(
                experiment_id=f"breadth-{record['variant']}-{locked_definition.strategy_id}",
                research_id=RESEARCH_ID,
                config={"experiment_type": "breadth_overlay",
                        "variant": record["variant"], "selection_relevant": False},
                dataset_version=DATASET_ID, experiment_type="diagnostic",
                status="completed", selection_relevant=False, result=record,
            ),
            ROOT,
        )
    report = {
        "status": "AVAILABLE",
        "definition": "share of eligible-universe stocks with Close > MA60",
        "threshold": BREADTH_THRESHOLD,
        "exposure_high": BREADTH_HIGH,
        "exposure_low": BREADTH_LOW,
        "risk_off_days": risk_off_days,
        "risk_off_pct": round(risk_off_days / total_days, 4) if total_days else None,
        "rows": [off, on],
    }
    _write(ROOT / "data" / "research" / RESEARCH_ID / "breadth_report.json", report)
    return report


def _build_evidence(factor_results, admissible, performance, breadth_report,
                    validation_payload, statistics, strategy_evaluable) -> dict[str, Any]:
    if performance is not None:
        perf_status = performance["cross_check"]["status"]
        perf_status = "PASS" if perf_status in {
            "PASS", "PASS_WITH_DEFINITION_DIFFERENCE"
        } else "FAIL"
    else:
        perf_status = "UNAVAILABLE"
    oos_status = "PASS" if validation_payload.get("folds") else "UNAVAILABLE"
    return {
        "factor_evidence": {
            "status": "PASS" if admissible else "REJECT",
            "admitted_factors": sorted(admissible),
            "results": {fid: diag.to_dict() for fid, diag in factor_results.items()},
        },
        "strategy_incremental_value": {
            "status": "PASS" if strategy_evaluable else "UNAVAILABLE",
        },
        "execution": {"status": "PASS" if strategy_evaluable else "UNAVAILABLE"},
        "performance": {"status": perf_status, "report": performance},
        "attribution": {"status": "UNAVAILABLE"},
        "robustness": {"status": "UNAVAILABLE"},
        "oos": {"status": oos_status, "report": validation_payload},
        "statistics": statistics,
    }


def _write_report(research, summary, per_date, ablation_rows, stage2_rows, cost_rows,
                  lock_row, performance, breadth_report, validation_payload, statistics,
                  acceptance, included_tickers, factor_results) -> None:
    universe_counts = summary.get("universe_count", {})
    joint = summary.get("fundamental_joint_count", {})
    eps_c = summary.get("eps_valid_count", {})
    roe_c = summary.get("roe_valid_count", {})
    candidate_cfg = lock_row["net_definition"].to_config(
        directions={fid: factor_results[fid].direction for fid in lock_row["net_definition"].factor_ids},
        families={fid: "fundamental" for fid in lock_row["net_definition"].factor_ids},
        pit_required={fid: True for fid in lock_row["net_definition"].factor_ids},
        verdicts={fid: factor_results[fid].verdict for fid in lock_row["net_definition"].factor_ids},
        cost_model=_cost("base_cost"),
    ) if lock_row else {}

    factor_rows = [
        {
            "factor": fid, "family": "fundamental", "direction": diag.direction,
            "primary_horizon": PRIMARY_HORIZON, "verdict": diag.verdict,
        }
        for fid, diag in factor_results.items()
    ]
    validation_rows = [
        {"fold": f["fold"], "test_start": f["test_start"], "test_end": f["test_end"],
         "cagr": f["metrics"].get("cagr"), "sharpe": f["metrics"].get("sharpe"),
         "sortino": f["metrics"].get("sortino"), "mdd": f["metrics"].get("max_drawdown")}
        for f in validation_payload.get("folds", [])
    ]
    metric_def = (performance or {}).get("metric_definition", canonical_metric_metadata())

    model = ResearchReportModel(
        research=ReportSection(data={
            "research_id": research.research_id, "hypothesis": research.hypothesis,
            "universe": research.universe, "is_start": research.is_start,
            "is_end": research.is_end, "oos_start": research.oos_start,
            "oos_end": research.oos_end,
        }),
        dataset=ReportSection(data={"dataset_version": research.dataset_version}),
        factor_evidence=ReportSection(data={"rows": factor_rows}),
        strategy_trials=ReportSection(data={"rows": stage2_rows}),
        locked_candidate=ReportSection(data={"config": candidate_cfg}),
        execution_validation=ReportSection(status="UNAVAILABLE", data={}),
        performance=ReportSection(
            status="AVAILABLE" if performance else "UNAVAILABLE",
            data=performance or {},
        ),
        pyfolio=ReportSection(
            status="AVAILABLE" if performance else "UNAVAILABLE",
            data={"metrics": (performance or {}).get("pyfolio_metrics")},
        ),
        attribution=ReportSection(status="UNAVAILABLE", data={}),
        robustness=ReportSection(status="UNAVAILABLE", data={}),
        oos=ReportSection(status="UNAVAILABLE", data={}),
        statistics=ReportSection(data=statistics),
        acceptance=ReportSection(data=acceptance),
        reproducibility=ReportSection(data={"freeze_version": "research-freeze-v2"}),
        limitations=ReportSection(data={"items": [
            "Universe is current_listed_only (survivorship bias; no delisted history).",
            "Liquidity uses a close*volume proxy, not official TWSE traded value.",
            f"Fundamental coverage limits factor ranking to {len(included_tickers)} tickers.",
            "2024-2025 was already observed by v1; validation is walk-forward, not fresh OOS.",
            "Top-3 concentration is high relative to the broad liquid universe.",
        ]}),
        universe_coverage=ReportSection(data={
            "methodology": "listed -> tradable -> liquid -> fundamental_available (PIT)",
            "liquidity_rule": f"20D median traded value > TWD {LIQUIDITY_THRESHOLD:,}",
            "liquidity_measure_source": summary["liquidity_measure_source"],
            "mean_universe_count": universe_counts.get("mean"),
            "median_universe_count": universe_counts.get("median"),
            "min_universe_count": universe_counts.get("min"),
            "eps_coverage": eps_c.get("mean"),
            "roe_coverage": roe_c.get("mean"),
            "joint_coverage": joint.get("mean"),
            "survivorship": summary["survivorship"],
        }),
        factor_ablation=ReportSection(data={"rows": ablation_rows}),
        cost_turnover=ReportSection(data={"rows": [
            {k: v for k, v in row.items()
             if k not in {"net_trial", "net_definition", "factor_ids",
                          "selection_relevant"}}
            for row in cost_rows
        ]}),
        performance_metric_definition=ReportSection(
            status="AVAILABLE" if performance else "UNAVAILABLE", data=metric_def),
        breadth_overlay=ReportSection(
            status=breadth_report.get("status", "UNAVAILABLE"),
            data={k: v for k, v in breadth_report.items() if k != "status"}),
        validation=ReportSection(data={
            "label": validation_payload.get("label"),
            "reason": validation_payload.get("reason"),
            "warmup_note": validation_payload.get("warmup_note"),
            "rows": validation_rows,
        }),
    )
    directory = ROOT / "data" / "research" / RESEARCH_ID / "report"
    assert_write_allowed(directory / "research_report.md", ROOT).parent.mkdir(
        parents=True, exist_ok=True)
    (directory / "research_report.md").write_text(render_markdown(model), encoding="utf-8")
    (directory / "research_report.html").write_text(render_html(model), encoding="utf-8")
    _write(directory / "research_report_model.json", model.to_dict())


if __name__ == "__main__":
    main()

# ruff: noqa: E501

"""Final Strategy Validation v2 with the audited numerical parity contract."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance import engine_parity_audit as audit
from twse_factor_lab.acceptance.final_validation import (
    LOCKED_FINGERPRINT,
    STATISTICAL_THRESHOLD,
    final_verdict,
    freeze_three_folds,
    moving_block_bootstrap,
    sha256_json,
)
from twse_factor_lab.acceptance.psr import (
    daily_sharpe,
    deflated_sharpe_ratio,
    moments,
    probabilistic_sharpe_ratio,
)
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.governance.isolation import assert_write_allowed

ROOT_NAME = "final-strategy-validation-v2"
CANDIDATE_ID = "S3"
ATOL = 4.547473508864641e-12
RTOL = 1.4210854715202004e-14
PARITY_FORMULA = "abs(a - b) <= atol + rtol * max(abs(a), abs(b), 1)"
BASE_COST = CostModel()
STRESS_COST = CostModel(0.001425, 0.001425, 0.003, 0.002)
EXPECTED_CANDIDATE = {
    "strategy_id": "S3",
    "components": ["L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D"],
    "weights": {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5},
    "top_n": 5,
    "rebalance": "monthly",
    "buffer": False,
    "weighting": "equal_weight",
    "cost_model": "base_cost",
}
EXPECTED_COST_PARAMETERS = {
    "buy_fee_rate": 0.001425,
    "sell_fee_rate": 0.001425,
    "transaction_tax_rate": 0.003,
    "slippage_rate": 0.001,
}


class FinalValidationV2Error(ValueError):
    """A frozen v2 validation precondition or artifact was violated."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        return float(value) if np.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(root: Path, out: Path, name: str, value: Any) -> Path:
    path = assert_write_allowed(out / name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(root: Path, out: Path, name: str, frame: pd.DataFrame) -> Path:
    path = assert_write_allowed(out / name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _write_parquet(root: Path, out: Path, name: str, frame: pd.DataFrame) -> Path:
    path = assert_write_allowed(out / name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _assert_manifest(root: Path, base: Path, manifest: dict[str, Any], label: str) -> None:
    for name, expected in manifest.get("artifact_hashes", {}).items():
        path = base / name
        if not path.exists() or file_sha256(path) != expected:
            raise FinalValidationV2Error(f"{label}_HASH_MISMATCH:{name}")


def _assert_code_hashes(root: Path, manifest: dict[str, Any], label: str) -> None:
    for name, expected in manifest.get("code_sha", {}).items():
        path = root / name
        if not path.exists() or file_sha256(path) != expected:
            raise FinalValidationV2Error(f"{label}_CODE_HASH_MISMATCH:{name}")


def _assert_contract(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise FinalValidationV2Error("V2_CONTRACT_MUTATED")


def _versions() -> dict[str, str]:
    names = ("backtrader", "numpy", "pandas", "pyarrow", "pytest", "ruff", "vectorbt")
    return {
        name: importlib.metadata.version(name)
        for name in names
        if _package_version(name) is not None
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _assert_candidate(lock: dict[str, Any]) -> None:
    if lock.get("candidate_fingerprint") != LOCKED_FINGERPRINT:
        raise FinalValidationV2Error("CANDIDATE_FINGERPRINT_MISMATCH")
    candidate = lock.get("candidate", {})
    if any(candidate.get(key) != value for key, value in EXPECTED_CANDIDATE.items()):
        raise FinalValidationV2Error("CANDIDATE_FIELD_MISMATCH")
    if candidate.get("buffer_hold_rank") != 0:
        raise FinalValidationV2Error("CANDIDATE_BUFFER_MISMATCH")
    if candidate.get("cost_parameters") != EXPECTED_COST_PARAMETERS:
        raise FinalValidationV2Error("CANDIDATE_COST_MISMATCH")


def preflight(root: Path) -> dict[str, Any]:
    """Read and hash every frozen input before any v2 performance work."""
    root = root.resolve()
    c1 = root / "data/research/composite-factor-admission-v1"
    c2 = root / "data/research/composite-strategy-lab-v1"
    v1 = root / "data/research/final-strategy-validation-v1"
    audit_out = root / "data/research/engine-parity-audit-v1"
    audit_archive = root / "openspec/changes/archive/2026-09-16-audit-engine-parity-and-numerical-tolerance-v1"
    if not audit_archive.is_dir():
        raise FinalValidationV2Error("AUDIT_CHANGE_NOT_ARCHIVED")
    c2_manifest = _read_json(c2 / "run_manifest.json")
    c2_lock = _read_json(c2 / "candidate_lock.json")
    c2_input = _read_json(c2 / "strategy_input_contract.json")
    v1_manifest = _read_json(v1 / "run_manifest.json")
    v1_contract = _read_json(v1 / "final_validation_contract.json")
    v1_acceptance = _read_json(v1 / "final_acceptance.json")
    audit_manifest = _read_json(audit_out / "run_manifest.json")
    audit_contract = _read_json(audit_out / "engine_parity_audit_contract.json")
    audit_result = _read_json(audit_out / "engine_parity_audit_result.json")
    audit_recommendation = _read_json(audit_out / "recommended_parity_contract.json")
    _assert_candidate(c2_lock)
    if c2_input.get("composite_definition_sha") != v1_contract.get("factor_definition_sha"):
        raise FinalValidationV2Error("FACTOR_DEFINITION_MISMATCH")
    if c2_input.get("components") != EXPECTED_CANDIDATE["components"]:
        raise FinalValidationV2Error("STRATEGY_COMPONENT_MISMATCH")
    if c2_input.get("weights") != EXPECTED_CANDIDATE["weights"]:
        raise FinalValidationV2Error("STRATEGY_WEIGHT_MISMATCH")
    _assert_manifest(root, c1, {"artifact_hashes": c2_input["source_artifact_sha"]}, "CHANGE_1")
    _assert_manifest(root, c2, c2_manifest, "CHANGE_2")
    _assert_manifest(root, v1, v1_manifest, "FINAL_VALIDATION_V1")
    _assert_manifest(root, audit_out, audit_manifest, "ENGINE_AUDIT_V1")
    _assert_code_hashes(root, audit_manifest, "ENGINE_AUDIT_V1")
    if v1_acceptance.get("final_verdict") != "REJECT":
        raise FinalValidationV2Error("HISTORICAL_V1_VERDICT_CHANGED")
    if v1_acceptance.get("candidate_fingerprint") != LOCKED_FINGERPRINT:
        raise FinalValidationV2Error("HISTORICAL_V1_CANDIDATE_CHANGED")
    if v1_acceptance.get("gates", {}).get("engine_parity") != "FAIL":
        raise FinalValidationV2Error("HISTORICAL_V1_PARITY_EXPECTATION_CHANGED")
    if audit_result.get("classification") != "NUMERICAL_ONLY_DIVERGENCE":
        raise FinalValidationV2Error("AUDIT_CLASSIFICATION_NOT_QUALIFIED")
    if audit_result.get("economic_equivalence") is not True:
        raise FinalValidationV2Error("AUDIT_ECONOMIC_EQUIVALENCE_NOT_QUALIFIED")
    if audit_result.get("recommendation") != "READY_FOR_FINAL_VALIDATION_V2":
        raise FinalValidationV2Error("AUDIT_RECOMMENDATION_NOT_QUALIFIED")
    if audit_recommendation.get("recommended_atol") != ATOL:
        raise FinalValidationV2Error("AUDIT_ATOL_MISMATCH")
    if audit_recommendation.get("recommended_rtol") != RTOL:
        raise FinalValidationV2Error("AUDIT_RTOL_MISMATCH")
    if audit_recommendation.get("applied_to_this_change") is not False:
        raise FinalValidationV2Error("AUDIT_CONTRACT_WAS_APPLIED")
    if audit_contract.get("canonical_tolerance") != 1e-8:
        raise FinalValidationV2Error("LEGACY_TOLERANCE_CHANGED")
    v1_trials = _read_json(v1 / "trial_registry.json")
    c2_trials = _read_json(c2 / "strategy_trial_registry.json")
    strategy_trials = c2_trials.get("strategy_selection_trials", [])
    population = [
        row for row in v1_trials.get("trials", []) if row.get("dsr_selection_population")
    ]
    if (
        len(strategy_trials) != 4
        or c2_trials.get("strategy_selection_trial_count") != 4
        or v1_trials.get("strategy_trial_count") != 4
        or len(population) != 4
        or any(
            not (
                row.get("category") == "strategy_selection"
                and row.get("terminal_trial") is True
                and row.get("selection_relevant") is True
            )
            for row in population
        )
    ):
        raise FinalValidationV2Error("DSR_POPULATION_MISMATCH")
    split_path = v1 / "walk_forward_splits.json"
    if file_sha256(split_path) != v1_manifest["walk_forward_split_sha"]:
        raise FinalValidationV2Error("V1_SPLIT_HASH_MISMATCH")
    return {
        "root": root,
        "c1": c1,
        "c2": c2,
        "v1": v1,
        "audit_out": audit_out,
        "c1_artifact_sha": c2_input["source_artifact_sha"],
        "c2_artifact_sha": c2_manifest["artifact_hashes"],
        "v1_artifact_sha": v1_manifest["artifact_hashes"],
        "audit_artifact_sha": audit_manifest["artifact_hashes"],
        "c2_manifest": c2_manifest,
        "c2_lock": c2_lock,
        "c2_input": c2_input,
        "v1_manifest": v1_manifest,
        "v1_contract": v1_contract,
        "v1_acceptance": v1_acceptance,
        "audit_manifest": audit_manifest,
        "audit_contract": audit_contract,
        "audit_result": audit_result,
        "audit_recommendation": audit_recommendation,
        "v1_split_sha": v1_manifest["walk_forward_split_sha"],
        "v1_contract_sha": v1_manifest["final_validation_contract_sha"],
        "audit_result_sha": file_sha256(audit_out / "engine_parity_audit_result.json"),
        "audit_recommendation_sha": file_sha256(audit_out / "recommended_parity_contract.json"),
        "factor_definition_sha": v1_contract["factor_definition_sha"],
        "strategy_input_sha": c2_manifest["strategy_input_contract_sha"],
        "cost_sha": sha256_json(BASE_COST.summary()),
    }


def _contract(state: dict[str, Any]) -> dict[str, Any]:
    v1 = state["v1_contract"]
    return {
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "historical_v1_verdict": "REJECT",
        "historical_v1_modified": False,
        "v1_validation_contract_sha": state["v1_contract_sha"],
        "v1_split_sha": state["v1_split_sha"],
        "audit_result_sha": state["audit_result_sha"],
        "audit_recommended_contract_sha": state["audit_recommendation_sha"],
        "audit_classification": "NUMERICAL_ONLY_DIVERGENCE",
        "economic_equivalence": True,
        "factor_definition_sha": state["factor_definition_sha"],
        "strategy_input_sha": state["strategy_input_sha"],
        "component_factor_sha": state["c2_input"]["component_factor_sha"],
        "top_n": 5,
        "rebalance": "monthly",
        "buffer": False,
        "weighting": "equal_weight",
        "execution_semantics": "signal_T_to_next_valid_trading_session_T_plus_1",
        "base_cost": BASE_COST.summary(),
        "stress_cost": STRESS_COST.summary(),
        "walk_forward_contract": v1["walk_forward"],
        "bootstrap_contract": v1["bootstrap"],
        "cost_stress_contract": {
            "scenario_count": 1,
            "base_slippage": 0.001,
            "stress_slippage": 0.002,
            "other_costs_unchanged": True,
        },
        "breadth_contract": v1["breadth"],
        "psr_threshold": v1["psr_dsr"]["threshold"],
        "dsr_threshold": v1["psr_dsr"]["threshold"],
        "dsr_population": {
            "strategy_trial_count": 4,
            "excluded": [
                "factor_trials",
                "composite_gate",
                "walk_forward_folds",
                "bootstrap_draws",
                "breadth",
                "cost_stress",
                "subgroups",
                "pyfolio",
            ],
        },
        "final_acceptance_rule": v1["acceptance_rule"],
        "parity_contract": {
            "type": "absolute_plus_relative",
            "formula": PARITY_FORMULA,
            "atol": ATOL,
            "rtol": RTOL,
            "scale": "max(abs(a), abs(b), 1)",
        },
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
        "strategy_research_performed": False,
    }


def _canonical_outputs(root: Path, out: Path, state: dict[str, Any], contract_path: Path, contract_sha: str) -> dict[str, Any]:
    v1 = importlib.import_module("run_final_strategy_validation_v1")

    returns_frame, positions, turnover, exposure = v1._aligned_base()
    returns = returns_frame["returns"]
    close = pd.read_parquet(root / "data/processed/close_matrix.parquet")
    volume = pd.read_parquet(root / "data/processed/volume_matrix.parquet")
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    close.columns = close.columns.astype(str)
    volume.columns = volume.columns.astype(str)
    frozen_splits = _read_json(state["v1"] / "walk_forward_splits.json")
    folds = frozen_splits["folds"]
    if len(folds) != 3 or folds != freeze_three_folds(returns.index, warmup_sessions=252):
        raise FinalValidationV2Error("TEMPORAL_SPLIT_MISMATCH")
    _assert_contract(contract_path, contract_sha)
    fold_frame, temporal = v1._fold_metrics(returns, turnover, exposure, folds)
    _write_csv(root, out, "walk_forward_results.csv", fold_frame)
    _write_json(
        root,
        out,
        "walk_forward_summary.json",
        {**temporal, "temporal_validation": temporal["status"], "fresh_oos": False},
    )
    _assert_contract(contract_path, contract_sha)
    bootstrap = moving_block_bootstrap(returns)
    _write_json(root, out, "bootstrap_summary.json", bootstrap)
    _assert_contract(contract_path, contract_sha)
    slices, slices_report = v1._slices(returns, positions, turnover, exposure, close, volume)
    _write_csv(root, out, "robustness_slices.csv", slices)
    _write_json(root, out, "robustness_slices_report.json", slices_report)
    _assert_contract(contract_path, contract_sha)
    targets = v1._rebuild_targets(close, volume)
    stress_result, _ = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=targets,
        cost_model=STRESS_COST,
        initial_cash=1_000_000.0,
        top_n=5,
        use_vectorbt=False,
        allow_fallback=False,
    )
    base_metrics = compute_metrics(returns, turnover=turnover, exposure=exposure)
    stress_metrics = compute_metrics(
        stress_result["returns"],
        turnover=stress_result["turnover"],
        exposure=stress_result["exposure"],
    )
    gross = float((1.0 + stress_result["gross_returns"].fillna(0.0)).prod() - 1.0)
    trials = _read_json(state["c2"] / "strategy_trial_registry.json")["strategy_selection_trials"]
    if len(trials) != 4 or any(
        not (
            row.get("trial_stage") == "strategy_selection"
            and row.get("terminal_trial") is True
            and row.get("selection_relevant") is True
        )
        for row in trials
    ):
        raise FinalValidationV2Error("DSR_POPULATION_MISMATCH")
    base_trial = next(row for row in trials if row["strategy_id"] == CANDIDATE_ID)
    cost = pd.DataFrame(
        [
            {
                "scenario": "Base",
                **base_metrics,
                "cost_drag": float(base_trial["cost_drag"]),
                "slippage": 0.001,
            },
            {
                "scenario": "Stress",
                **stress_metrics,
                "cost_drag": gross - float(stress_metrics["total_return"]),
                "slippage": 0.002,
            },
        ]
    )
    cost["status"] = np.where((cost["cagr"] > 0) & (cost["sharpe"] > 0), "PASS", "FAIL")
    cost["strategy_id"] = CANDIDATE_ID
    cost["candidate_fingerprint"] = LOCKED_FINGERPRINT
    cost["portfolio_unchanged"] = True
    cost_status = "PASS" if stress_metrics["cagr"] > 0 and stress_metrics["sharpe"] > 0 else "FAIL"
    _write_csv(root, out, "cost_stress_comparison.csv", cost)
    breadth = pd.read_csv(state["c2"] / "breadth_comparison.csv")
    breadth["candidate_changed"] = False
    breadth["diagnostic_post_lock"] = True
    breadth["candidate_fingerprint"] = LOCKED_FINGERPRINT
    _write_csv(root, out, "breadth_sensitivity.csv", breadth)
    trial_registry = _read_json(state["v1"] / "trial_registry.json")
    _write_json(root, out, "trial_registry.json", trial_registry)
    _assert_contract(contract_path, contract_sha)
    trial_sharpes = [float(row["net_sharpe"]) / np.sqrt(252) for row in trials]
    observed = daily_sharpe(returns)
    skewness, kurtosis = moments(returns)
    psr = probabilistic_sharpe_ratio(observed, len(returns), 0.0, skewness, kurtosis - 3.0)
    dsr, sr0 = deflated_sharpe_ratio(observed, len(returns), trial_sharpes)
    stats = {
        "psr": psr,
        "dsr": dsr,
        "sr0_daily": sr0,
        "observed_sharpe_daily": observed,
        "sample_size": len(returns),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "benchmark_sharpe": 0.0,
        "strategy_trial_count": 4,
        "factor_trials_excluded": True,
        "status": "PASS" if psr >= STATISTICAL_THRESHOLD and dsr >= STATISTICAL_THRESHOLD else "FAIL",
        "threshold": STATISTICAL_THRESHOLD,
    }
    _write_json(root, out, "psr_dsr_report.json", stats)
    full_metrics = compute_metrics(returns, turnover=turnover, exposure=exposure)
    correlation = _read_json(state["c1"] / "factor_redundancy_report.json")["pairs"][0]["median_rank_correlation"]
    return {
        "returns": returns,
        "positions": positions,
        "turnover": turnover,
        "exposure": exposure,
        "close": close,
        "volume": volume,
        "fold_frame": fold_frame,
        "temporal": temporal,
        "bootstrap": bootstrap,
        "slices": slices,
        "slices_report": slices_report,
        "cost": cost,
        "cost_status": cost_status,
        "breadth": breadth,
        "stats": stats,
        "risk": {
            "max_drawdown": full_metrics["max_drawdown"],
            "high_drawdown_risk": True,
            "l2_l4_correlation": correlation,
            "near_high_redundancy_risk": True,
        },
    }


def _numeric_row(left: np.ndarray, right: np.ndarray) -> dict[str, float | bool]:
    difference = np.abs(left - right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
    limit = ATOL + RTOL * scale
    ratio = difference / limit
    return {
        "max_abs_error": float(np.max(difference)),
        "max_rel_error": float(np.max(difference / scale)),
        "max_scaled_error_ratio": float(np.max(ratio)),
        "mean_abs_error": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "passed": bool(np.all(difference <= limit)),
    }


def _parity_detail(
    traces: dict[str, pd.DataFrame],
    fields: tuple[str, ...],
    daily_fields: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs = (("CUSTOM", "VECTORBT"), ("CUSTOM", "BACKTRADER"), ("VECTORBT", "BACKTRADER"))
    detail: list[dict[str, Any]] = []
    pair_summary: dict[str, dict[str, Any]] = {}
    for left, right in pairs:
        pair_name = f"{left.lower()}_vs_{right.lower()}"
        pair_rows: list[dict[str, Any]] = []
        for field in fields:
            left_frame = traces[left].sort_values(["date", "ticker"])
            right_frame = traces[right].sort_values(["date", "ticker"])
            if not left_frame[["date", "ticker"]].reset_index(drop=True).equals(
                right_frame[["date", "ticker"]].reset_index(drop=True)
            ):
                raise FinalValidationV2Error("ENGINE_TRACE_KEY_MISMATCH")
            metric = _numeric_row(
                left_frame[field].to_numpy(dtype=float), right_frame[field].to_numpy(dtype=float)
            )
            row = {"pair": pair_name.upper(), "field": field, **metric, "final_abs_error": None, "final_rel_error": None}
            detail.append(row)
            pair_rows.append(row)
        for field in daily_fields:
            left_frame = traces[left].drop_duplicates("date").sort_values("date")
            right_frame = traces[right].drop_duplicates("date").sort_values("date")
            if not left_frame["date"].reset_index(drop=True).equals(right_frame["date"].reset_index(drop=True)):
                raise FinalValidationV2Error("ENGINE_DAILY_KEY_MISMATCH")
            left_values = left_frame[field].to_numpy(dtype=float)
            right_values = right_frame[field].to_numpy(dtype=float)
            metric = _numeric_row(left_values, right_values)
            scale = max(abs(float(left_values[-1])), abs(float(right_values[-1])), 1.0)
            row = {
                "pair": pair_name.upper(),
                "field": field,
                **metric,
                "final_abs_error": float(abs(left_values[-1] - right_values[-1])),
                "final_rel_error": float(abs(left_values[-1] - right_values[-1]) / scale),
            }
            detail.append(row)
            pair_rows.append(row)
        pair_summary[pair_name] = {
            "max_abs_error": max(float(row["max_abs_error"]) for row in pair_rows),
            "max_rel_error": max(float(row["max_rel_error"]) for row in pair_rows),
            "max_scaled_error_ratio": max(float(row["max_scaled_error_ratio"]) for row in pair_rows),
            "final_equity_diff": next(row for row in pair_rows if row["field"] == "equity")["final_abs_error"],
            "parity_status": "PASS" if all(bool(row["passed"]) for row in pair_rows) else "FAIL",
        }
    return pd.DataFrame(detail), pair_summary


def _run_engine_parity(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    audit_state = audit.preflight(root)
    inputs = audit.load_replay_inputs(root, audit_state)
    with TemporaryDirectory(prefix=".v2-engine-replay-", dir=root / "data/research") as temp:
        replay = audit.run_engines(root, inputs, Path(temp))
        traces = replay["traces"]
        dates, selection = audit.compare_rebalance_and_selection(inputs, traces, replay["orders"])
        target_frame, targets = audit.compare_targets(inputs, traces)
        execution_frame, execution = audit._order_compare(replay["orders"], traces, inputs)
        cost_frame, costs = audit._cost_compare(replay["orders"], inputs)
        position_frame, positions = audit._position_compare(traces)
        cash_frame, cash = audit._daily_compare(traces, "cash", "cash")
        return_frame, returns = audit._daily_compare(traces, "daily_return", "return")
        equity_frame, equity = audit._daily_compare(traces, "equity", "equity")
    cash_dates = {
        tuple(pd.to_datetime(traces[engine]["date"]).drop_duplicates().sort_values())
        for engine in audit.ENGINE_ORDER
    }
    cash_dates_match = len(cash_dates) == 1
    semantic_checks = {
        "rebalance_dates": dates["status"] == "PASS",
        "selection": selection.empty or bool(selection["match"].all()),
        "target_allocations": targets.get("material_difference_count", 0) == 0,
        "directions": execution.get("material_difference_count", 0) == 0,
        "target_replacement": bool(execution.get("complete_target_replacement")),
        "removed_ticker_target_zero": bool(execution.get("removed_ticker_target_zero")),
        "t_plus_1": bool(execution.get("t_to_t_plus_1")),
        "cost_semantics": costs.get("material_difference_count", 0) == 0,
        "cash_semantics": cash_dates_match,
    }
    semantic_status = "PASS" if all(semantic_checks.values()) else "FAIL"
    detail, pair_summary = _parity_detail(
        traces,
        ("position_value",),
        ("cash", "daily_return", "equity"),
    )
    cash_rows = detail[detail["field"] == "cash"]
    numeric_status = "PASS" if bool(detail["passed"].all()) else "FAIL"
    cash_semantic_parity = cash_dates_match and semantic_status == "PASS"
    final_diffs = {
        pair: values["final_equity_diff"] for pair, values in pair_summary.items()
    }
    parity = {
        "semantic_parity": {"checks": semantic_checks, "status": semantic_status},
        "cash_semantic_parity": cash_semantic_parity,
        "cash_numerical_parity": bool(cash_rows["passed"].all()),
        "numerical_parity": {
            "status": numeric_status,
            "fields": sorted(detail["field"].unique().tolist()),
            "diagnostic_fields": ["position_quantity", "executed_weight"],
        },
        "custom_vs_vectorbt": pair_summary["custom_vs_vectorbt"],
        "custom_vs_backtrader": pair_summary["custom_vs_backtrader"],
        "vectorbt_vs_backtrader": pair_summary["vectorbt_vs_backtrader"],
        "input_sha": inputs["engine_input_sha"],
        "actual_engines": replay["actual_engines"],
        "atol": ATOL,
        "rtol": RTOL,
        "formula": PARITY_FORMULA,
        "max_abs_error": float(detail["max_abs_error"].max()),
        "max_rel_error": float(detail["max_rel_error"].max()),
        "max_scaled_error_ratio": float(detail["max_scaled_error_ratio"].max()),
        "final_equity_diff": final_diffs,
        "parity_status": "PASS"
        if semantic_status == "PASS" and numeric_status == "PASS"
        else "FAIL",
        "legacy_absolute_only_cash_status": cash["status"],
        "cash_numeric_max_abs_error": float(cash["max_abs_error"]),
    }
    return {
        "parity": parity,
        "detail": detail,
        "traces": traces,
        "raw": {
            "dates": dates,
            "selection": selection,
            "targets": targets,
            "execution": execution,
            "costs": costs,
            "positions": positions,
            "cash": cash,
            "returns": returns,
            "equity": equity,
        },
    }


def _delta(state: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    c = state["c2_lock"]["candidate"]
    checks = {
        "strategy_changed": c.get("strategy_id") != CANDIDATE_ID,
        "candidate_changed": state["c2_lock"].get("candidate_fingerprint") != LOCKED_FINGERPRINT,
        "factor_changed": state["factor_definition_sha"] != contract["factor_definition_sha"],
        "weights_changed": c.get("weights") != EXPECTED_CANDIDATE["weights"],
        "top_n_changed": c.get("top_n") != 5,
        "rebalance_changed": c.get("rebalance") != "monthly",
        "buffer_changed": c.get("buffer") is not False,
        "cost_changed": c.get("cost_parameters") != EXPECTED_COST_PARAMETERS,
        "walk_forward_changed": contract["walk_forward_contract"] != state["v1_contract"]["walk_forward"],
        "bootstrap_changed": contract["bootstrap_contract"] != state["v1_contract"]["bootstrap"],
        "psr_changed": contract["psr_threshold"] != state["v1_contract"]["psr_dsr"]["threshold"],
        "dsr_threshold_changed": contract["dsr_threshold"] != state["v1_contract"]["psr_dsr"]["threshold"],
        "dsr_population_changed": contract["dsr_population"]["strategy_trial_count"] != 4,
        "acceptance_logic_changed": contract["final_acceptance_rule"] != state["v1_contract"]["acceptance_rule"],
        "engine_numerical_parity_contract_changed": True,
    }
    if any(value for name, value in checks.items() if name != "engine_numerical_parity_contract_changed"):
        raise FinalValidationV2Error("PROHIBITED_V1_V2_DELTA")
    return {**checks, "only_allowed_change": True, "historical_v1_modified": False}


def _report(
    contract_sha: str,
    state: dict[str, Any],
    canonical: dict[str, Any],
    parity: dict[str, Any],
    delta: dict[str, Any],
    acceptance: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    temporal = canonical["temporal"]
    bootstrap = canonical["bootstrap"]
    cost = canonical["cost"]
    risk = canonical["risk"]
    gates = acceptance["gates"]
    baseline = manifest["baseline"]
    return f"""# Final Strategy Validation v2

## A. Baseline

pytest = {baseline.get("pytest", "NOT_RUN")}  
ruff = {baseline.get("ruff", "NOT_RUN")}  
OpenSpec = {baseline.get("openspec", "NOT_RUN")}  
RC1 = {baseline.get("rc1", "NOT_RUN")}  
historical freezes = PASS

## B. Candidate

strategy = S3  
fingerprint = {LOCKED_FINGERPRINT}  
immutable = YES  
components = L2_AMIHUD_20D 0.50 + L4_DOLLAR_VOLUME_20D 0.50  
TopN = 5  
rebalance = MONTHLY  
buffer = OFF  
cost = Base Cost

## C. Historical v1

verdict = REJECT  
reason = legacy absolute-only parity failure  
modified = NO  
canonical tolerance = 1e-8

## D. Engine Audit

classification = NUMERICAL_ONLY_DIVERGENCE  
economic equivalence = TRUE  
recommendation = READY_FOR_FINAL_VALIDATION_V2

## E. V2 Parity Contract

atol = {ATOL}  
rtol = {RTOL}  
formula = {PARITY_FORMULA}  
contract SHA = {contract_sha}

## F. Engine Parity v2

Semantic parity = {parity["semantic_parity"]["status"]}  
Custom vs Vectorbt = {parity["custom_vs_vectorbt"]["parity_status"]}  
Custom vs Backtrader = {parity["custom_vs_backtrader"]["parity_status"]}  
Vectorbt vs Backtrader = {parity["vectorbt_vs_backtrader"]["parity_status"]}  
cash semantic parity = {parity["cash_semantic_parity"]}  
cash numerical parity = {parity["cash_numerical_parity"]}  
max abs error = {parity["max_abs_error"]}  
max rel error = {parity["max_rel_error"]}  
max scaled error ratio = {parity["max_scaled_error_ratio"]}  
status = {parity["parity_status"]}

## G. Temporal Validation

folds = 3  
positive return folds = {temporal["positive_return_fold_count"]}/3  
positive Sharpe folds = {temporal["positive_sharpe_fold_count"]}/3  
median Sharpe = {temporal["median_fold_sharpe"]}  
worst Sharpe = {temporal["worst_fold_sharpe"]}  
aggregate return = {temporal["aggregate_validation_return"]}  
status = {gates["temporal_validation"]}

## H. Bootstrap

P(Sharpe > 0) = {bootstrap["p_sharpe_positive"]}  
P(CAGR > 0) = {bootstrap["p_cagr_positive"]}  
block length = 20; draws = 2000; seed = 42  
status = {gates["bootstrap"]}

## I. Cost Stress

stress slippage = 0.002  
stress CAGR = {float(cost.loc[cost["scenario"] == "Stress", "cagr"].iloc[0])}  
stress Sharpe = {float(cost.loc[cost["scenario"] == "Stress", "sharpe"].iloc[0])}  
status = {gates["cost_stress"]}

## J. Robustness / Breadth

liquidity = reused v1 slices  
industry = INDUSTRY_SLICE_UNAVAILABLE when unchanged  
market regime = MA60 breadth threshold 0.40  
breadth candidate changed = NO

## K. PSR / DSR

PSR = {canonical["stats"]["psr"]}  
DSR = {canonical["stats"]["dsr"]}  
strategy trial count = 4  
factor/gate/fold/bootstrap/breadth/stress/subgroup/Pyfolio trials = EXCLUDED

## L. Risk

MDD = {risk["max_drawdown"]}  
HIGH_DRAWDOWN_RISK = YES  
L2/L4 correlation = {risk["l2_l4_correlation"]}  
NEAR_HIGH_REDUNDANCY_RISK = YES

## M. V1 -> V2 Delta

Strategy changed = {"YES" if delta["strategy_changed"] else "NO"}  
Validation methodology changed = NO  
Statistical contract changed = NO  
Parity numerical contract changed = YES

## N. Final Verdict

Temporal = {gates["temporal_validation"]}  
Bootstrap = {gates["bootstrap"]}  
Cost = {gates["cost_stress"]}  
Statistics = {gates["statistical"]}  
Engine = {gates["engine_parity"]}  
Reproducibility = {gates["reproducibility"]}  
FINAL VERDICT = {acceptance["final_verdict"]}

## O. Interpretation

acceptance label = {acceptance["acceptance_label"]}  
fresh OOS available = NO  
fresh OOS claimed = NO  
production ready = NO

## P. Governance

v1 REJECT preserved = YES  
audit preserved = YES  
historical v1 modified = false  
audit artifacts modified = NO

## Q. Git / Archive

commit = NO  
push = NO  
tag = NO  
archive = NO  
READY_FOR_REVIEW = YES
"""


def run_validation(root: str | Path, baseline: dict[str, str] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    out = root / "data/research" / ROOT_NAME
    out.mkdir(parents=True, exist_ok=True)
    if baseline and any(value != "PASS" for value in baseline.values()):
        raise FinalValidationV2Error("HARD_GATE_PRECONDITION_FAILED")
    state = preflight(root)
    contract = _contract(state)
    contract_path = _write_json(root, out, "final_validation_v2_contract.json", contract)
    contract_sha = file_sha256(contract_path)
    _assert_contract(contract_path, contract_sha)
    candidate_verification = {
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "historical_v1_verdict": "REJECT",
        "historical_v1_modified": False,
        "factor_definition_sha": state["factor_definition_sha"],
        "strategy_input_sha": state["strategy_input_sha"],
        "immutable_fields_verified": True,
        "immutable_fields": EXPECTED_CANDIDATE,
        "cost_parameters": EXPECTED_COST_PARAMETERS,
    }
    _write_json(root, out, "candidate_verification.json", candidate_verification)
    delta = _delta(state, contract)
    _write_json(root, out, "validation_v1_v2_delta.json", delta)
    _assert_contract(contract_path, contract_sha)
    canonical = _canonical_outputs(root, out, state, contract_path, contract_sha)
    parity_result = _run_engine_parity(root, state)
    parity = parity_result["parity"]
    detail = parity_result["detail"]
    for engine, trace in parity_result["traces"].items():
        _write_parquet(root, out, f"{engine.lower()}_trace.parquet", trace)
    _write_json(root, out, "engine_parity_v2.json", parity)
    _write_csv(root, out, "engine_parity_v2_detail.csv", detail)
    _assert_contract(contract_path, contract_sha)
    if any(
        state["audit_out"].joinpath(name).exists()
        and file_sha256(state["audit_out"] / name) != expected
        for name, expected in state["audit_artifact_sha"].items()
    ):
        raise FinalValidationV2Error("AUDIT_ARTIFACT_MUTATED")
    gates = {
        "temporal_validation": "PASS" if canonical["temporal"]["status"] == "PASS" else "FAIL",
        "bootstrap": canonical["bootstrap"]["status"],
        "cost_stress": canonical["cost_status"],
        "statistical": canonical["stats"]["status"],
        "engine_parity": parity["parity_status"],
        "fingerprint": "PASS",
        "reproducibility": "PASS",
    }
    verdict = final_verdict(gates)
    acceptance = {
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "historical_v1_verdict": "REJECT",
        "historical_v1_modified": False,
        "audit_classification": "NUMERICAL_ONLY_DIVERGENCE",
        "economic_equivalence": True,
        "parity_contract": {
            "type": "absolute_plus_relative",
            "atol": ATOL,
            "rtol": RTOL,
            "formula": PARITY_FORMULA,
        },
        "temporal_validation": gates["temporal_validation"],
        "bootstrap_status": gates["bootstrap"],
        "cost_stress_status": gates["cost_stress"],
        "psr": canonical["stats"]["psr"],
        "dsr": canonical["stats"]["dsr"],
        "strategy_trial_count": 4,
        "engine_parity_status": gates["engine_parity"],
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
        "high_drawdown_risk": True,
        "near_high_redundancy_risk": True,
        "final_verdict": verdict,
        "acceptance_label": "RESEARCH_ACCEPTED_WITHOUT_FRESH_OOS"
        if verdict == "ACCEPT"
        else "CANDIDATE_WITHOUT_FRESH_OOS"
        if verdict == "CANDIDATE"
        else "RESEARCH_REJECTED",
        "gates": gates,
        "limitations": [
            "historical v1 selection is not untouched OOS",
            "MDD near -50%",
            "L2/L4 near-high redundancy",
            "industry slice unavailable",
        ],
    }
    _write_json(root, out, "final_acceptance_v2.json", acceptance)
    baseline = baseline or {
        "pytest": "NOT_RUN",
        "ruff": "NOT_RUN",
        "openspec": "NOT_RUN",
        "rc1": "NOT_RUN",
    }
    manifest = {
        "manifest_self_hash_excluded": True,
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "change_1_artifact_sha": state["c1_artifact_sha"],
        "change_2_artifact_sha": state["c2_artifact_sha"],
        "final_validation_v1_artifact_sha": state["v1_artifact_sha"],
        "engine_parity_audit_v1_artifact_sha": state["audit_artifact_sha"],
        "validation_v2_contract_sha": contract_sha,
        "parity_contract_sha": state["audit_recommendation_sha"],
        "walk_forward_split_sha": state["v1_split_sha"],
        "factor_definition_sha": state["factor_definition_sha"],
        "strategy_input_sha": state["strategy_input_sha"],
        "ohlcv_sha": state["audit_manifest"]["ohlcv_sha"],
        "universe_sha": state["audit_manifest"]["universe_sha"],
        "cost_sha": state["cost_sha"],
        "trial_registry_sha": file_sha256(out / "trial_registry.json"),
        "engine_input_sha": parity["input_sha"],
        "engine_input_sha_by_engine": {
            engine: parity["input_sha"] for engine in parity_result["traces"]
        },
        "trace_hashes": {
            f"{engine.lower()}_trace.parquet": file_sha256(
                out / f"{engine.lower()}_trace.parquet"
            )
            for engine in parity_result["traces"]
        },
        "actual_engine_labels": parity["actual_engines"],
        "engine_versions": _versions(),
        "python_version": platform.python_version(),
        "dependency_snapshot": _versions(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "code_sha": {
            "run_final_strategy_validation_v2.py": file_sha256(root / "run_final_strategy_validation_v2.py"),
            "src/twse_factor_lab/acceptance/final_validation_v2.py": file_sha256(root / "src/twse_factor_lab/acceptance/final_validation_v2.py"),
        },
        "historical_freezes": state["v1_manifest"]["historical_freezes"],
        "baseline": baseline,
        "final_verdict": verdict,
        "acceptance_label": acceptance["acceptance_label"],
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
        "artifact_hashes": {},
    }
    report = _report(contract_sha, state, canonical, parity, delta, acceptance, manifest)
    report_path = assert_write_allowed(out / "final_validation_v2_report.md", root)
    report_path.write_text(report, encoding="utf-8")
    generated = sorted(path for path in out.iterdir() if path.is_file() and path.name != "run_manifest.json")
    manifest["artifact_hashes"] = {path.name: file_sha256(path) for path in generated}
    _write_json(root, out, "run_manifest.json", manifest)
    for name, expected in manifest["artifact_hashes"].items():
        if file_sha256(out / name) != expected:
            raise FinalValidationV2Error(f"V2_ARTIFACT_HASH_MISMATCH:{name}")
    if "run_manifest.json" in manifest["artifact_hashes"]:
        raise FinalValidationV2Error("V2_MANIFEST_SELF_HASH_NOT_EXCLUDED")
    _assert_manifest(
        root,
        state["v1"],
        {"artifact_hashes": state["v1_artifact_sha"]},
        "FINAL_VALIDATION_V1",
    )
    _assert_manifest(
        root,
        state["c1"],
        {"artifact_hashes": state["c1_artifact_sha"]},
        "CHANGE_1",
    )
    _assert_manifest(
        root,
        state["c2"],
        {"artifact_hashes": state["c2_artifact_sha"]},
        "CHANGE_2",
    )
    _assert_manifest(root, state["audit_out"], {"artifact_hashes": state["audit_artifact_sha"]}, "ENGINE_AUDIT_V1")
    return {
        "output": str(out),
        "contract_sha": contract_sha,
        "parity": parity,
        "acceptance": acceptance,
        "manifest": manifest,
    }


__all__ = [
    "ATOL",
    "BASE_COST",
    "CANDIDATE_ID",
    "FinalValidationV2Error",
    "LOCKED_FINGERPRINT",
    "PARITY_FORMULA",
    "RTOL",
    "file_sha256",
    "preflight",
    "run_validation",
]

"""Contract checks for the frozen S3 final-validation v2 replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance.final_validation import final_verdict
from twse_factor_lab.acceptance.final_validation_v2 import (
    ATOL,
    LOCKED_FINGERPRINT,
    PARITY_FORMULA,
    RTOL,
    _numeric_row,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/research/final-strategy-validation-v2"
V1 = ROOT / "data/research/final-strategy-validation-v1"
AUDIT = ROOT / "data/research/engine-parity-audit-v1"
TRACE_COLUMNS = [
    "date",
    "engine",
    "signal_date",
    "execution_date",
    "ticker",
    "selected",
    "target_weight",
    "executed_weight",
    "position_quantity",
    "position_value",
    "buy_notional",
    "sell_notional",
    "buy_fee",
    "sell_fee",
    "sell_tax",
    "slippage_cost",
    "total_cost",
    "cash",
    "gross_exposure",
    "net_exposure",
    "daily_pnl",
    "daily_return",
    "equity",
]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_historical_audit_candidate_and_strategy_freezes_are_unchanged() -> None:
    acceptance = _json(OUT / "final_acceptance_v2.json")
    contract = _json(OUT / "final_validation_v2_contract.json")
    audit = _json(AUDIT / "engine_parity_audit_result.json")
    v1 = _json(V1 / "final_acceptance.json")
    candidate = _json(OUT / "candidate_verification.json")

    assert v1["final_verdict"] == acceptance["historical_v1_verdict"] == "REJECT"
    assert v1["candidate_fingerprint"] == LOCKED_FINGERPRINT
    assert acceptance["historical_v1_modified"] is False
    assert (
        audit["classification"]
        == contract["audit_classification"]
        == "NUMERICAL_ONLY_DIVERGENCE"
    )
    assert audit["economic_equivalence"] is True
    assert audit["recommendation"] == "READY_FOR_FINAL_VALIDATION_V2"
    assert candidate["candidate_fingerprint"] == LOCKED_FINGERPRINT
    assert candidate["immutable_fields_verified"] is True
    assert candidate["immutable_fields"] == {
        "strategy_id": "S3",
        "components": ["L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D"],
        "weights": {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5},
        "top_n": 5,
        "rebalance": "monthly",
        "buffer": False,
        "weighting": "equal_weight",
        "cost_model": "base_cost",
    }


def test_v2_parity_contract_is_exact_and_scale_aware() -> None:
    contract = _json(OUT / "final_validation_v2_contract.json")
    parity = contract["parity_contract"]
    assert parity["atol"] == ATOL == 4.547473508864641e-12
    assert parity["rtol"] == RTOL == 1.4210854715202004e-14
    assert parity["formula"] == PARITY_FORMULA
    assert parity["scale"] == "max(abs(a), abs(b), 1)"

    left = np.array([1.0, 1_000_000.0])
    right = np.array([1.0 + ATOL, 1_000_000.0 + RTOL * 1_000_000.0])
    result = _numeric_row(left, right)
    scale = np.maximum(np.maximum(abs(left), abs(right)), 1.0)
    difference = abs(left - right)
    limit = ATOL + RTOL * scale
    assert result["max_abs_error"] == float(difference.max())
    assert result["max_scaled_error_ratio"] == float(np.max(difference / limit))
    assert result["passed"] is True


def test_v2_semantic_and_cash_parity_are_separate() -> None:
    parity = _json(OUT / "engine_parity_v2.json")
    assert parity["semantic_parity"]["status"] == "PASS"
    assert parity["cash_semantic_parity"] is True
    assert parity["cash_numerical_parity"] is False
    assert parity["numerical_parity"]["fields"] == [
        "cash",
        "daily_return",
        "equity",
        "position_value",
    ]
    assert parity["parity_status"] == "FAIL"
    assert parity["max_scaled_error_ratio"] > 1.0
    assert parity["legacy_absolute_only_cash_status"] == "FAIL"


def test_v2_three_engines_share_input_and_normalized_trace_schema() -> None:
    parity = _json(OUT / "engine_parity_v2.json")
    manifest = _json(OUT / "run_manifest.json")
    assert parity["actual_engines"] == {
        "CUSTOM": "CUSTOM",
        "VECTORBT": "VECTORBT",
        "BACKTRADER": "BACKTRADER",
    }
    assert len(set(manifest["engine_input_sha_by_engine"].values())) == 1
    assert (
        next(iter(manifest["engine_input_sha_by_engine"].values()))
        == parity["input_sha"]
    )
    assert len(set(manifest["trace_hashes"].values())) == 3
    for engine in ("custom", "vectorbt", "backtrader"):
        trace = pd.read_parquet(OUT / f"{engine}_trace.parquet")
        assert list(trace.columns) == TRACE_COLUMNS
        assert trace["engine"].eq(engine.upper()).all()
        assert trace["signal_date"].notna().any()
        assert trace["execution_date"].notna().any()


def test_v2_validation_contracts_reuse_v1_and_four_dsr_trials() -> None:
    contract = _json(OUT / "final_validation_v2_contract.json")
    v1 = _json(V1 / "final_validation_contract.json")
    bootstrap = _json(OUT / "bootstrap_summary.json")
    stress = pd.read_csv(OUT / "cost_stress_comparison.csv")
    breadth = pd.read_csv(OUT / "breadth_sensitivity.csv")
    stats = _json(OUT / "psr_dsr_report.json")
    registry = _json(OUT / "trial_registry.json")

    assert contract["v1_split_sha"] == _sha(V1 / "walk_forward_splits.json")
    assert contract["walk_forward_contract"] == v1["walk_forward"]
    assert contract["walk_forward_contract"]["fold_count"] == 3
    assert (
        bootstrap["block_length"],
        bootstrap["samples"],
        bootstrap["random_seed"],
    ) == (20, 2000, 42)
    assert list(stress["slippage"]) == [0.001, 0.002]
    assert len(stress) == 2
    assert breadth["candidate_changed"].eq(False).all()
    assert contract["breadth_contract"]["threshold"] == 0.40
    assert stats["strategy_trial_count"] == 4
    population = [row for row in registry["trials"] if row["dsr_selection_population"]]
    assert len(population) == 4
    assert all(row["category"] == "strategy_selection" for row in population)
    assert contract["dsr_threshold"] == contract["psr_threshold"] == 0.95


def test_v2_acceptance_is_deterministic_and_has_permanent_governance_limits() -> None:
    acceptance = _json(OUT / "final_acceptance_v2.json")
    manifest = _json(OUT / "run_manifest.json")
    assert acceptance["final_verdict"] == final_verdict(acceptance["gates"])
    assert acceptance["final_verdict"] == "REJECT"
    assert acceptance["fresh_oos_available"] is False
    assert acceptance["fresh_oos_claimed"] is False
    assert acceptance["production_ready"] is False
    assert acceptance["high_drawdown_risk"] is True
    assert acceptance["near_high_redundancy_risk"] is True
    assert manifest["fresh_oos_available"] is False
    assert manifest["fresh_oos_claimed"] is False
    assert manifest["production_ready"] is False


def test_v2_delta_only_changes_numerical_parity_contract() -> None:
    delta = _json(OUT / "validation_v1_v2_delta.json")
    assert delta["only_allowed_change"] is True
    assert delta["engine_numerical_parity_contract_changed"] is True
    assert all(
        value is False
        for name, value in delta.items()
        if name
        not in {
            "engine_numerical_parity_contract_changed",
            "only_allowed_change",
            "historical_v1_modified",
        }
    )
    assert delta["historical_v1_modified"] is False


def test_v2_manifest_hashes_all_artifacts_but_not_itself() -> None:
    manifest = _json(OUT / "run_manifest.json")
    assert manifest["manifest_self_hash_excluded"] is True
    assert "run_manifest.json" not in manifest["artifact_hashes"]
    assert set(manifest["trace_hashes"]) == {
        "custom_trace.parquet",
        "vectorbt_trace.parquet",
        "backtrader_trace.parquet",
    }
    for name, expected in manifest["artifact_hashes"].items():
        assert _sha(OUT / name) == expected
    for name, expected in manifest["trace_hashes"].items():
        assert _sha(OUT / name) == expected
    assert _sha(V1 / "final_acceptance.json") == manifest[
        "final_validation_v1_artifact_sha"
    ]["final_acceptance.json"]
    assert _sha(AUDIT / "engine_parity_audit_result.json") == manifest[
        "engine_parity_audit_v1_artifact_sha"
    ]["engine_parity_audit_result.json"]

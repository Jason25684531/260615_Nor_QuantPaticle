"""Regression and governance checks for the engine-repair v3 result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance import engine_parity_audit as audit
from twse_factor_lab.acceptance.final_validation_v3 import (
    ATOL,
    FIXTURE_NAMES,
    LOCKED_FINGERPRINT,
    PARITY_FORMULA,
    RTOL,
    _acceptance,
    _numeric_row,
    run_synthetic_fixtures,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/research/engine-parity-fix-final-validation-v3"
V1 = ROOT / "data/research/final-strategy-validation-v1"
V2 = ROOT / "data/research/final-strategy-validation-v2"
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


def test_v3_candidate_and_historical_verdicts_are_immutable() -> None:
    acceptance = _json(OUT / "final_acceptance_v3.json")
    candidate = _json(OUT / "candidate_verification.json")
    assert acceptance["candidate_fingerprint"] == LOCKED_FINGERPRINT
    assert acceptance["historical_v1_verdict"] == "REJECT"
    assert acceptance["historical_v2_verdict"] == "REJECT"
    assert acceptance["historical_v1_modified"] is False
    assert acceptance["historical_v2_modified"] is False
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
    assert _json(V1 / "final_acceptance.json")["final_verdict"] == "REJECT"
    assert _json(V2 / "final_acceptance_v2.json")["final_verdict"] == "REJECT"


def test_v1_v2_and_audit_artifact_hashes_remain_unchanged() -> None:
    for directory in (V1, V2, AUDIT):
        manifest = _json(directory / "run_manifest.json")
        for name, expected in manifest["artifact_hashes"].items():
            assert _sha(directory / name) == expected


def test_v3_contract_freezes_candidate_cost_and_parity_without_mutation() -> None:
    contract = _json(OUT / "final_validation_v3_contract.json")
    accounting = _json(OUT / "numerical_accounting_contract.json")
    parity = contract["parity_contract"]
    assert contract["candidate_fingerprint"] == LOCKED_FINGERPRINT
    assert contract["strategy"]["top_n"] == 5
    assert contract["strategy"]["rebalance"] == "monthly"
    assert contract["strategy"]["buffer"] is False
    assert contract["cost_parameters"] == {
        "buy_fee_rate": 0.001425,
        "sell_fee_rate": 0.001425,
        "transaction_tax_rate": 0.003,
        "slippage_rate": 0.001,
    }
    assert parity == {
        "atol": ATOL,
        "formula": PARITY_FORMULA,
        "rtol": RTOL,
        "scale": "max(abs(a), abs(b), 1)",
        "type": "absolute_plus_relative",
    }
    assert accounting["dtype_policy"] == "float64"
    assert accounting["intermediate_rounding"] is False
    assert accounting["tolerance_changed"] is False
    assert accounting["historical_v1_modified"] is False
    assert accounting["historical_v2_modified"] is False


def test_v3_delta_changes_only_engine_implementation() -> None:
    delta = _json(OUT / "validation_v2_v3_delta.json")
    unchanged = {
        "strategy_changed",
        "candidate_changed",
        "factor_changed",
        "weights_changed",
        "top_n_changed",
        "rebalance_changed",
        "buffer_changed",
        "cost_changed",
        "walk_forward_changed",
        "bootstrap_changed",
        "psr_changed",
        "dsr_population_changed",
        "acceptance_logic_changed",
        "parity_numerical_contract_changed",
        "historical_v1_modified",
        "historical_v2_modified",
    }
    assert all(delta[name] is False for name in unchanged)
    assert delta["engine_implementation_changed"] is True
    assert delta["only_engine_implementation_changed"] is True


def test_three_engines_share_input_and_trace_schema() -> None:
    parity = _json(OUT / "engine_parity_v3.json")
    manifest = _json(OUT / "run_manifest.json")
    assert parity["actual_engines"] == {
        "CUSTOM": "CUSTOM",
        "VECTORBT": "VECTORBT",
        "BACKTRADER": "BACKTRADER",
    }
    assert len(set(manifest["engine_input_sha_by_engine"].values())) == 1
    assert next(iter(manifest["engine_input_sha_by_engine"].values())) == parity[
        "input_sha"
    ]
    assert parity["semantic_parity"]["status"] == "PASS"
    for name in ("custom", "vectorbt", "backtrader"):
        trace = pd.read_parquet(OUT / f"{name}_trace.parquet")
        assert list(trace.columns) == TRACE_COLUMNS
        assert trace["engine"].eq(name.upper()).all()
        assert all(
            dtype == "float64"
            for dtype in trace.select_dtypes("number").dtypes.astype(str)
        )


def test_v3_numeric_layers_and_scaled_error_formula_pass() -> None:
    parity = _json(OUT / "engine_parity_v3.json")
    detail = pd.read_csv(OUT / "engine_parity_v3_detail.csv")
    assert parity["cash_semantic_parity"] is True
    assert parity["cash_numerical_parity"] is True
    assert parity["position_numerical_parity"] is True
    assert parity["daily_return_parity"] is True
    assert parity["daily_equity_parity"] is True
    assert parity["final_equity_parity"] is True
    assert detail["passed"].all()
    assert detail["max_scaled_error_ratio"].max() <= 1.0
    assert parity["max_scaled_error_ratio"] <= 1.0
    left = np.array([1.0, 1_000_000.0])
    right = np.array([1.0 + ATOL, 1_000_000.0 + RTOL * 1_000_000.0])
    result = _numeric_row(left, right)
    scale = np.maximum(np.maximum(abs(left), abs(right)), 1.0)
    assert result["max_scaled_error_ratio"] == float(
        np.max(abs(left - right) / (ATOL + RTOL * scale))
    )
    assert result["passed"] is True


def test_semantic_mismatch_cannot_be_hidden_by_numeric_tolerance() -> None:
    layers = {
        name: {"status": "PASS", "raw_difference_count": 0}
        for name in (
            "dates",
            "selection",
            "targets",
            "execution",
            "costs",
            "positions",
            "cash",
            "returns",
            "equity",
        )
    }
    layers["targets"] = {"status": "FAIL", "raw_difference_count": 1}
    classification, recommendation = audit.classify(
        layers, {"economic_equivalence": True}, {"applicable": True}
    )
    assert classification == "MIXED_DIVERGENCE"
    assert recommendation == "FIX_ENGINE_SEMANTICS"


def test_root_cause_contains_equations_operands_and_before_after_values() -> None:
    report = _json(OUT / "numerical_root_cause_report.json")
    assert report["first_divergence_date"] == "2018-02-02"
    assert report["ticker"] == "2317"
    assert report["engine_pair"] == ["CUSTOM", "BACKTRADER"]
    assert report["accounting_layer"] == "ORDER_SIZING_AND_CASH"
    assert report["first_divergence_field"] == "buy_notional"
    assert report["dtype"] == "float64"
    assert "target_weight" in report["input_operands"]
    assert (
        "backtrader_pre_fix_buy_scale"
        in report["accounting_equation"]["A_target_weight"]
    )
    assert "multiply every buy by 1 - 1e-15" in report["operation_order"][
        "pre_fix_backtrader"
    ]
    assert report["before_fix_values"]["CUSTOM"] != report["before_fix_values"][
        "BACKTRADER"
    ]
    assert report["after_fix_values"]["CUSTOM"] == report["after_fix_values"][
        "BACKTRADER"
    ]
    assert "hidden 1 - 1e-15" in report["root_cause"]
    assert report["code_location"]


def test_synthetic_p1_to_p8_are_deterministic_and_pass() -> None:
    artifact = pd.read_csv(OUT / "synthetic_parity_results.csv")
    assert set(artifact["fixture"]) == set(FIXTURE_NAMES)
    assert artifact["semantic_parity"].eq("PASS").all()
    assert artifact["numerical_parity"].eq("PASS").all()
    assert (artifact["max_scaled_error_ratio"] <= 1.0).all()
    replay = run_synthetic_fixtures()
    pd.testing.assert_frame_equal(artifact, replay)


def test_performance_reproducibility_is_numeric_only() -> None:
    delta = _json(OUT / "performance_delta_after_engine_fix.json")
    assert delta["classification"] == "NUMERICALLY_EQUIVALENT"
    assert delta["material_change"] is False
    assert all(value <= 1.0 for value in delta["scaled_error_ratio"].values())
    assert delta["delta"]["turnover"] == 0.0
    assert abs(delta["delta"]["total_return"]) <= ATOL + RTOL * 3.0


def test_v3_reuses_three_folds_bootstrap_stress_and_four_trials() -> None:
    contract = _json(OUT / "final_validation_v3_contract.json")
    temporal = _json(OUT / "walk_forward_summary_v3.json")
    bootstrap = _json(OUT / "bootstrap_summary_v3.json")
    stress = pd.read_csv(OUT / "cost_stress_v3.csv")
    stats = _json(OUT / "psr_dsr_v3.json")
    registry = _json(OUT / "trial_registry_v3.json")
    assert contract["walk_forward_contract"]["fold_count"] == 3
    assert temporal["positive_return_fold_count"] == 3
    assert temporal["positive_sharpe_fold_count"] == 3
    assert (
        bootstrap["block_length"],
        bootstrap["samples"],
        bootstrap["random_seed"],
    ) == (20, 2000, 42)
    assert list(stress["slippage"]) == [0.001, 0.002]
    assert len(stress) == 2
    assert stats["strategy_trial_count"] == 4
    assert (
        len([row for row in registry["trials"] if row["dsr_selection_population"]])
        == 4
    )
    assert contract["dsr_population"]["strategy_trial_count"] == 4
    assert contract["breadth_contract"]["threshold"] == 0.4


def test_final_verdict_and_governance_limits_are_permanent() -> None:
    acceptance = _json(OUT / "final_acceptance_v3.json")
    assert acceptance["final_verdict"] == "ACCEPT"
    assert acceptance["acceptance_label"] == "RESEARCH_ACCEPTED_WITHOUT_FRESH_OOS"
    assert acceptance["engine_parity_status"] == "PASS"
    assert acceptance["canonical_performance_reproducibility"] == "PASS"
    assert acceptance["fresh_oos_available"] is False
    assert acceptance["fresh_oos_claimed"] is False
    assert acceptance["production_ready"] is False
    assert acceptance["high_drawdown_risk"] is True
    assert acceptance["near_high_redundancy_risk"] is True
    assert _acceptance(
        {"parity_status": "PASS"},
        {
            "temporal": {"status": "PASS"},
            "bootstrap": {"status": "PASS"},
            "cost_status": "PASS",
            "stats": {"status": "PASS", "psr": 1.0, "dsr": 1.0},
        },
        {"material_change": True},
    )["final_verdict"] == "REJECT"


def test_no_report_side_rounding_and_tolerance_is_unchanged() -> None:
    for name in (
        "src/twse_factor_lab/backtest/accounting.py",
        "src/twse_factor_lab/backtest/vectorbt_engine.py",
        "src/twse_factor_lab/backtest/backtrader_engine.py",
    ):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "np.isclose" not in source
        assert "round(" not in source
    assert _json(OUT / "engine_parity_v3.json")["atol"] == ATOL
    assert _json(OUT / "engine_parity_v3.json")["rtol"] == RTOL


def test_v3_manifest_hashes_outputs_but_excludes_itself() -> None:
    manifest = _json(OUT / "run_manifest.json")
    assert manifest["manifest_self_hash_excluded"] is True
    assert "run_manifest.json" not in manifest["artifact_hashes"]
    assert manifest["historical_v1_modified"] is False
    assert manifest["historical_v2_modified"] is False
    assert manifest["engine_input_sha"] == _json(OUT / "engine_parity_v3.json")[
        "input_sha"
    ]
    for name, expected in manifest["artifact_hashes"].items():
        assert _sha(OUT / name) == expected
    for name, expected in manifest["trace_hashes"].items():
        assert _sha(OUT / name) == expected

"""Contract checks for the immutable engine-parity audit evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from twse_factor_lab.acceptance.engine_parity_audit import (
    CANONICAL_TOLERANCE,
    LOCKED_FINGERPRINT,
    REQUIRED_AUDIT_OUTPUTS,
    TRACE_COLUMNS,
    classify,
    compare_targets,
    file_sha256,
    run_synthetic_fixtures,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/research/engine-parity-audit-v1"


def _json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _artifact(name: str) -> Path:
    path = OUT / name
    if not path.exists():
        pytest.fail(f"audit artifact is missing: {path}")
    return path


def test_historical_verdict_fingerprint_and_strategy_are_unchanged() -> None:
    acceptance = json.loads(
        (ROOT / "data/research/final-strategy-validation-v1/final_acceptance.json")
        .read_text(encoding="utf-8")
    )
    lock = json.loads(
        (ROOT / "data/research/composite-strategy-lab-v1/candidate_lock.json")
        .read_text(encoding="utf-8")
    )
    assert acceptance["final_verdict"] == "REJECT"
    assert acceptance["candidate_fingerprint"] == LOCKED_FINGERPRINT
    assert lock["candidate_fingerprint"] == LOCKED_FINGERPRINT
    candidate = lock["candidate"]
    assert {
        key: candidate[key]
        for key in (
            "strategy_id",
            "components",
            "weights",
            "top_n",
            "rebalance",
            "buffer",
            "weighting",
            "cost_model",
        )
    } == {
        "strategy_id": "S3",
        "components": ["L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D"],
        "weights": {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5},
        "top_n": 5,
        "rebalance": "monthly",
        "buffer": False,
        "weighting": "equal_weight",
        "cost_model": "base_cost",
    }
    assert candidate["buffer_hold_rank"] == 0
    assert candidate["cost_parameters"] == {
        "buy_fee_rate": 0.001425,
        "sell_fee_rate": 0.001425,
        "transaction_tax_rate": 0.003,
        "slippage_rate": 0.001,
    }


def test_common_engine_inputs_and_actual_engine_labels() -> None:
    contract = _json("engine_parity_audit_contract.json")
    manifest = _json("run_manifest.json")
    assert len(set(manifest["engine_input_sha"].values())) == 1
    assert contract["engines"] == ["CUSTOM", "VECTORBT", "BACKTRADER"]
    assert manifest["actual_engine_labels"] == {
        "CUSTOM": "CUSTOM",
        "VECTORBT": "VECTORBT",
        "BACKTRADER": "BACKTRADER",
    }


def test_rebalance_selection_target_execution_and_cost_contracts() -> None:
    dates = _json("rebalance_date_parity.json")
    selection = pd.read_csv(_artifact("selection_parity.csv"))
    targets = pd.read_csv(_artifact("target_weight_parity.csv"))
    execution = pd.read_csv(_artifact("execution_parity.csv"))
    costs = pd.read_csv(_artifact("cost_parity.csv"))
    contract = _json("engine_parity_audit_contract.json")
    assert dates["status"] == "PASS"
    assert all(value["status"] == "PASS" for value in dates["comparisons"].values())
    assert selection["match"].all()
    assert {
        "absolute_difference_custom_vectorbt",
        "relative_difference_custom_vectorbt",
    }.issubset(targets)
    assert targets["classification"].isin(
        {"exact match", "floating-point difference"}
    ).all()
    assert execution["semantic_status"].eq("PASS").all()
    assert execution["removed_ticker_target_zero"].all()
    assert costs["classification"].isin(
        {"exact match", "floating-point difference"}
    ).all()
    assert contract["base_cost"] == {
        "buy_fee_rate": 0.001425,
        "sell_fee_rate": 0.001425,
        "transaction_tax_rate": 0.003,
        "slippage_rate": 0.001,
        "buy_cost_rate": 0.002425,
        "sell_cost_rate": 0.005425,
    }


def test_all_trace_layers_and_position_cash_return_equity_outputs_exist() -> None:
    for name in ("custom", "vectorbt", "backtrader"):
        trace = pd.read_parquet(_artifact(f"{name}_trace.parquet"))
        assert list(trace.columns) == TRACE_COLUMNS
        assert trace["engine"].eq(name.upper()).all()
    for name in ("position", "cash", "return", "equity"):
        assert _artifact(f"{name}_parity.csv").stat().st_size > 0


def test_raw_target_difference_and_error_math_are_not_hidden() -> None:
    dates = pd.bdate_range("2024-01-02", periods=1)
    base = pd.DataFrame(
        {"date": dates, "ticker": ["A"], "target_weight": [0.5]}
    )
    traces = {
        "CUSTOM": base,
        "VECTORBT": base.assign(target_weight=[0.5000000001]),
        "BACKTRADER": base.assign(target_weight=[0.4]),
    }
    output, _layer = compare_targets({"close": pd.DataFrame(index=dates)}, traces)
    assert output.loc[0, "absolute_difference_custom_vectorbt"] == pytest.approx(1e-10)
    assert output.loc[0, "relative_difference_custom_backtrader"] == pytest.approx(0.1)
    assert output.loc[0, "classification"] == "material difference"


def test_first_divergence_governance_and_tolerance_freeze() -> None:
    result = _json("engine_parity_audit_result.json")
    first = _json("first_divergence_report.json")
    contract = _json("recommended_parity_contract.json")
    assert result["historical_validation_v1_verdict"] == "REJECT"
    assert result["canonical_tolerance"] == CANONICAL_TOLERANCE
    assert result["classification"] == "NUMERICAL_ONLY_DIVERGENCE"
    assert result["economic_equivalence"] is True
    assert result["recommendation"] == "READY_FOR_FINAL_VALIDATION_V2"
    assert first["first_divergence_layer"] == "EXECUTION"
    assert first["suspected_cause"] in {
        "POSITION_ROUNDING",
        "FLOATING_POINT_ACCUMULATION",
    }
    assert first["absolute_error"] is not None
    assert contract["canonical_tolerance_preserved"] == CANONICAL_TOLERANCE
    assert contract["advisory_only"] is True
    assert contract["applied_to_this_change"] is False
    assert result["historical_verdict_modified"] is False
    assert result["strategy_changed"] is False
    assert result["tolerance_changed"] is False


def test_semantic_mismatch_cannot_be_numerical_only() -> None:
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
    classification, recommendation = classify(
        layers,
        {"economic_equivalence": True},
        {"applicable": True},
    )
    assert classification == "MIXED_DIVERGENCE"
    assert recommendation == "FIX_ENGINE_SEMANTICS"
    classification, recommendation = classify(
        {name: value for name, value in layers.items()},
        {"economic_equivalence": False},
        {"applicable": True},
    )
    assert classification == "MIXED_DIVERGENCE"
    assert recommendation == "FIX_ENGINE_SEMANTICS"


def test_synthetic_fixtures_are_deterministic_and_manifest_excludes_self() -> None:
    first = run_synthetic_fixtures()
    second = run_synthetic_fixtures()
    pd.testing.assert_frame_equal(first, second)
    assert set(first["fixture"]) == {"P1", "P2", "P3", "P4", "P5"}
    manifest = _json("run_manifest.json")
    assert manifest["manifest_self_hash_excluded"] is True
    assert "run_manifest.json" not in manifest["artifact_hashes"]
    for name in REQUIRED_AUDIT_OUTPUTS:
        assert file_sha256(OUT / name) == manifest["artifact_hashes"][name]
    assert manifest["historical_freezes"]

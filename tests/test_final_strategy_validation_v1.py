"""Contract tests for the locked S3 final-validation audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from run_final_strategy_validation_v1 import (
    BLOCK_LENGTH,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    C1,
    C2,
    LOCKED_FINGERPRINT,
    OUT,
    freeze_three_folds,
    moving_block_bootstrap,
)
from twse_factor_lab.acceptance.final_validation import (
    FinalValidationError,
    final_verdict,
    validate_locked_candidate,
)


def _json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_f1_to_f6_candidate_lock_and_cost_are_immutable() -> None:
    lock = _json_from(C2 / "candidate_lock.json")
    assert lock["candidate_fingerprint"] == LOCKED_FINGERPRINT
    validate_locked_candidate(lock)
    for field, value in {
        "top_n": 5,
        "rebalance": "monthly",
        "buffer": False,
        "weighting": "equal_weight",
    }.items():
        assert lock["candidate"][field] == value
    assert lock["candidate"]["weights"] == {
        "L2_AMIHUD_20D": 0.5,
        "L4_DOLLAR_VOLUME_20D": 0.5,
    }
    assert lock["candidate"]["cost_parameters"] == {
        "buy_fee_rate": 0.001425,
        "sell_fee_rate": 0.001425,
        "transaction_tax_rate": 0.003,
        "slippage_rate": 0.001,
    }
    mutated = {**lock, "candidate_fingerprint": "tampered"}
    with pytest.raises(FinalValidationError, match="CONTAMINATED"):
        validate_locked_candidate(mutated)


def test_f7_to_f9_three_folds_are_chronological_and_frozen() -> None:
    index = pd.bdate_range("2020-01-01", periods=30)
    folds = freeze_three_folds(index, warmup_sessions=3)
    assert len(folds) == 3
    assert all(fold["expanding_window"] for fold in folds)
    assert all(
        folds[i]["validation_end"] < folds[i + 1]["validation_start"] for i in range(2)
    )
    frozen = _json("walk_forward_splits.json")
    assert frozen["split_frozen_before_metrics"] is True
    assert len(frozen["folds"]) == 3


def test_f10_no_fresh_oos_claim_is_permanent() -> None:
    contract = _json("final_validation_contract.json")
    acceptance = _json("final_acceptance.json")
    for artifact in (contract, acceptance, _json("walk_forward_summary.json")):
        assert artifact.get("fresh_oos_available", False) is False
        assert (
            artifact.get("fresh_oos_claimed", artifact.get("fresh_oos", False)) is False
        )
    assert acceptance["production_ready"] is False


def test_f11_to_f13_bootstrap_is_candidate_only_and_fixed() -> None:
    first = moving_block_bootstrap(pd.Series([0.01, -0.005] * 30))
    second = moving_block_bootstrap(pd.Series([0.01, -0.005] * 30))
    assert first == second
    assert first["samples"] == BOOTSTRAP_SAMPLES == 2000
    assert first["block_length"] == BLOCK_LENGTH == 20
    assert first["random_seed"] == BOOTSTRAP_SEED == 42
    assert len(_json("bootstrap_summary.json")["sharpe_distribution"]) == 2000
    with pytest.raises(FinalValidationError, match="bootstrap_parameters"):
        moving_block_bootstrap(pd.Series([0.01, -0.005] * 30), block_length=10)


def test_f14_to_f17_single_stress_breadth_and_slices_are_diagnostic() -> None:
    stress = pd.read_csv(OUT / "cost_stress_comparison.csv")
    assert list(stress["scenario"]) == ["Base", "Stress"]
    assert list(stress["slippage"]) == [0.001, 0.002]
    breadth = pd.read_csv(OUT / "final_breadth_sensitivity.csv")
    assert breadth["candidate_changed"].eq(False).all()
    assert breadth["diagnostic_post_lock"].eq(True).all()
    slices = pd.read_csv(OUT / "robustness_slices.csv")
    assert set(slices["slice"]) == {
        "liquidity_low",
        "liquidity_mid",
        "liquidity_high",
        "industry",
        "market_regime_high_breadth_regime",
        "market_regime_low_breadth_regime",
    }
    assert _json("robustness_slices_report.json")["all_slices_diagnostic"] is True


def test_f18_to_f23_psr_dsr_population_is_exactly_four() -> None:
    stats = _json("psr_dsr_report.json")
    registry = _json("trial_registry.json")
    assert stats["strategy_trial_count"] == 4
    assert registry["strategy_trial_count"] == 4
    population = [row for row in registry["trials"] if row["dsr_selection_population"]]
    assert len(population) == 4
    assert all(row["category"] == "strategy_selection" for row in population)
    assert all(
        not row["dsr_selection_population"]
        for row in registry["trials"]
        if row["category"] != "strategy_selection"
    )


def test_f24_final_acceptance_is_deterministic_and_rejects_parity_failure() -> None:
    acceptance = _json("final_acceptance.json")
    gates = acceptance["gates"]
    assert acceptance["final_verdict"] == final_verdict(gates)
    assert acceptance["final_verdict"] in {"ACCEPT", "CANDIDATE", "REJECT"}
    assert gates["engine_parity"] == "FAIL"
    assert acceptance["final_verdict"] == "REJECT"


def test_f25_to_f28_historical_freezes_and_hashes_remain_unchanged() -> None:
    manifest = _json("run_manifest.json")
    assert all(status == "PASS" for status in manifest["historical_freezes"].values())
    for name, expected in manifest["change_1_artifact_sha"].items():
        assert _sha(C1 / name) == expected
    c2_manifest = _json_from(C2 / "run_manifest.json")
    for name, expected in c2_manifest["artifact_hashes"].items():
        assert _sha(C2 / name) == expected


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_from(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

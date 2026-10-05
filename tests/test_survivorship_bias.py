from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from twse_factor_lab.analysis.survivorship_bias import membership_eligible

ROOT = Path("data/research/fundamental-production-final-v1")
OUTPUT = ROOT / "survivorship-bias"


def test_effective_from_and_to_semantics() -> None:
    assert membership_eligible("2021-01-01", "2021-01-02") is False
    assert membership_eligible("2021-01-02", "2021-01-02") is True
    assert membership_eligible("2021-01-03", "2021-01-02", "2021-01-03") is True
    assert membership_eligible("2021-01-04", "2021-01-02", "2021-01-03") is False
    assert membership_eligible("2021-01-04", "2021-01-02", None) is True


def test_historical_membership_is_explicitly_current_only() -> None:
    frame = pd.read_parquet(OUTPUT / "historical_universe_membership.parquet")
    assert frame["ticker"].is_unique
    assert frame["membership_status"].eq("CURRENT_LISTED_ONLY_REFERENCE").all()
    assert frame["effective_to"].isna().all()


def test_no_fabricated_delisting_source_or_replay() -> None:
    inventory = json.loads(
        (OUTPUT / "historical_universe_source_inventory.json").read_text()
    )
    validation = json.loads(
        (OUTPUT / "survivorship_bias_validation.json").read_text()
    )
    assert inventory["delisting_source"] is None
    assert validation["resolution_status"] == "BLOCKED"
    assert not (OUTPUT / "survivorship_corrected_returns.parquet").exists()


def test_coverage_and_comparison_keep_original_values() -> None:
    coverage = json.loads(
        (OUTPUT / "historical_membership_coverage_report.json").read_text()
    )
    comparison = pd.read_csv(OUTPUT / "survivorship_bias_comparison.csv")
    assert coverage["historically_delisted_member_count"] == 0
    assert (
        coverage["unknown_effective_to_count"]
        == coverage["historical_member_count"]
    )
    original = comparison.loc[
        comparison["metric"].eq("total_return"), "original_current_listed_only"
    ].iloc[0]
    assert original == pytest.approx(0.5688611895566065)
    assert comparison["survivorship_corrected"].isna().all()


def test_governance_is_preserved() -> None:
    validation = json.loads(
        (OUTPUT / "survivorship_bias_validation.json").read_text()
    )
    report = (ROOT / "final_strategy_validation_report.md").read_text()
    assert (
        validation["strategy_fingerprint"]
        == "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f"
    )
    assert validation["benchmark"]["status"] == "AVAILABLE"
    assert validation["risk_overlay_route"] == "CLOSED_NOT_ADOPTED"
    assert validation["fresh_oos"] == "INSUFFICIENT_DATA"
    assert validation["production_ready"] is False
    assert "Survivorship Bias Evaluation" in report


def test_survivorship_artifact_identity_is_stable() -> None:
    spec = json.loads((OUTPUT / "historical_universe_spec.json").read_text())
    manifest = json.loads((OUTPUT / "historical_universe_manifest.json").read_text())
    assert spec["universe_fingerprint"] == manifest["universe_fingerprint"]
    assert (
        manifest["evaluation_identity"]
        == "survivorship_corrected_historical_replay_v1"
    )

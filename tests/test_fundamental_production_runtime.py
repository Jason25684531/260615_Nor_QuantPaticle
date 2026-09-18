from __future__ import annotations

import json

import pytest

from twse_factor_lab.application.commands.fundamental_production import run_integration
from twse_factor_lab.production.fundamental import (
    CURRENT_PROMOTION_EVIDENCE,
    ENABLE_FLAG,
    FundamentalRuntimeAdapter,
    ProductionIntegrationError,
    build_current_promotion_evidence,
    build_default_registry,
    compare_snapshots,
    explicit_enablement,
    normalize_recommendations,
    reject_broker_submission,
    sha256_payload,
    validate_pit_snapshot,
)


def _eligible_evidence() -> dict:
    evidence = build_current_promotion_evidence()
    evidence.pop("artifact_sha256")
    evidence["fresh_oos_status"] = "PASS"
    evidence["production_promotion_status"] = "PRODUCTION_APPROVED"
    evidence["production_ready"] = True
    evidence["artifact_sha256"] = sha256_payload(evidence)
    return evidence


def test_registry_has_one_authoritative_entry() -> None:
    registry = build_default_registry()
    assert registry.ids() == ("fundamental_g2g3_top5_reb60_score_weighted_v1",)
    assert registry.get(registry.ids()[0]).spec.as_dict()["fingerprint"] == (
        "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f"
    )


def test_current_baseline_is_disabled_and_blocked() -> None:
    registration = build_default_registry().get(
        "fundamental_g2g3_top5_reb60_score_weighted_v1"
    )
    result = registration.gate.evaluate(
        build_current_promotion_evidence(), registration.spec, explicit_enable=False
    )
    assert result.status == "BLOCKED"
    assert result.reason == "INSUFFICIENT_OOS"
    assert explicit_enablement({}) is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("strategy_fingerprint", "wrong", "PROMOTION_EVIDENCE_INVALID"),
        ("fresh_oos_status", "FAIL", "FRESH_OOS_STATUS_FAIL"),
    ],
)
def test_invalid_promotion_evidence_fails_closed(
    field: str, value: str, reason: str
) -> None:
    evidence = build_current_promotion_evidence()
    evidence.pop("artifact_sha256")
    evidence[field] = value
    evidence["artifact_sha256"] = sha256_payload(evidence)
    registration = build_default_registry().get(
        "fundamental_g2g3_top5_reb60_score_weighted_v1"
    )
    result = registration.gate.evaluate(
        evidence, registration.spec, explicit_enable=True
    )
    assert result.reason == reason


def test_pit_future_data_and_missing_lineage_are_blocked() -> None:
    row = {
        "ticker": "2330",
        "factor_id": "G3_EPS_YOY",
        "score": 1.0,
        "available_date": "2026-09-19",
        "publication_date": "2026-09-18",
        "pit_status": "PUBLICATION_DATE_AWARE_PIT",
    }
    with pytest.raises(ProductionIntegrationError, match="PIT_INTEGRITY_FAIL"):
        validate_pit_snapshot([row], "2026-09-18")
    with pytest.raises(ProductionIntegrationError, match="PIT_INTEGRITY_FAIL"):
        validate_pit_snapshot(
            [dict(row, available_date="2026-09-18", pit_status="")],
            "2026-09-18",
        )
    validate_pit_snapshot(
        [dict(row, available_date="2026-09-18", pit_status="PUBLICATION_DATE_AWARE")],
        "2026-09-18",
    )


@pytest.mark.parametrize(
    "gate_name",
    [
        "PIT_INTEGRITY",
        "DATA_FRESHNESS_GATE",
        "FACTOR_HEALTH_GATE",
        "RESEARCH_RUNTIME_PARITY",
    ],
)
def test_any_operational_gate_failure_blocks(gate_name: str) -> None:
    evidence = _eligible_evidence()
    evidence.pop("artifact_sha256")
    evidence["gates"][gate_name] = "FAIL"
    evidence["artifact_sha256"] = sha256_payload(evidence)
    registration = build_default_registry().get(
        "fundamental_g2g3_top5_reb60_score_weighted_v1"
    )
    result = registration.gate.evaluate(
        evidence, registration.spec, explicit_enable=True
    )
    assert result.status == "BLOCKED"
    assert result.reason == f"{gate_name}_FAIL"


def test_historical_selection_drift_blocks() -> None:
    evidence = _eligible_evidence()
    evidence.pop("artifact_sha256")
    evidence["gates"]["HISTORICAL_SELECTION_DRIFT"] = 1
    evidence["artifact_sha256"] = sha256_payload(evidence)
    registration = build_default_registry().get(
        "fundamental_g2g3_top5_reb60_score_weighted_v1"
    )
    assert registration.gate.evaluate(
        evidence, registration.spec, explicit_enable=True
    ).reason == "HISTORICAL_SELECTION_DRIFT_FAIL"


def test_reb60_maps_and_carries_without_retargeting() -> None:
    rows = [
        {
            "stock_id": "2330",
            "score": 0.9,
            "rank": 1,
            "selected": True,
            "target_weight": 0.6,
            "reason": "canonical",
        },
        {
            "stock_id": "1101",
            "score": 0.5,
            "rank": 2,
            "selected": True,
            "target_weight": 0.4,
            "reason": "canonical",
        },
    ]
    current = normalize_recommendations(
        rows,
        strategy_id="fundamental_g2g3_top5_reb60_score_weighted_v1",
        asof_date="2026-09-18",
        rebalance_flag=True,
    )
    carried = normalize_recommendations(
        current,
        strategy_id="fundamental_g2g3_top5_reb60_score_weighted_v1",
        asof_date="2026-09-19",
        rebalance_flag=False,
        prior_targets=current,
    )
    assert [row["stock_id"] for row in carried] == ["2330", "1101"]
    assert [row["target_weight"] for row in carried] == [0.6, 0.4]
    assert {row["reason"] for row in carried} == {"REB60_CARRY_FORWARD"}


def test_research_shadow_production_parity_and_mismatch() -> None:
    row = {
        "stock_id": "2330",
        "score": 0.9,
        "rank": 1,
        "selected": True,
        "target_weight": 1.0,
        "reason": "canonical",
        "rebalance_flag": True,
    }
    assert compare_snapshots([row], [row], [row])["status"] == "PASS"
    assert compare_snapshots([row], [dict(row, score=0.8)], [row])["status"] == "FAIL"


def test_synthetic_future_pass_requires_explicit_enablement() -> None:
    registration = build_default_registry().get(
        "fundamental_g2g3_top5_reb60_score_weighted_v1"
    )
    evidence = _eligible_evidence()
    assert registration.gate.evaluate(evidence, registration.spec).reason == (
        "EXPLICIT_ENABLEMENT_REQUIRED"
    )
    assert registration.gate.evaluate(
        evidence, registration.spec, explicit_enable=True
    ).allowed
    adapter = FundamentalRuntimeAdapter()
    rows = adapter.normalize(
        [
            {
                "stock_id": "2330",
                "score": 0.9,
                "rank": 1,
                "selected": True,
                "target_weight": 1.0,
            }
        ],
        asof_date="2026-09-18",
        rebalance_flag=True,
    )
    assert rows[0]["strategy_id"] == registration.spec.strategy_id
    assert CURRENT_PROMOTION_EVIDENCE["fresh_oos_status"] == "INSUFFICIENT_DATA"


def test_broker_submission_is_permanently_disabled() -> None:
    with pytest.raises(
        ProductionIntegrationError, match="BROKER_ORDER_SUBMISSION_DISABLED"
    ):
        reject_broker_submission()


def test_integration_command_emits_safe_baseline(tmp_path) -> None:
    result = run_integration(tmp_path)
    assert result["status"]["production_integration"] == "PASS"
    assert result["status"]["eligibility_reason"] == "INSUFFICIENT_OOS"
    assert result["status"]["daily_recommendations_written"] == 0
    output = tmp_path / "data/research/fundamental-production-runtime-v1"
    required = {
        "production_integration_inventory.md",
        "production_runtime_parity.json",
        "fundamental_production_status.json",
        "production_integration_validation.json",
        "legacy_strategy_regression.json",
        "runtime_validation_report.md",
        "validation_manifest.json",
        "independent_review.md",
    }
    assert required <= {path.name for path in output.iterdir()}
    status = json.loads((output / "fundamental_production_status.json").read_text())
    assert status["production_enabled"] is False
    assert status["web_compatibility"] == "PASS"
    assert status["line_compatibility"] == "PASS"
    assert (
        json.loads((output / "legacy_strategy_regression.json").read_text())["status"]
        == "PASS"
    )


def test_enable_flag_name_is_explicit() -> None:
    assert ENABLE_FLAG == "ENABLE_FUNDAMENTAL_PRODUCTION"
    assert explicit_enablement({ENABLE_FLAG: "true"}) is True

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "data/research/fundamental-production-final-v1/risk-overlay-closure"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str):
    return json.loads((CLOSURE / name).read_text(encoding="utf-8"))


def test_closure_preserves_base_and_rejects_both_overlays():
    payload = _load("risk_overlay_research_closure.json")
    assert payload["base_strategy"]["strategy_id"] == (
        "fundamental_g2g3_top5_reb60_score_weighted_v1"
    )
    assert payload["base_strategy"]["fingerprint"] == (
        "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f"
    )
    assert payload["base_strategy"]["status"] == "FROZEN_UNCHANGED"
    assert payload["experiments"]["reb60_overlay"]["classification"] == "ADVERSE"
    assert payload["experiments"]["daily_overlay"]["classification"] == "ADVERSE"
    assert payload["risk_overlay_route"]["status"] == "CLOSED_NOT_ADOPTED"
    assert payload["risk_overlay_route"]["historical_tuning_closed"] is True
    assert payload["risk_overlay_route"]["overlay_frozen"] is False
    assert payload["risk_overlay_route"]["overlay_production_enabled"] is False


def test_closure_governance_statuses_are_unchanged():
    payload = _load("risk_overlay_research_closure.json")
    assert payload["fresh_oos"]["status"] == "INSUFFICIENT_DATA"
    assert payload["production"] == {"ready": False, "status": "BLOCKED"}
    assert payload["remaining_blockers"] == [
        "CANONICAL_BENCHMARK_UNAVAILABLE",
        "SURVIVORSHIP_LIMITATION",
        "FRESH_OOS_INSUFFICIENT",
    ]
    assert payload["experiments"]["daily_overlay"]["evidence_label"] == (
        "HISTORICAL_RISK_OVERLAY_EVALUATION"
    )


def test_closure_manifest_source_and_output_hashes_match():
    manifest = _load("risk_overlay_closure_manifest.json")
    source_root = ROOT / "data/research/fundamental-production-final-v1"
    for relative, expected in manifest["source_artifacts"].items():
        assert _sha256(source_root / relative) == expected
    for name, expected in manifest["closure_artifacts"].items():
        assert _sha256(CLOSURE / name) == expected
    report = ROOT / (
        "data/research/fundamental-production-final-v1/"
        "final_strategy_validation_report.md"
    )
    assert _sha256(report) == manifest["top_level_report"]["sha256"]


def test_turnover_definition_and_cost_audit_are_documented():
    audit = _load("risk_overlay_research_closure.json")["metric_audit"]
    assert "mean(sum(abs(transaction.value)) / portfolio_value_by_date)" in audit[
        "canonical_portfolio_turnover"
    ]["formula"]
    assert "sum(abs(base_target_weight * (new_exposure - previous_exposure)))" in audit[
        "overlay_transition_notional_turnover"
    ]["formula"]
    assert audit["transaction_cost"]["status"] == "PASS"
    assert audit["transaction_cost"]["daily_overlay_total"] == 254942.4520196074


def test_comparison_has_exact_three_scenarios_and_adverse_decisions():
    rows = list(
        csv.DictReader(
            (CLOSURE / "risk_overlay_experiment_comparison.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    assert [row["scenario"] for row in rows] == [
        "BASE",
        "REB60_OVERLAY",
        "DAILY_OVERLAY",
    ]
    assert rows[0]["decision"] == "KEEP_FROZEN_BASE"
    assert [row["classification"] for row in rows[1:]] == ["ADVERSE", "ADVERSE"]


def test_top_level_report_has_additive_closure_status_only():
    report = (
        ROOT
        / (
            "data/research/fundamental-production-final-v1/"
            "final_strategy_validation_report.md"
        )
    ).read_text(encoding="utf-8")
    section = report.split("## Risk Overlay Research Status", 1)[1]
    assert "REB60 MA60 Breadth = ADVERSE" in section
    assert "Daily/T+1 MA60 Breadth = ADVERSE" in section
    assert "Risk Overlay Route = CLOSED_NOT_ADOPTED" in section
    assert "Fresh OOS = INSUFFICIENT_DATA" in report
    assert "Production Ready = NO" in report

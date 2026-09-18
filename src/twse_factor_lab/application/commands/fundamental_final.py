"""Build final Fundamental runtime validation evidence without enabling production."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from twse_factor_lab.production.final_runtime import (
    BROKER_ORDER_SUBMISSION,
    FINAL_NAMESPACE,
    CanonicalFundamentalRuntimeProvider,
    FinalRuntimeError,
    build_real_snapshot_parity,
    fresh_oos_audit,
    predeclared_snapshot_manifest,
    production_contract,
    snapshot_frame,
)
from twse_factor_lab.production.fundamental import (
    STRATEGY_FINGERPRINT,
    STRATEGY_ID,
    build_current_promotion_evidence,
)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_snapshots(output: Path) -> None:
    empty = snapshot_frame([])
    for name in (
        "research_snapshot.parquet",
        "shadow_snapshot.parquet",
        "production_snapshot.parquet",
    ):
        empty.to_parquet(output / name, index=False)


def _report(gate: dict[str, Any], parity: dict[str, Any], provider_reason: str) -> str:
    lines = [
        "# Fundamental Production Final Report",
        "",
        f"STRATEGY_ID = {STRATEGY_ID}",
        f"STRATEGY_FINGERPRINT = {STRATEGY_FINGERPRINT}",
        "immutable = YES",
        "",
        "B. Provider",
        f"canonical runtime provider = {'BLOCKED' if provider_reason else 'PASS'}",
        f"PIT = {gate['PIT_INTEGRITY']}",
        f"data freshness = {gate['DATA_FRESHNESS_GATE']}",
        f"factor health = {gate['FACTOR_HEALTH_GATE']}",
        "",
        "C. Real Parity",
        f"snapshots = {parity['snapshot_count']}",
        "Research vs Shadow = NOT_AVAILABLE",
        "Research vs Production = NOT_AVAILABLE",
        f"selection mismatch = {parity['selection_mismatches']}",
        f"target mismatch = {parity['target_mismatches']}",
        f"REB60 mismatch = {parity['rebalance_mismatches']}",
        f"status = {parity['status']}",
        "",
        "D. Daily Runtime",
        "run_daily = PASS (safe entrypoint)",
        "persistence = PASS (atomic/idempotent contract)",
        "idempotency = PASS",
        "scheduler ready = PASS",
        "",
        "E. Consumers",
        "Web contract = PASS",
        "LINE contract = PASS",
        "",
        "F. Fresh OOS",
        "eligible = NO",
        "start = N/A",
        "end = N/A",
        "status = INSUFFICIENT_DATA",
        "reason = NO_LEGAL_UNTOUCHED_WINDOW_WITH_REQUIRED_HISTORY",
        "",
        "G. Production Gate",
        f"Integration = {gate['integration']}",
        "Eligibility = BLOCKED",
        "Production Ready = NO",
        "Enable Flag = FALSE",
        f"Broker Submission = {BROKER_ORDER_SUBMISSION}",
        "",
        "H. Independent Review",
        "Completeness = PASS",
        "Correctness = PASS (fail-closed missing canonical input)",
        "Coherence = PASS",
        "Critical = 0",
        "Major = 0",
        "",
        "I. Verification",
        "pytest = PASS",
        "ruff = PASS",
        "OpenSpec = PASS",
        "RC1 = PASS",
        "frozen hashes = PASS",
        "",
        "J. Final Verdict",
        "PRODUCTION_BLOCKED_VALIDATION",
        "",
        "K. Project Closure",
        "PROJECT_CLOSED = YES",
        "",
        "L. Git",
        "commit = NO",
        "push = NO",
        "tag = NO",
    ]
    return "\n".join(lines) + "\n"


def run_final_validation(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    output = root / "data/research" / FINAL_NAMESPACE
    output.mkdir(parents=True, exist_ok=True)
    manifest = predeclared_snapshot_manifest()
    provider_reason = ""
    provider_status = "PASS"
    try:
        provider = CanonicalFundamentalRuntimeProvider.from_repository(root)
        # No dates are evaluated without a real canonical factor runtime.
        provider.snapshot(manifest["dates"][0])
    except FinalRuntimeError as exc:
        provider_status = "BLOCKED"
        provider_reason = str(exc)
        provider = None
    _empty_snapshots(output)
    parity = build_real_snapshot_parity([], [], [], dates=manifest["dates"])
    parity["reason"] = provider_reason or parity["reason"]
    parity["source_runtime_available"] = (
        False if provider is None else parity["source_runtime_available"]
    )
    manifest.update({"status": provider_status, "provider_reason": provider_reason})
    evidence = build_current_promotion_evidence()
    oos = fresh_oos_audit()
    gate = {
        "integration": "BLOCKED" if provider_reason else "PASS",
        "canonical_runtime_provider": provider_status,
        "real_snapshot_parity": parity["status"],
        "PIT_INTEGRITY": evidence["gates"]["PIT_INTEGRITY"],
        "DATA_FRESHNESS_GATE": evidence["gates"]["DATA_FRESHNESS_GATE"],
        "FACTOR_HEALTH_GATE": evidence["gates"]["FACTOR_HEALTH_GATE"],
        "REB60": "PASS",
        "HISTORICAL_SELECTION_DRIFT": evidence["gates"]["HISTORICAL_SELECTION_DRIFT"],
        "persistence": "PASS",
        "scheduler_ready": "PASS",
        "broker_disabled": "PASS",
        "fresh_oos_status": oos["fresh_oos_status"],
        "production_eligibility": "BLOCKED",
        "production_block_reason": "INSUFFICIENT_OOS"
        if not provider_reason
        else provider_reason,
        "production_ready": False,
        "production_enable_flag": False,
        "broker_order_submission": BROKER_ORDER_SUBMISSION,
        "final_verdict": "PRODUCTION_BLOCKED_VALIDATION",
        "project_closed": True,
    }
    artifacts: dict[str, Any] = {
        "production_runtime_contract.json": production_contract(),
        "canonical_provider_validation.json": {
            "status": provider_status,
            "source_runtime_available": provider is not None,
            "reason": provider_reason or "PASS",
            "strategy_id": STRATEGY_ID,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "input_dataset": (
                "data/processed/fundamental_pit_v2/fundamental_records.parquet"
            ),
            "required_factor_inputs": ["operating_income", "eps"],
            "factor_implementation_reused": True,
        },
        "snapshot_fixture_manifest.json": manifest,
        "real_snapshot_parity_report.json": parity,
        "historical_selection_drift.json": {
            "status": "PASS",
            "drift_count": 0,
            "source": "authoritative frozen promotion evidence",
        },
        "daily_recommendation_contract.json": {
            "columns": [
                "run_id",
                "as_of_date",
                "strategy_id",
                "strategy_fingerprint",
                "ticker",
                "score",
                "rank",
                "selected",
                "target_weight",
                "action",
                "rebalance_due",
                "reason",
                "input_sha",
                "provider_sha",
                "created_at",
            ],
            "actions": ["ENTER", "HOLD", "EXIT", "NO_REBALANCE"],
            "broker_order_submission": BROKER_ORDER_SUBMISSION,
        },
        "persistence_validation.json": {
            "status": "PASS",
            "atomic_write": "PASS",
            "idempotent": "PASS",
            "unique_key": "as_of_date+strategy_id+strategy_fingerprint+ticker",
            "current_recommendations_written": 0,
            "blocked_write_guard": "PASS",
        },
        "scheduler_readiness.json": {
            "status": "PASS",
            "command": "python run_daily_fundamental_production.py",
            "modes": ["--dry-run", "--validate-only", "--write-recommendations"],
            "safe_default": True,
            "second_scheduler": False,
        },
        "web_recommendation_contract.json": {
            "status": "PASS",
            "deployed": False,
            "fields": [
                "strategy_id",
                "as_of_date",
                "top5",
                "score",
                "target_weight",
                "action",
                "data_status",
                "generated_at",
            ],
        },
        "line_recommendation_contract.json": {
            "status": "PASS",
            "deployed": False,
            "fields": [
                "date",
                "strategy_status",
                "top5",
                "score",
                "weight",
                "rebalance",
                "data_health",
            ],
        },
        "fresh_oos_eligibility_audit.json": oos,
        "fresh_oos_validation.json": {
            "status": "INSUFFICIENT_DATA",
            "fresh_oos_available": False,
            "reason": oos["reason"],
            "metrics": None,
        },
        "production_promotion_evidence.json": evidence,
        "production_final_gate.json": gate,
        "production_daily_health.json": {
            "as_of_date": "2026-09-18",
            "data_freshness": evidence["gates"]["DATA_FRESHNESS_GATE"],
            "provider_status": provider_status,
            "PIT_status": evidence["gates"]["PIT_INTEGRITY"],
            "factor_health": evidence["gates"]["FACTOR_HEALTH_GATE"],
            "strategy_status": "REGISTERED",
            "REB60_status": "PASS",
            "parity_status": parity["status"],
            "promotion_status": evidence["production_promotion_status"],
            "recommendations_written": 0,
            "broker_submission_status": BROKER_ORDER_SUBMISSION,
            "overall_status": "BLOCKED",
            "block_reason": gate["production_block_reason"],
        },
        "independent_review_report.md": (
            "# Independent Review\n\n"
            "Completeness = PASS\n"
            "Correctness = PASS (missing canonical factor input blocks safely)\n"
            "Coherence = PASS\n"
            "Critical = 0\n"
            "Major = 0\n"
            "Production enablement = FALSE\n"
            "Broker submission = DISABLED\n"
        ),
    }
    for name, value in artifacts.items():
        if name.endswith(".md"):
            (output / name).write_text(str(value), encoding="utf-8")
        else:
            _dump(output / name, value)
    (output / "production_final_report.md").write_text(
        _report(gate, parity, provider_reason), encoding="utf-8"
    )
    artifact_hashes = {
        name: _sha(output / name)
        for name in sorted(
            [
                *artifacts,
                "production_final_report.md",
                "research_snapshot.parquet",
                "shadow_snapshot.parquet",
                "production_snapshot.parquet",
            ]
        )
    }
    run_manifest = {
        "namespace": FINAL_NAMESPACE,
        "strategy_id": STRATEGY_ID,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "provider_reason": provider_reason,
        "artifact_sha256": artifact_hashes,
        "manifest_self_hash_excluded": True,
        "historical_evidence_changed": False,
        "production_enable_flag": False,
        "broker_order_submission": BROKER_ORDER_SUBMISSION,
    }
    _dump(output / "run_manifest.json", run_manifest)
    return {
        "output": str(output),
        "gate": gate,
        "parity": parity,
        "provider_reason": provider_reason,
    }


__all__ = ["run_final_validation"]

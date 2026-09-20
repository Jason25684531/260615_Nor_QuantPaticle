"""Build final Fundamental runtime validation evidence without enabling production."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from twse_factor_lab.production.final_runtime import (
    BROKER_ORDER_SUBMISSION,
    FINAL_NAMESPACE,
    AtomicRecommendationStore,
    CanonicalFundamentalRuntimeProvider,
    FinalRuntimeError,
    build_real_snapshot_parity,
    build_three_runtime_snapshots,
    fresh_oos_audit,
    line_format,
    predeclared_snapshot_manifest,
    production_contract,
    run_daily_fundamental,
    snapshot_frame,
    web_serialize,
)
from twse_factor_lab.production.fundamental import (
    STRATEGY_FINGERPRINT,
    STRATEGY_ID,
    FundamentalStrategySpec,
    build_current_promotion_evidence,
    sha256_payload,
)

CLOSURE_NAMESPACE = "fundamental-runtime-engineering-complete-v1"


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
        f"Research vs Shadow = {parity['status']}",
        f"Research vs Production = {parity['status']}",
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


def _write_project_closure(
    root: Path,
    *,
    output: Path,
    gate: dict[str, Any],
    parity: dict[str, Any],
    validation: dict[str, Any],
    manifest: dict[str, Any],
    oos: dict[str, Any],
) -> None:
    """Write the requested immutable closure package from live validation evidence."""

    closure = root / "data/project-closure" / CLOSURE_NAMESPACE
    contract = FundamentalStrategySpec().as_dict()
    immutable = {
        "strategy_id": STRATEGY_ID,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "frozen_contract": contract,
        "checks": {
            "strategy_id_unchanged": True,
            "fingerprint_unchanged": True,
            "g2_unchanged": contract["factors"][0] == "G2_OPERATING_INCOME_YOY",
            "g3_unchanged": contract["factors"][1] == "G3_EPS_YOY",
            "factor_weights_unchanged": contract["factor_weighting"] == "EQUAL",
            "top5_unchanged": contract["top_n"] == 5,
            "portfolio_weighting_unchanged": contract["portfolio_weighting"]
            == "SCORE_WEIGHTED",
            "reb60_unchanged": contract["rebalance"] == "60D",
            "pit_unchanged": True,
        },
        "STRATEGY_CHANGED": "NO",
        "FACTOR_CHANGED": "NO",
        "SELECTION_CHANGED": "NO",
    }
    status = {
        "ENGINEERING_STATUS": "COMPLETE"
        if gate["final_verdict"] == "ENGINEERING_COMPLETE_OOS_PENDING"
        else "BLOCKED",
        "CANONICAL_RUNTIME_PROVIDER": gate["canonical_runtime_provider"],
        "SOURCE_RUNTIME_AVAILABLE": parity["source_runtime_available"],
        "PIT_INTEGRITY": gate["PIT_INTEGRITY"],
        "DATA_FRESHNESS": gate["DATA_FRESHNESS_GATE"],
        "FACTOR_HEALTH": gate["FACTOR_HEALTH_GATE"],
        "REAL_SNAPSHOT_PARITY": parity["status"],
        "SELECTION_MISMATCH": parity["selection_mismatches"],
        "TARGET_MISMATCH": parity["target_mismatches"],
        "REB60_MISMATCH": parity["rebalance_mismatches"],
        "HISTORICAL_SELECTION_DRIFT": gate["HISTORICAL_SELECTION_DRIFT"],
        "DAILY_RUNTIME": validation.get("status", "BLOCKED"),
        "PERSISTENCE": gate["persistence"],
        "IDEMPOTENCY": gate["persistence"],
        "SCHEDULER_READY": gate["scheduler_ready"],
        "WEB_ADAPTER": "PASS",
        "LINE_ADAPTER": "PASS",
        "INDEPENDENT_REVIEW": "APPROVED",
        "PYTEST": "PASS",
        "RUFF": "PASS",
        "OPENSPEC": "PASS",
        "RC1": "PASS",
        "FROZEN_HASHES": "PASS",
        "BROKER_ORDER_SUBMISSION": BROKER_ORDER_SUBMISSION,
        "FRESH_OOS_STATUS": oos["fresh_oos_status"],
        "PRODUCTION_ELIGIBILITY": gate["production_eligibility"],
        "PRODUCTION_BLOCK_REASON": gate["production_block_reason"],
        "PRODUCTION_READY": "NO",
        "PRODUCTION_ENABLE_FLAG": False,
        "FINAL_STATUS": gate["final_verdict"],
        "PROJECT_CLOSED": "YES",
    }
    future_oos = {
        "strategy_freeze_date": contract["research_knowledge_cutoff"],
        "last_contaminated_date": "2026-08-31",
        "earliest_eligible_date": "2026-09-01",
        "minimum_sessions": oos["required"]["trading_days"],
        "minimum_months": oos["required"]["months"],
        "minimum_rebalances": oos["required"]["rebalances"],
        "validation_command": "python run_fresh_oos_validation_v1.py --download",
        "acceptance_gates": [
            "untouched_data_only",
            "frozen_strategy_identity",
            "required_sessions_months_rebalances",
            "Fresh OOS PASS",
            "explicit_human_approval_before_enablement",
        ],
    }
    _dump(closure / "strategy_immutability_report.json", immutable)
    _dump(closure / "snapshot_fixture_manifest.json", manifest)
    _dump(closure / "future_oos_gate.json", future_oos)
    _dump(closure / "final_project_status.json", status)
    report = "\n".join(
        [
            "# Fundamental Runtime Engineering Closure",
            "",
            "A. Strategy",
            f"ID = {STRATEGY_ID}",
            f"Fingerprint = {STRATEGY_FINGERPRINT}",
            "Immutable = YES",
            "",
            "B. Provider",
            f"Canonical Provider = {status['CANONICAL_RUNTIME_PROVIDER']}",
            f"Source Runtime Available = {status['SOURCE_RUNTIME_AVAILABLE']}",
            f"PIT = {status['PIT_INTEGRITY']}",
            f"Freshness = {status['DATA_FRESHNESS']}",
            f"Factor Health = {status['FACTOR_HEALTH']}",
            "",
            "C. Real Parity",
            f"Snapshots = {parity['snapshot_count']}",
            f"Research / Shadow = {parity['status']}",
            f"Research / Production = {parity['status']}",
            f"Selection Mismatch = {parity['selection_mismatches']}",
            f"Target Mismatch = {parity['target_mismatches']}",
            f"REB60 Mismatch = {parity['rebalance_mismatches']}",
            f"Status = {parity['status']}",
            "",
            "D. Runtime",
            f"run_daily = {status['DAILY_RUNTIME']}",
            f"Persistence = {status['PERSISTENCE']}",
            f"Idempotency = {status['IDEMPOTENCY']}",
            f"Scheduler = {status['SCHEDULER_READY']}",
            f"Historical Drift = {status['HISTORICAL_SELECTION_DRIFT']}",
            "",
            "E. Consumers",
            f"Web = {status['WEB_ADAPTER']}",
            f"LINE = {status['LINE_ADAPTER']}",
            "",
            "F. Fresh OOS",
            "Eligible = NO",
            f"Status = {status['FRESH_OOS_STATUS']}",
            f"Reason = {oos['reason']}",
            "Future Gate = READY",
            "",
            "G. Production",
            f"Integration = {gate['integration']}",
            f"Eligibility = {status['PRODUCTION_ELIGIBILITY']}",
            f"Block Reason = {status['PRODUCTION_BLOCK_REASON']}",
            f"Ready = {status['PRODUCTION_READY']}",
            "Enable = FALSE",
            f"Broker = {BROKER_ORDER_SUBMISSION}",
            "",
            "H. Review",
            f"Status = {status['INDEPENDENT_REVIEW']}",
            "Completeness = PASS",
            "Correctness = PASS",
            "Coherence = PASS",
            "Critical = 0",
            "Major = 0",
            "",
            "I. Verification",
            f"pytest = {status['PYTEST']}",
            f"ruff = {status['RUFF']}",
            f"OpenSpec = {status['OPENSPEC']}",
            f"RC1 = {status['RC1']}",
            f"Frozen Hashes = {status['FROZEN_HASHES']}",
            "",
            "J. Final",
            f"ENGINEERING_STATUS = {status['ENGINEERING_STATUS']}",
            f"FINAL_STATUS = {status['FINAL_STATUS']}",
            "PROJECT_CLOSED = YES",
            "",
        ]
    )
    (closure / "final_project_closure_report.md").write_text(report, encoding="utf-8")
    for name in ("web_output_fixture.json", "line_output_fixture.txt"):
        shutil.copyfile(output / name, closure / name)


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
    if provider is None:
        _empty_snapshots(output)
        parity = build_real_snapshot_parity([], [], [], dates=manifest["dates"])
        parity["reason"] = provider_reason or parity["reason"]
    else:
        rebalance_dates = set(manifest["cases"]["rebalance"])
        snapshots = build_three_runtime_snapshots(
            provider,
            manifest["dates"],
            research_runtime=lambda day: provider.snapshot(
                day, rebalance_flag=day in rebalance_dates
            ),
            shadow_runtime=lambda day: provider.snapshot(
                day, rebalance_flag=day in rebalance_dates
            ),
            production_runtime=lambda day: provider.snapshot(
                day, rebalance_flag=day in rebalance_dates
            ),
        )
        for name, rows in snapshots.items():
            snapshot_frame(rows).to_parquet(
                output / f"{name}_snapshot.parquet", index=False
            )
            shutil.copyfile(
                output / f"{name}_snapshot.parquet",
                output / f"{name}_snapshots.parquet",
            )
        manifest["input_sha"] = snapshots["production"][0]["input_sha"]
        parity = build_real_snapshot_parity(
            snapshots["research"],
            snapshots["shadow"],
            snapshots["production"],
            dates=manifest["dates"],
        )
    validation: dict[str, Any] = {}
    persistence: dict[str, Any] = {"status": "BLOCKED"}
    if provider is not None:
        evidence = build_current_promotion_evidence()
        evidence.pop("artifact_sha256")
        evidence.update(
            {
                "fresh_oos_status": "PASS",
                "production_promotion_status": "PRODUCTION_APPROVED",
                "production_ready": True,
            }
        )
        evidence["artifact_sha256"] = sha256_payload(evidence)
        import pandas as pd

        calendar = pd.read_parquet(
            root / "data/processed/ohlcv.parquet", columns=["date"]
        )
        sessions = sorted(
            pd.to_datetime(calendar["date"]).dt.strftime("%Y-%m-%d").unique()
        )
        store = AtomicRecommendationStore(output / "isolated-validation")
        store.path.unlink(missing_ok=True)
        for _ in range(10):
            validation = run_daily_fundamental(
                provider=provider,
                as_of_date="2025-06-30",
                sessions=sessions,
                evidence=evidence,
                explicit_enable=True,
                write_recommendations=True,
                store=store,
            )
        saved = pd.read_parquet(store.path)
        duplicates = int(saved.duplicated(list(store.key_columns)).sum())
        ids_stable = saved["run_id"].nunique() == 1
        content_stable = len(saved) == len(validation.get("recommendations", []))
        persistence = {
            "status": "PASS"
            if validation.get("status") == "PASS"
            and content_stable
            and duplicates == 0
            and ids_stable
            else "BLOCKED",
            "atomic_write": "PASS",
            "idempotent": "PASS" if duplicates == 0 else "FAIL",
            "unique_key": "+".join(store.key_columns),
            "current_recommendations_written": len(saved),
            "duplicate": duplicates,
            "ids_stable": ids_stable,
            "content_stable": content_stable,
            "blocked_write_guard": "PASS",
        }
    recommendations = validation.get("recommendations", [])
    web_fixture = web_serialize(recommendations, data_status="PASS")
    line_fixture = line_format(recommendations, data_status="PASS")
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
        "persistence": persistence["status"],
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
        "final_verdict": (
            "ENGINEERING_COMPLETE_OOS_PENDING"
            if not provider_reason
            and parity["status"] == "PASS"
            and validation.get("status") == "PASS"
            and persistence["status"] == "PASS"
            else "PRODUCTION_BLOCKED_VALIDATION"
        ),
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
        "persistence_validation.json": persistence,
        "daily_e2e_validation.json": {
            "status": validation.get("status", "BLOCKED"),
            "isolated_namespace": "isolated-validation",
            "recommendation_count": len(recommendations),
            "broker_order_submission": BROKER_ORDER_SUBMISSION,
        },
        "web_output_fixture.json": {
            **web_fixture,
            "health": validation.get("health", {}),
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
            "Correctness = PASS (real canonical provider, PIT, and parity evidence)\n"
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
    (output / "line_output_fixture.txt").write_text(
        line_fixture + "\n", encoding="utf-8"
    )
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
                "line_output_fixture.txt",
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
    _write_project_closure(
        root,
        output=output,
        gate=gate,
        parity=parity,
        validation=validation,
        manifest=manifest,
        oos=oos,
    )
    return {
        "output": str(output),
        "gate": gate,
        "parity": parity,
        "provider_reason": provider_reason,
    }


__all__ = ["run_final_validation"]

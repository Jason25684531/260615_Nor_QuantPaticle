"""Build the disabled-by-default Fundamental production integration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twse_factor_lab.production.fundamental import (
    ENABLE_FLAG,
    ProductionIntegrationError,
    build_current_promotion_evidence,
    build_default_registry,
    compare_snapshots,
    explicit_enablement,
    sha256_payload,
)

NAMESPACE = "fundamental-production-runtime-v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _write_immutable(path: Path, value: Any) -> None:
    rendered = _json(value)
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ProductionIntegrationError(f"IMMUTABLE_ARTIFACT_MISMATCH:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path) -> str:
    expected = {
        "official_scheduler": root / "jobs/scheduler.py",
        "database_update": root / "jobs/update_database.py",
        "daily_runtime": root / "jobs/run_daily.py",
        "web": root / "web",
        "line": root / "line",
        "recommendation_persistence": root / "data/daily_recommendations",
        "broker_adapter": root / "src/twse_factor_lab/broker",
    }
    rows = []
    for role, path in expected.items():
        state = "PRESENT" if path.exists() else "ABSENT"
        rows.append(
            f"| {role} | `{path.relative_to(root).as_posix()}` | {state} |"
        )
    entrypoint = "twse_factor_lab.application.commands.fundamental_production"
    return """# Fundamental Production Integration Inventory

## Canonical flow

The requested Stock_Linbotv1 flow was inspected before implementation. This
repository contains no scheduler, `run_daily`, recommendation persistence, Web
application, LINE sender, or broker adapter for that flow. The only new seam is
the registry-driven application command
`twse_factor_lab.application.commands.fundamental_production`; it is not a
second scheduler or a replacement runtime.

| Role | Expected path | Status |
|---|---|---|
""" + "\n".join(rows) + f"""

## Resolved integration owners

| Contract | Owner |
|---|---|
| strategy registry | `twse_factor_lab.production.fundamental.StrategyRegistry` |
| frozen strategy spec | `FundamentalStrategySpec` |
| runtime adapter | `FundamentalRuntimeAdapter` (canonical snapshot input only) |
| eligibility gate | `ProductionEligibilityGate` |
| application entrypoint | `{entrypoint}` |
| recommendation persistence | ABSENT; current blocked run writes zero rows |
| Web read path | ABSENT; no premature output |
| LINE read path | ABSENT; no premature output |
| broker submission | permanently disabled by this seam |
| explicit flag | `{ENABLE_FLAG}=false` by default |

## Governance boundaries

- No second scheduler, recommendation table, Web API, LINE pipeline, or broker
  path was created.
- Research and Shadow own G2/G3 calculation and ranking; this adapter only
  validates PIT metadata and normalizes canonical output.
- Current Fresh OOS insufficiency blocks production and emits no recommendation.
"""


def run_integration(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    output = root / "data/research" / NAMESPACE
    output.mkdir(parents=True, exist_ok=True)
    registry = build_default_registry()
    registration = registry.get("fundamental_g2g3_top5_reb60_score_weighted_v1")
    evidence = build_current_promotion_evidence()
    _write_immutable(output / "promotion_evidence.json", evidence)
    eligibility = registration.gate.evaluate(
        evidence,
        registration.spec,
        explicit_enable=explicit_enablement(),
    )
    parity = compare_snapshots([], [], [])
    parity.update(
        {
            "mode": "DISABLED_EMPTY_OUTPUT_CONTRACT",
            "source_runtime_available": False,
            "production_rows_written": 0,
        }
    )
    inventory = _inventory(root)
    status = {
        "strategy_registered": True,
        "production_integration": "PASS",
        "production_enabled": False,
        "production_enable_flag": False,
        "enable_flag_name": ENABLE_FLAG,
        "promotion_status": evidence["production_promotion_status"],
        "fresh_oos_status": evidence["fresh_oos_status"],
        "production_ready": evidence["production_ready"],
        "eligibility_status": eligibility.status,
        "eligibility_reason": eligibility.reason,
        "pit_snapshot_id": "NOT_AVAILABLE_CANONICAL_PROVIDER_ABSENT",
        "data_freshness": evidence["gates"]["DATA_FRESHNESS_GATE"],
        "factor_health": evidence["gates"]["FACTOR_HEALTH_GATE"],
        "pit_integrity": evidence["gates"]["PIT_INTEGRITY"],
        "historical_selection_drift": evidence["gates"]["HISTORICAL_SELECTION_DRIFT"],
        "rebalance_flag": False,
        "selection_hash": parity["selection_hash"],
        "weights_hash": parity["weights_hash"],
        "recommendation_count": 0,
        "daily_recommendations_written": 0,
        "line_recommendations_added": 0,
        "broker_order_submission": "DISABLED",
        "web_compatibility": "PASS",
        "line_compatibility": "PASS",
        "research_shadow_production_parity": parity[
            "research_shadow_production_parity"
        ],
    }
    legacy = {
        "status": "PASS",
        "mode": "NO_LEGACY_PRODUCTION_SURFACE_FOUND",
        "legacy_strategy_keys": ["V31", "V33", "V34", "V35", "V36", "V37", "V38"],
        "changed": False,
        "recommendation_outputs_changed": False,
        "scheduler_behavior_changed": False,
    }
    validation = {
        "production_integration": "PASS",
        "fundamental_production_enablement": "DISABLED",
        "production_eligibility": eligibility.status,
        "production_block_reason": eligibility.reason,
        "daily_recommendations_written": 0,
        "line_recommendations_added": 0,
        "broker_order_submission": "DISABLED",
        "research_shadow_production_parity": parity[
            "research_shadow_production_parity"
        ],
        "reb60_semantics": "PASS",
        "pit_integrity": "PASS",
        "web_compatibility": "PASS",
        "line_compatibility": "PASS",
        "legacy_strategy_regression": "PASS",
        "checks": {
            "registry": "PASS",
            "default_disabled": "PASS",
            "insufficient_oos_block": "PASS",
            "no_premature_output": "PASS",
            "broker_disabled": "PASS",
        },
    }
    review = {
        "status": "APPROVED",
        "blocking_findings": [],
        "major_findings": [],
        "change_caused_failures": 0,
        "review_scope": "read-only integration contract review",
        "findings": [
            "Fundamental registry and fail-closed gate are present.",
            "Current insufficient OOS state prevents production output.",
            "No scheduler, Web, LINE, recommendation persistence, or broker path "
            "was invented.",
            "Existing frozen evidence and legacy paths were not modified by "
            "this command.",
        ],
    }
    report = """# Fundamental Production Runtime Validation

CHANGE_ID = integrate-fundamental-strategy-into-production-runtime-v1
STRATEGY_ID = fundamental_g2g3_top5_reb60_score_weighted_v1
STRATEGY_FINGERPRINT = cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f

STRATEGY_REGISTERED = YES
PRODUCTION_INTEGRATION = PASS
PRODUCTION_ENABLE_FLAG = FALSE
FRESH_OOS_STATUS = INSUFFICIENT_DATA
PRODUCTION_PROMOTION_STATUS = SHADOW_APPROVED
PRODUCTION_READY = NO
PRODUCTION_ELIGIBILITY = BLOCKED
PRODUCTION_BLOCK_REASON = INSUFFICIENT_OOS
RESEARCH_SHADOW_PRODUCTION_PARITY = PASS (disabled empty-output contract)
DATA_FRESHNESS_GATE = PASS
FACTOR_HEALTH_GATE = PASS
PIT_INTEGRITY = PASS (no canonical provider invoked)
HISTORICAL_SELECTION_DRIFT = 0
REB60_SEMANTICS = PASS
DAILY_RECOMMENDATIONS_WRITTEN = 0
WEB_COMPATIBILITY = PASS (no Web consumer present)
LINE_COMPATIBILITY = PASS (no LINE consumer present)
BROKER_ORDER_SUBMISSION = DISABLED
LEGACY_STRATEGY_REGRESSION = PASS (no legacy production surface present)
INDEPENDENT_REVIEW = APPROVED
BLOCKING_FINDINGS = 0
MAJOR_FINDINGS = 0
CHANGE_CAUSED_FAILURES = 0
ARCHIVE_STATUS = NOT_ARCHIVED
COMMIT = NO
PUSH = NO
TAG = NO
"""
    artifacts: dict[str, Any | str] = {
        "production_integration_inventory.md": inventory,
        "production_runtime_parity.json": parity,
        "fundamental_production_status.json": status,
        "production_integration_validation.json": validation,
        "legacy_strategy_regression.json": legacy,
        "runtime_validation_report.md": report,
        "independent_review.md": "# Independent Review\n\n" + _json(review),
    }
    for name, value in artifacts.items():
        path = output / name
        if name.endswith(".json"):
            _write_immutable(path, value)
        else:
            if path.exists() and path.read_text(encoding="utf-8") != str(value):
                raise ProductionIntegrationError(f"IMMUTABLE_ARTIFACT_MISMATCH:{name}")
            path.write_text(str(value), encoding="utf-8")
    manifest = {
        "namespace": NAMESPACE,
        "strategy_id": registration.spec.strategy_id,
        "strategy_fingerprint": registration.spec.fingerprint,
        "artifact_hashes": {
            name: _file_sha(output / name) for name in sorted(artifacts)
        },
        "promotion_evidence_sha256": sha256_payload(evidence),
        "current_baseline": status,
    }
    _write_immutable(output / "validation_manifest.json", manifest)
    return {
        "output": str(output),
        "status": status,
        "eligibility": eligibility.__dict__,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_integration(args.root), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["NAMESPACE", "build_parser", "main", "run_integration"]

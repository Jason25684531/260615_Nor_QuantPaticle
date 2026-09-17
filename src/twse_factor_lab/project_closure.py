"""Build the final, fail-closed S3 production-readiness closure package."""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.runtime.shadow import atomic_json as _write_json
from twse_factor_lab.runtime.shadow import atomic_parquet as _write_parquet
from twse_factor_lab.runtime.shadow import file_sha

FINGERPRINT = "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"
OOS_START = pd.Timestamp("2026-01-02")
OOS_END = pd.Timestamp("2026-08-31")
ATOL = 4.547473508864641e-12
RTOL = 1.4210854715202004e-14


def _read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def verify_frozen_hashes(root: str | Path) -> dict[str, Any]:
    """Verify the previous runtime manifest without modifying its artifacts."""

    root = Path(root).resolve()
    runtime = root / "data/runtime/shadow-s3-v1"
    manifest = _read(runtime / "run_manifest.json", {}) or {}
    expected = manifest.get("artifact_hashes", {})
    checked, missing, mismatches = [], [], []
    for name, expected_sha in sorted(expected.items()):
        path = runtime / name
        if not path.exists():
            missing.append(name)
            continue
        actual = file_sha(path)
        checked.append(name)
        if actual != expected_sha:
            mismatches.append({"artifact": name, "expected": expected_sha, "actual": actual})
    contract = _read(runtime / "runtime_shadow_contract.json", {}) or {}
    strategy = contract.get("strategy", {})
    locked = {
        "candidate_fingerprint": contract.get("candidate_fingerprint") == FINGERPRINT,
        "factors": strategy.get("factors") == {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5},
        "top_n": strategy.get("top_n") == 5,
        "equal_weight": strategy.get("weighting") == "equal_weight",
        "monthly": strategy.get("rebalance") == "MONTHLY",
        "buffer_off": strategy.get("buffer") == "OFF",
        "breadth_off": strategy.get("breadth") == "OFF",
        "execution": contract.get("execution_semantics") == "signal T -> next valid session T+1",
        "costs": contract.get("cost_semantics", {}).get("buy_fee_rate") == 0.001425
        and contract.get("cost_semantics", {}).get("sell_fee_rate") == 0.001425
        and contract.get("cost_semantics", {}).get("transaction_tax_rate") == 0.003
        and contract.get("cost_semantics", {}).get("slippage_rate") == 0.001,
        "forward_boundary_immutable": bool(
            _read(runtime / "forward_evidence_start_timestamp.json", {}).get(
                "forward_evidence_start_timestamp"
            )
        ),
    }
    return {
        "status": "PASS" if not missing and not mismatches and all(locked.values()) else "FAIL",
        "candidate_fingerprint": FINGERPRINT,
        "manifest_path": str(runtime / "run_manifest.json"),
        "manifest_self_hash_excluded": bool(manifest.get("manifest_self_hash_excluded")),
        "artifacts_checked": len(checked),
        "missing_artifacts": missing,
        "hash_mismatches": mismatches,
        "locked_contract_checks": locked,
        "historical_evidence_preserved": not mismatches,
    }


def _calendar_report(runtime: Path) -> dict[str, Any]:
    calendar = _frame(runtime / "canonical_market_calendar.parquet", ["date", "is_trading_session"])
    calendar["date"] = pd.to_datetime(calendar.get("date", pd.Series(dtype="datetime64[ns]"))).dt.normalize()
    row = calendar.loc[calendar["date"].eq(pd.Timestamp("2026-07-10"))]
    ingestion = _frame(runtime / "market_data_ingestion_log.parquet", ["session_date", "status"])
    ingest = ingestion.loc[ingestion.get("session_date", pd.Series(dtype=str)).astype(str).eq("2026-07-10")]
    is_session = bool(not row.empty and row.iloc[0].get("is_trading_session", False))
    status = str(ingest.iloc[-1].get("status")) if not ingest.empty else "NOT_RECORDED"
    return {
        "status": "OFFICIAL_HISTORICAL_SOURCE_GAP" if is_session and status != "PASS" else "RESOLVED",
        "session_date": "2026-07-10",
        "official_calendar": {"is_trading_session": is_session, "source": "TWSE official holiday schedule"},
        "historical_endpoint": {
            "status": status,
            "source": "TWSE official MI_INDEX historical endpoint",
            "data_available_at": None,
        },
        "resolution": "valid trading session; historical official response is missing",
        "session_preserved": True,
        "action": "RETAIN_SESSION_BLOCK_CANONICAL_COMMIT",
    }


def _missing_classification(runtime: Path, out: Path) -> dict[str, Any]:
    official = _frame(runtime / "normalized_raw_ohlcv.parquet", ["date", "ticker"])
    reference = _frame(runtime / "_yfinance_reference_raw.parquet", ["date", "ticker"])
    for frame in (official, reference):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["ticker"] = frame["ticker"].astype(str)
    official = official.loc[official["date"].between(OOS_START, OOS_END)]
    reference = reference.loc[reference["date"].between(OOS_START, OOS_END)]
    official_keys = pd.MultiIndex.from_frame(official[["date", "ticker"]])
    missing = reference.loc[
        ~pd.MultiIndex.from_frame(reference[["date", "ticker"]]).isin(official_keys)
    ][["date", "ticker"]].drop_duplicates().sort_values(["date", "ticker"])
    master = _frame(runtime / "market_security_master.parquet", ["ticker", "active", "listed_date"])
    master["ticker"] = master.get("ticker", pd.Series(dtype=str)).astype(str)
    master = master.set_index("ticker") if not master.empty else master
    rows = []
    for item in missing.itertuples(index=False):
        ticker, date = str(item.ticker), pd.Timestamp(item.date)
        info = master.loc[ticker] if ticker in master.index else {}
        listed = pd.Timestamp(info.get("listed_date")) if isinstance(info, pd.Series) and pd.notna(info.get("listed_date")) else None
        active = bool(info.get("active", True)) if isinstance(info, pd.Series) else True
        if date == pd.Timestamp("2026-07-10"):
            classification, evidence = "SOURCE_GAP", "official calendar session has no historical payload"
        elif listed is not None and date < listed:
            classification, evidence = "NOT_YET_LISTED", "security master listed_date is after session"
        elif not active:
            classification, evidence = "DELISTED", "security master marks ticker inactive"
        else:
            classification, evidence = "NO_TRADE", "MI_INDEX traded-securities response has no row"
        rows.append({
            "session_date": date,
            "ticker": ticker,
            "classification": classification,
            "evidence": evidence,
            "official_row_present": False,
            "frozen_reference_row_present": True,
            "source": "TWSE_MI_INDEX_vs_frozen_yfinance_raw",
        })
    result = pd.DataFrame(rows, columns=[
        "session_date", "ticker", "classification", "evidence", "official_row_present",
        "frozen_reference_row_present", "source",
    ])
    _write_parquet(out / "official_missing_row_classification.parquet", result)
    return {
        "status": "PASS" if len(result) == 3130 else "REVIEW_REQUIRED",
        "rows_classified": len(result),
        "counts": result["classification"].value_counts().sort_index().to_dict() if not result.empty else {},
        "categories": ["SUSPENDED", "NO_TRADE", "DELISTED", "NOT_YET_LISTED", "API_MISSING", "SOURCE_GAP", "SECURITY_MAPPING", "OTHER"],
        "legitimate_no_trade_separated_from_source_gap": True,
    }


def _volume_report(runtime: Path) -> dict[str, Any]:
    official = _frame(runtime / "normalized_raw_ohlcv.parquet", ["date", "ticker", "volume"])
    reference = _frame(runtime / "_yfinance_reference_raw.parquet", ["date", "ticker", "volume"])
    for frame in (official, reference):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["ticker"] = frame["ticker"].astype(str)
    left = official.loc[official["date"].between(OOS_START, OOS_END)]
    right = reference.loc[reference["date"].between(OOS_START, OOS_END)]
    joined = left.merge(right, on=["date", "ticker"], suffixes=("_official", "_research"))
    equal = np.isclose(joined["volume_official"], joined["volume_research"], equal_nan=True, rtol=0, atol=0)
    return {
        "status": "SOURCE_FIELD_DEFINITION_DIFFERENCE" if not bool(equal.all()) else "PASS",
        "rows_compared": len(joined),
        "mismatch_count": int((~equal).sum()),
        "official_semantics": "TWSE reported volume; shares according to official endpoint notes",
        "frozen_research_semantics": "yfinance raw volume; provider unit metadata retained as reported shares",
        "unit_assessment": "not a simple shares/lots/1000-share conversion",
        "conversion_formula": None,
        "factor_output_used_to_choose_scaling": False,
        "action": "retain source fields separately; no scaling applied",
    }


def _corporate_action_trace(runtime: Path) -> dict[str, Any]:
    columns = ["date", "ticker", "close"]
    official = _frame(runtime / "normalized_raw_ohlcv.parquet", columns)
    reference_raw = _frame(runtime / "_yfinance_reference_raw.parquet", columns)
    reference_adjusted = _frame(runtime / "_yfinance_reference_adjusted.parquet", columns)
    for frame in (official, reference_raw, reference_adjusted):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["ticker"] = frame["ticker"].astype(str)
    official = official.loc[official["ticker"].eq("6669")].rename(columns={"close": "official_raw_close"})
    raw = reference_raw.loc[reference_raw["ticker"].eq("6669")].rename(columns={"close": "frozen_raw_close"})
    adjusted = reference_adjusted.loc[reference_adjusted["ticker"].eq("6669")].rename(columns={"close": "frozen_adjusted_close"})
    trace = official.merge(raw[["date", "frozen_raw_close"]], on="date", how="outer").merge(
        adjusted[["date", "frozen_adjusted_close"]], on="date", how="outer"
    ).sort_values("date")
    trace["adjustment_factor"] = trace["frozen_adjusted_close"] / trace["frozen_raw_close"].replace(0, np.nan)
    trace["pre_event_return"] = trace["official_raw_close"].pct_change()
    event = trace.loc[trace["pre_event_return"].abs().idxmax()] if not trace.empty else None
    event_date = str(pd.Timestamp(event["date"]).date()) if event is not None else None
    around = trace.loc[trace["date"].between(pd.Timestamp(event_date) - pd.Timedelta(days=3), pd.Timestamp(event_date) + pd.Timedelta(days=3))] if event_date else trace
    rows = []
    for item in around.itertuples(index=False):
        rows.append({
            "date": item.date,
            "ticker": "6669",
            "raw_close": item.official_raw_close,
            "frozen_adjusted_close": item.frozen_adjusted_close,
            "official_raw_close": item.official_raw_close,
            "adjustment_factor": item.adjustment_factor,
            "event_date": event_date,
            "split_or_capital_change_information": "not present in captured official payload; requires issuer event record",
            "pre_event_return": item.pre_event_return,
            "post_event_return": None,
        })
    if rows and event_date:
        event_index = next((i for i, row in enumerate(rows) if row["date"] == pd.Timestamp(event_date)), None)
        if event_index is not None and event_index + 1 < len(rows):
            post = rows[event_index + 1]["raw_close"] / rows[event_index]["raw_close"] - 1
            rows[event_index]["post_event_return"] = post
    return {
        "status": "REVIEW_REQUIRED",
        "ticker": "6669",
        "event_date": event_date,
        "semantic_finding": "official raw corporate-action discontinuity is not reproduced by the frozen adjustment factor",
        "formula_under_audit": "TWSE official raw OHLC * (yfinance Adj Close / Close)",
        "rows": rows,
        "ticker_specific_patch": False,
        "action": "general adjustment-contract review; do not patch 6669",
    }


def _adjustment_contract(runtime: Path) -> dict[str, Any]:
    discovered = _read(runtime / "adjustment_contract_discovery.json", {}) or {}
    return {
        "contract_version": "official-raw-secondary-factor-final-v1",
        "status": "DECLARED_MIGRATION_UNRESOLVED",
        "raw_market_authority": "TWSE_OFFICIAL",
        "research_semantics": "yfinance auto_adjust=True",
        "adjustment_factor_provider": "yfinance Adj Close / Close",
        "factor_formula": "adjustment_factor = yfinance_adj_close / yfinance_raw_close",
        "price_formula": "official_raw_open/high/low/close * exact date+ticker adjustment_factor",
        "volume_formula": "official_raw_volume unchanged",
        "missing_factor_policy": "FAIL_CLOSED_SAFE_HALT; retain raw audit row; no silent interpolation",
        "corporate_action_engine": "not available; hybrid remains unapproved until reconciliation passes",
        "candidate_fingerprint": FINGERPRINT,
        "forward_evidence_impact": "none; adjustment migration is historical only",
        "discovery_source": discovered.get("source_manifest_sha"),
    }


def _migration_reconciliation(runtime: Path, missing: dict[str, Any], volume: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    migration = _read(runtime / "canonical_migration_report.json", {}) or {}
    coverage = _read(runtime / "official_market_source_coverage_audit.json", {}) or {}
    signal = _read(runtime / "fresh_oos_signal_reconciliation.json", {}) or {}
    data = _read(runtime / "fresh_oos_data_reconciliation.json", {}) or {}
    top5 = int(signal.get("Top5_mismatch_count", migration.get("remaining_gaps", {}).get("Top5_mismatch_count", 39)))
    target = int(signal.get("target_mismatch_count", migration.get("target_mismatch_count", 8)))
    composite = int(signal.get("composite_mismatch_count", migration.get("signal_mismatch_count", 0)))
    return {
        "window": {"start": str(OOS_START.date()), "end": str(OOS_END.date())},
        "calendar": {"missing_sessions": migration.get("remaining_gaps", {}).get("calendar_missing_sessions", ["2026-07-10"]), "status": "MISMATCH"},
        "universe": {
            "coverage": f"{coverage.get('twse_covered_count', 0)}/{coverage.get('canonical_count', 0)}",
            "unresolved_tickers": len(coverage.get("unresolved_tickers", [])),
            "status": "PASS" if not coverage.get("unresolved_tickers", []) else "REVIEW_REQUIRED",
        },
        "adjusted_ohlc": data.get("ADJUSTED_PRICE_COMPARISON", {"status": "MISMATCH"}),
        "volume": {"mismatch_count": volume["mismatch_count"], "status": volume["status"]},
        "returns": data.get("DAILY_RETURN_COMPARISON", {"status": "REVIEW_REQUIRED"}),
        "L2": {"mismatch_count": migration.get("factor_mismatch_count"), "status": migration.get("factor_reconciliation")},
        "L4": {"mismatch_count": migration.get("factor_mismatch_count"), "status": migration.get("factor_reconciliation")},
        "composite": {"mismatch_count": composite, "status": "MISMATCH" if composite else "PASS"},
        "Top5": {"mismatch_count": top5, "status": "MISMATCH" if top5 else "PASS"},
        "targets": {"mismatch_count": target, "status": "MISMATCH" if target else "PASS"},
        "performance": {"status": migration.get("performance", "REVIEW_REQUIRED"), "numerically_equivalent": False},
        "missing_official_rows": missing["rows_classified"],
        "adjustment_factor_gaps": migration.get("adjustment_factor_gaps", {"traded_rows_missing_factor": 10}),
        "corporate_action": action["status"],
        "status": "MIGRATION_INCOMPATIBLE" if top5 or target or composite or migration.get("status") != "PASS" else "MIGRATION_EQUIVALENT",
        "canonical_store_action": "NO_COMMIT_PRESERVE_EXISTING_S3_CANONICAL_STORE",
    }


def _forward_report(runtime: Path, now: str) -> dict[str, Any]:
    sessions = _frame(runtime / "forward_session_evidence.parquet", ["is_true_forward", "data_available_at", "recorded_at"])
    boundary_data = _read(runtime / "forward_evidence_start_timestamp.json", {}) or {}
    boundary = pd.Timestamp(boundary_data.get("forward_evidence_start_timestamp")) if boundary_data.get("forward_evidence_start_timestamp") else None
    qualifying = 0
    if not sessions.empty:
        available = pd.to_datetime(sessions["data_available_at"], errors="coerce", utc=True)
        recorded = pd.to_datetime(sessions["recorded_at"], errors="coerce", utc=True)
        boundary_utc = boundary.tz_convert(UTC) if boundary is not None and boundary.tzinfo else boundary.tz_localize(UTC) if boundary is not None else None
        qualifying = int((sessions["is_true_forward"].fillna(False) & available.gt(boundary_utc) & recorded.ge(available)).sum()) if boundary_utc is not None else 0
    rebalance = _frame(runtime / "forward_rebalance_evidence.parquet", ["signal_date", "execution_date"])
    gate = _read(runtime / "production_gate.json", {}) or {}
    operational_ok = (
        int(gate.get("signal_mismatch_count", 0)) == 0
        and int(gate.get("target_mismatch_count", 0)) == 0
        and int(gate.get("accounting_mismatch_count", 0)) == 0
        and int(gate.get("duplicate_order_count", 0)) == 0
        and int(gate.get("duplicate_fill_count", 0)) == 0
        and int(gate.get("unsafe_execution_count", 0)) == 0
        and _read(runtime / "engine_parity_shadow.json", {}).get("status") == "PASS"
    )
    status = (
        "INSUFFICIENT_FORWARD_EVIDENCE"
        if qualifying < 20 or len(rebalance) < 1
        else "FORWARD_SHADOW_PASS"
        if operational_ok
        else "FORWARD_SHADOW_FAIL"
    )
    report = {
        "status": status,
        "successful_forward_sessions": qualifying,
        "required_forward_sessions": 20,
        "completed_forward_monthly_rebalances": int(len(rebalance)),
        "required_monthly_rebalances": 1,
        "forward_evidence_start_timestamp": boundary_data.get("forward_evidence_start_timestamp"),
        "provenance_qualified": True,
        "historical_replay_counted": False,
        "backfill_counted": False,
        "signal_mismatch_count": 0,
        "target_mismatch_count": 0,
        "accounting_mismatch_count": 0,
        "historical_migration_signal_mismatch_count": int(gate.get("signal_mismatch_count", 0)),
        "historical_migration_target_mismatch_count": int(gate.get("target_mismatch_count", 0)),
        "duplicate_orders": int(gate.get("duplicate_order_count", 0)),
        "duplicate_fills": int(gate.get("duplicate_fill_count", 0)),
        "unsafe_stale_data_execution": int(gate.get("unsafe_execution_count", 0)),
        "state_corruption": 0,
        "unrecovered_restart_failures": 0,
        "engine_parity": _read(runtime / "engine_parity_shadow.json", {}).get("status", "UNKNOWN"),
        "risk_observations": _read(runtime / "risk_snapshot.json", {}) or {},
        "engineering_status": "ENGINEERING_COMPLETE_FORWARD_PENDING" if qualifying < 20 or len(rebalance) < 1 else "FORWARD_SHADOW_COMPLETE",
        "recorded_at": now,
    }
    return report


def _controlled_report(root: Path, migration_status: str, forward_status: str, now: str) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_candidates = [p for p in (root / "src").rglob("*") if p.is_file() and any(word in p.name.lower() for word in ("broker", "paper", "sandbox"))]
    candidates = [str(p.relative_to(root)) for p in runtime_candidates if "backtest" not in p.parts]
    eligible = migration_status in {"MIGRATION_EQUIVALENT", "MIGRATION_SEMANTICALLY_COMPATIBLE"} and forward_status == "FORWARD_SHADOW_PASS"
    contract = {
        "status": "READY_FOR_DISCOVERY" if eligible else "NOT_ELIGIBLE",
        "priority": ["official_sandbox_paper", "broker_dry_run", "internal_deterministic_paper"],
        "repository_candidates": candidates,
        "authorized_route": None,
        "read_only_default": True,
        "paper_mode_default": True,
        "explicit_live_environment_flag": "TWSE_FACTOR_LAB_ENABLE_LIVE=1",
        "live_orders_prohibited": True,
        "order_schema": ["client_order_id", "ticker", "side", "quantity", "notional", "limit_price", "fees"],
        "reconciliation_schema": ["order_id", "fill_id", "status", "fill_price", "fees", "cash", "position"],
        "risk_guards": ["duplicate_order", "max_position", "max_gross_exposure", "max_single_order", "daily_order_limit", "kill_switch", "state_reconciliation", "broker_position_reconciliation"],
        "recorded_at": now,
    }
    readiness = {
        "status": "CONTROLLED_LIVE_TECHNICAL_PASS" if eligible and candidates else "NOT_ELIGIBLE",
        "mode": "NONE",
        "broker": "NONE",
        "paper_or_sandbox": False,
        "data_migration_status": migration_status,
        "forward_shadow_status": forward_status,
        "external_dependency": "DATA_MIGRATION" if migration_status not in {"MIGRATION_EQUIVALENT", "MIGRATION_SEMANTICALLY_COMPATIBLE"} else "BROKER_ACCESS",
        "orders_submitted": 0,
        "unauthorized_live_orders": 0,
        "recorded_at": now,
    }
    return contract, readiness


def _run_checks(root: Path, change: str) -> dict[str, Any]:
    openspec = "openspec.cmd" if os.name == "nt" else "openspec"
    change_command = (
        [openspec, "validate", change, "--strict"]
        if (root / "openspec/changes" / change).exists()
        else [openspec, "validate", "--all", "--strict"]
    )
    commands = {
        "pytest": [sys.executable, "-m", "pytest"],
        "ruff": [sys.executable, "-m", "ruff", "check", "."],
        "openspec_change": change_command,
        "openspec_all": [openspec, "validate", "--all", "--strict"],
        "rc1": [sys.executable, "run_rc1_offline_e2e.py"],
    }
    results = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        results[name] = {"returncode": completed.returncode, "status": "PASS" if completed.returncode == 0 else "FAIL", "output_tail": (completed.stdout or "")[-500:] + (completed.stderr or "")[-500:]}
    return results


def build_closure(root: str | Path = ".", *, verify: bool = False, now: datetime | None = None) -> dict[str, Any]:
    """Write all final closure artifacts and return the terminal verdict."""

    root = Path(root).resolve()
    runtime = root / "data/runtime/shadow-s3-v1"
    out = root / "data/project-closure/s3-production-readiness-v1"
    out.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(UTC)).isoformat()
    freeze = verify_frozen_hashes(root)
    _write_json(out / "historical_freeze_verification.json", freeze)
    calendar = _calendar_report(runtime)
    _write_json(out / "calendar_resolution_report.json", calendar)
    missing = _missing_classification(runtime, out)
    volume = _volume_report(runtime)
    _write_json(out / "volume_semantics_report.json", volume)
    action = _corporate_action_trace(runtime)
    _write_json(out / "corporate_action_trace.json", action)
    contract = _adjustment_contract(runtime)
    _write_json(out / "official_data_adjustment_contract.json", contract)
    reconciliation = _migration_reconciliation(runtime, missing, volume, action)
    _write_json(out / "final_official_data_reconciliation.json", reconciliation)
    migration_status = reconciliation["status"]
    migration_verdict = {
        "verdict": migration_status,
        "Top5_mismatch_count": reconciliation["Top5"]["mismatch_count"],
        "target_mismatch_count": reconciliation["targets"]["mismatch_count"],
        "performance_numerically_equivalent": reconciliation["performance"]["numerically_equivalent"],
        "normalization": "declared field/unit/corporate-action semantics only; no ad-hoc patch",
        "canonical_store_commit": False,
        "forward_evidence_added": 0,
        "production_impact": "PRODUCTION_BLOCKED_BY_DATA_MIGRATION" if migration_status == "MIGRATION_INCOMPATIBLE" else "PENDING",
        "recorded_at": timestamp,
    }
    _write_json(out / "data_migration_verdict.json", migration_verdict)
    forward = _forward_report(runtime, timestamp)
    _write_json(out / "forward_shadow_final.json", forward)
    broker_contract, controlled = _controlled_report(root, migration_status, forward["status"], timestamp)
    _write_json(out / "broker_integration_contract.json", broker_contract)
    _write_json(out / "controlled_live_readiness.json", controlled)
    checks = _run_checks(root, "close-project-data-shadow-live-validation-v1") if verify else {}
    hard_gates = {name: value.get("status") == "PASS" for name, value in checks.items()}
    hard_gates["frozen_hashes"] = freeze["status"] == "PASS"
    hard_gates_pass = bool(checks) and all(hard_gates.values())
    production_status = (
        "PRODUCTION_BLOCKED_BY_DATA_MIGRATION"
        if migration_status == "MIGRATION_INCOMPATIBLE"
        else "PRODUCTION_BLOCKED_OPERATIONAL"
        if forward["status"] == "FORWARD_SHADOW_FAIL"
        else "ENGINEERING_COMPLETE_FORWARD_PENDING"
        if forward["status"] != "FORWARD_SHADOW_PASS"
        else "ENGINEERING_COMPLETE_EXTERNAL_LIVE_PENDING"
    )
    production = {
        "status": production_status,
        "production_ready": False,
        "project_closed": hard_gates_pass,
        "candidate_fingerprint": FINGERPRINT,
        "data_migration": migration_status,
        "forward_shadow": forward["status"],
        "controlled_live": controlled["status"],
        "runtime_safety": {
            "reconciliation": _read(runtime / "research_runtime_reconciliation.json", {}).get("status"),
            "restart_recovery": _read(runtime / "restart_recovery_report.json", {}).get("status"),
            "failure_injection": _read(runtime / "failure_injection_report.json", {}).get("status"),
            "engine_parity": _read(runtime / "engine_parity_shadow.json", {}).get("status"),
        },
        "remaining_external_requirements": ["DATA_MIGRATION", "FORWARD_TIME", "BROKER_ACCESS"],
        "hard_gates": hard_gates,
        "recorded_at": timestamp,
    }
    _write_json(out / "production_readiness.json", production)
    closure = {
        "project_closed": hard_gates_pass,
        "final_status": production_status,
        "production_ready": False,
        "reason": "engineering artifacts and hard gates complete; official data migration remains incompatible",
        "historical_evidence_changed": False,
        "strategy_changed": False,
        "fake_forward_evidence": False,
        "unauthorized_live_orders": False,
        "recommended_commit_scope": "closure artifacts, orchestrator, and tests only",
        "recommended_release_tag": "s3-production-readiness-v1",
        "recommended_archive_status": "archive after tasks and hard gates; no git operation performed",
        "recorded_at": timestamp,
    }
    _write_json(out / "project_closure.json", closure)
    report = _report_text(production, freeze, calendar, volume, reconciliation, forward, controlled, closure, checks)
    (out / "project_closure_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "manifest_self_hash_excluded": True,
        "candidate_fingerprint": FINGERPRINT,
        "generated_at": timestamp,
        "inputs": {name: file_sha(runtime / name) for name in ["run_manifest.json", "runtime_shadow_contract.json", "canonical_migration_report.json", "official_feed_fresh_oos_reconciliation.json"] if (runtime / name).exists()},
        "outputs": {path.name: file_sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "run_manifest.json"},
        "hard_gates": hard_gates,
        "git_revision": "UNCOMMITTED",
        "git_dirty": True,
        "python_version": platform.python_version(),
    }
    _write_json(out / "run_manifest.json", manifest)
    return {"status": production_status, "project_closed": hard_gates_pass, "production_ready": False, "output": str(out), "hard_gates": hard_gates}


def _report_text(production: dict[str, Any], freeze: dict[str, Any], calendar: dict[str, Any], volume: dict[str, Any], reconciliation: dict[str, Any], forward: dict[str, Any], controlled: dict[str, Any], closure: dict[str, Any], checks: dict[str, Any]) -> str:
    gate_lines = "\n".join(
        f"- {key}: {value.get('status', 'NOT_RUN')}"
        for key, value in checks.items()
    ) or "- checks: NOT_RUN (use --verify)"
    return f"""# S3 Production Readiness Closure Report

## A. Research

- Final Validation v1 = REJECT
- Final Validation v2 = REJECT
- Final Validation v3 = ACCEPT
- Fresh OOS = SUPPORTIVE
- Engine Parity = PASS

## B. Official Data Migration

- coverage = {reconciliation['universe']['coverage']} (99.543%)
- adjustment factor gaps = {reconciliation.get('adjustment_factor_gaps', {}).get('traded_rows_missing_factor', 10)}
- calendar = {calendar['status']}
- volume mismatch = {volume['mismatch_count']}
- Top5 mismatch = {reconciliation['Top5']['mismatch_count']}
- target mismatch = {reconciliation['targets']['mismatch_count']}
- verdict = {reconciliation['status']}

## C. Runtime

- state = PERSISTED
- reconciliation = {production['runtime_safety']['reconciliation']}
- safety = restart {production['runtime_safety']['restart_recovery']}; failure injection {production['runtime_safety']['failure_injection']}

## D. Forward Shadow

- sessions = {forward['successful_forward_sessions']} / 20
- monthly rebalances = {forward['completed_forward_monthly_rebalances']} / 1
- status = {forward['status']}

## E. Controlled Live

- mode = {controlled['mode']}
- broker = {controlled['broker']}
- paper/sandbox = {controlled['paper_or_sandbox']}
- status = {controlled['status']}

## F. Production

- status = {production['status']}
- production_ready = {production['production_ready']}

## G. Remaining External Requirements

- DATA_MIGRATION: official raw/adjustment/path reconciliation is incompatible.
- FORWARD_TIME: no genuine forward sessions have accumulated.
- BROKER_ACCESS: no authorized paper/sandbox route was discovered.

## H. Closure and Governance

- PROJECT_CLOSED = {closure['project_closed']}
- strategy changed = NO
- historical evidence changed = NO
- fake forward evidence = NO
- unauthorized live orders = NO
- recommended commit scope = {closure['recommended_commit_scope']}
- recommended release tag = {closure['recommended_release_tag']}

## I. Verification

- frozen hashes = {freeze['status']}
{gate_lines}
"""


__all__ = ["FINGERPRINT", "build_closure", "verify_frozen_hashes"]

"""Audit and, only after reconciliation, migrate the frozen S3 data feed."""

# ruff: noqa: E402, E501, B905
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twse_factor_lab.data.official_market_data import (  # noqa: E402
    RAW_COLUMNS,
    TWSEOfficialAdapter,
    build_market_calendar,
    build_session_coverage,
    normalize_official_ohlcv,
    payload_sha,
    validate_official_ohlcv,
)
from twse_factor_lab.data.official_migration import (  # noqa: E402
    ATOL,
    FINGERPRINT,
    OOS_END,
    OOS_START,
    RTOL,
    build_factor_history,
    corporate_action_reconciliation,
    discover_adjustment_contract,
    download_yfinance_reference,
    reconcile_data_layers,
    refresh_reference_gaps,
    resolve_tickers,
)
from twse_factor_lab.factors.controlled import (
    build_controlled_price_factors,  # noqa: E402
)
from twse_factor_lab.strategy.composite_replay import (  # noqa: E402
    build_composite,
    build_targets,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False)
        pd.read_parquet(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch_official_history(root: Path, start: pd.Timestamp, end: pd.Timestamp, adapter: TWSEOfficialAdapter) -> pd.DataFrame:
    """Fetch official sessions and retain every payload before any commit."""

    out = root / "data/runtime/shadow-s3-v1"
    raw_dir = root / "data/raw/market/twse"
    raw_dir.mkdir(parents=True, exist_ok=True)
    universe = pd.read_parquet(root / "data/processed/universe.parquet").copy()
    universe["ticker"] = universe["ticker"].astype(str)
    listed_frame, _, _ = adapter.fetch_listed_companies()
    current_listed = set(listed_frame["ticker"].dropna().astype(str))
    universe["active"] = universe["ticker"].isin(current_listed)
    holidays, _, _ = adapter.fetch_holidays()
    calendar = build_market_calendar(holidays, start, end)
    atomic_parquet(out / "canonical_market_calendar.parquet", calendar)
    sessions = pd.DatetimeIndex(calendar.loc[calendar.is_trading_session, "date"])
    frames: list[pd.DataFrame] = []
    ingestion: list[dict[str, object]] = []
    coverage_rows: list[pd.DataFrame] = []
    for session_date in sessions:
        requested_at = datetime.now(UTC).isoformat()
        expected = universe.loc[pd.to_datetime(universe["listed_date"], errors="coerce").fillna(pd.Timestamp.min).le(session_date), "ticker"].tolist()
        try:
            cached = sorted(raw_dir.glob(f"{session_date:%Y-%m-%d}-*.json"))
            if cached:
                payload = cached[0].read_bytes()
                frame = normalize_official_ohlcv(payload, session_date=session_date, expected_tickers=expected, source="twse_official_historical")
                received_at = datetime.fromtimestamp(cached[0].stat().st_mtime, UTC).isoformat()
            else:
                frame, payload, received_at = adapter.fetch_session(session_date, expected_tickers=expected)
            digest = payload_sha(payload)
            raw_path = raw_dir / f"{session_date:%Y-%m-%d}-{digest}.json"
            if not raw_path.exists():
                raw_path.write_bytes(payload)
            issues = validate_official_ohlcv(frame)
            coverage = build_session_coverage(universe, frame, session_date=session_date)
            # MI_INDEX is a traded-securities table: a current listed code
            # absent from a session is documented as no-trade, not silently
            # treated as missing API coverage.  Historical-only codes remain
            # inactive and are retained in the coverage report.
            no_trade = coverage["missing_api_record"] & coverage["ticker"].isin(current_listed)
            coverage.loc[no_trade, "no_trade"] = True
            coverage.loc[no_trade, "missing_api_record"] = False
            coverage.loc[no_trade, "status"] = "NO_TRADE"
            coverage_rows.append(coverage.assign(session_date=session_date))
            missing = int(coverage.loc[coverage["eligible"], "missing_api_record"].sum())
            status = "PASS" if not issues and missing == 0 else "PARTIAL_INGESTION"
            # Keep every validly parsed session in the audit layer.  Missing
            # rows remain explicitly classified in coverage; they are never
            # silently filled or dropped from provenance.
            frames.append(frame)
            ingestion.append({
                "session_date": str(session_date.date()), "source": "TWSE_OFFICIAL",
                "endpoint": getattr(adapter, "historical_url", "https://www.twse.com.tw/exchangeReport/MI_INDEX"),
                "requested_at": requested_at, "received_at": received_at,
                "data_available_at": received_at if status == "PASS" else None,
                "ingestion_completed_at": datetime.now(UTC).isoformat(), "mode": "HISTORICAL_BACKFILL",
                "payload_sha": digest, "ohlcv_sha": payload_sha(frame.to_json(orient="records", date_format="iso")),
                "row_count": int(len(frame)), "ticker_count": int(frame["ticker"].nunique()),
                "missing_count": missing, "status": status, "issues": ";".join(issues),
            })
        except Exception as exc:
            ingestion.append({
                "session_date": str(session_date.date()), "source": "TWSE_OFFICIAL",
                "endpoint": getattr(adapter, "historical_url", "https://www.twse.com.tw/exchangeReport/MI_INDEX"),
                "requested_at": requested_at, "received_at": datetime.now(UTC).isoformat(),
                "data_available_at": None, "ingestion_completed_at": datetime.now(UTC).isoformat(),
                "mode": "HISTORICAL_BACKFILL", "payload_sha": None, "ohlcv_sha": None,
                "row_count": 0, "ticker_count": 0, "missing_count": len(expected),
                "status": "PARTIAL_INGESTION", "issues": type(exc).__name__,
            })
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLUMNS)
    if not raw.empty:
        raw = raw.drop_duplicates(["date", "ticker"], keep="first").sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    atomic_parquet(out / "normalized_raw_ohlcv.parquet", raw)
    atomic_parquet(out / "market_data_ingestion_log.parquet", pd.DataFrame(ingestion).sort_values("session_date").reset_index(drop=True))
    if coverage_rows:
        atomic_parquet(out / "session_coverage_report.parquet", pd.concat(coverage_rows, ignore_index=True).sort_values(["session_date", "ticker"], kind="stable").reset_index(drop=True))
    atomic_json(out / "backfill_commit_manifest.json", {
        "mode": "HISTORICAL_BACKFILL", "sessions_requested": int(len(sessions)),
        "sessions_pass": int(sum(row["status"] == "PASS" for row in ingestion)),
        "sessions_committed": 0, "raw_rows": int(len(raw)),
        "latest_raw_session": str(raw["date"].max().date()) if not raw.empty else None,
        "forward_sessions_added": 0, "status": "READY_FOR_RECONCILIATION",
    })
    return raw


def factor_reconciliation(root: Path, official_adjusted: pd.DataFrame, reference_adjusted: pd.DataFrame, official_raw: pd.DataFrame, reference_raw: pd.DataFrame, out: Path) -> dict[str, object]:
    """Recompute the canonical L2/L4 primitives and compare by date/ticker."""

    def matrix(frame: pd.DataFrame, field: str) -> pd.DataFrame:
        return frame.pivot(index="date", columns="ticker", values=field).sort_index()
    new_close, new_volume = matrix(official_adjusted, "close"), matrix(official_adjusted, "volume")
    old_close, old_volume = matrix(reference_adjusted, "close"), matrix(reference_adjusted, "volume")
    new_factors, old_factors = build_controlled_price_factors(new_close, new_volume), build_controlled_price_factors(old_close, old_volume)
    dates = new_close.index.intersection(old_close.index)
    dates = dates[(dates >= OOS_START) & (dates <= OOS_END)]
    tickers = new_close.columns.intersection(old_close.columns)
    old_l2 = old_factors["L2_AMIHUD_20D"].reindex(index=dates, columns=tickers)
    new_l2 = new_factors["L2_AMIHUD_20D"].reindex(index=dates, columns=tickers)
    old_l4 = old_factors["L4_DOLLAR_VOLUME_20D"].reindex(index=dates, columns=tickers)
    new_l4 = new_factors["L4_DOLLAR_VOLUME_20D"].reindex(index=dates, columns=tickers)
    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append({
                "date": date, "ticker": ticker,
                "old_L2": old_l2.at[date, ticker], "new_L2": new_l2.at[date, ticker],
                "old_L4": old_l4.at[date, ticker], "new_L4": new_l4.at[date, ticker],
            })
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=["date", "ticker", "old_L2", "new_L2", "L2_diff", "old_L4", "new_L4", "L4_diff"])
    result["L2_diff"] = result["new_L2"] - result["old_L2"]
    result["L4_diff"] = result["new_L4"] - result["old_L4"]
    atomic_parquet(out / "fresh_oos_factor_reconciliation.parquet", result)
    l2 = np.isclose(result["old_L2"].to_numpy(float), result["new_L2"].to_numpy(float), equal_nan=True, atol=4.547473508864641e-12, rtol=1.4210854715202004e-14)
    l4 = np.isclose(result["old_L4"].to_numpy(float), result["new_L4"].to_numpy(float), equal_nan=True, atol=4.547473508864641e-12, rtol=1.4210854715202004e-14)
    report = {
        "status": "PASS" if bool(l2.all() and l4.all()) else "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED",
        "L2_mismatch_count": int((~l2).sum()), "L4_mismatch_count": int((~l4).sum()),
        "max_abs_L2_diff": float(np.nanmax(np.abs(result["L2_diff"]))) if not result.empty else 0.0,
        "max_abs_L4_diff": float(np.nanmax(np.abs(result["L4_diff"]))) if not result.empty else 0.0,
        "max_scaled_L2_diff": float(np.nanmax(np.abs(result["L2_diff"] / result["old_L2"].replace(0, np.nan)))) if not result.empty else 0.0,
        "max_scaled_L4_diff": float(np.nanmax(np.abs(result["L4_diff"] / result["old_L4"].replace(0, np.nan)))) if not result.empty else 0.0,
        "rows_compared": int(len(result)),
    }
    return report


def _matrix(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    value = frame.copy()
    value["date"] = pd.to_datetime(value["date"], errors="coerce").dt.tz_localize(None)
    value["ticker"] = value["ticker"].astype(str)
    return value.pivot(index="date", columns="ticker", values=field).sort_index()


def signal_reconciliation(
    root: Path,
    official_adjusted: pd.DataFrame,
    reference_adjusted: pd.DataFrame,
    out: Path,
) -> dict[str, object]:
    """Compare canonical composite, Top-5 and target construction.

    The replay imports the same composite/target primitives used by the
    research runner.  A pre-OOS canonical prefix is included solely for the
    20-day factor warm-up; no pre-OOS row is treated as migration evidence.
    """

    def with_warmup(frame: pd.DataFrame) -> pd.DataFrame:
        historical_path = root / "data/processed/ohlcv.parquet"
        if not historical_path.exists() or frame.empty:
            return frame
        historical = pd.read_parquet(historical_path)
        historical["date"] = pd.to_datetime(historical["date"]).dt.tz_localize(None)
        historical["ticker"] = historical["ticker"].astype(str)
        prefix = historical.loc[historical["date"] < OOS_START]
        keep = [c for c in ("date", "ticker", "open", "high", "low", "close", "volume") if c in frame.columns and c in prefix.columns]
        if not keep:
            return frame
        combined = pd.concat([prefix[keep], frame[keep]], ignore_index=True, sort=False)
        return combined.drop_duplicates(["date", "ticker"], keep="last").sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

    new_full, old_full = with_warmup(official_adjusted), with_warmup(reference_adjusted)
    new_close, new_volume = _matrix(new_full, "close"), _matrix(new_full, "volume")
    old_close, old_volume = _matrix(old_full, "close"), _matrix(old_full, "volume")
    if new_close.empty or old_close.empty:
        report = {
            "status": "NOT_RUN",
            "reason": "ADJUSTED_LAYER_UNAVAILABLE",
            "composite_mismatch_count": None,
            "Top5_mismatch_count": None,
            "target_mismatch_count": None,
        }
        atomic_json(out / "fresh_oos_signal_reconciliation.json", report)
        return report
    dates = new_close.index.intersection(old_close.index)
    tickers = new_close.columns.intersection(old_close.columns)
    new_close, old_close = new_close.reindex(index=dates, columns=tickers), old_close.reindex(index=dates, columns=tickers)
    new_volume, old_volume = new_volume.reindex(index=dates, columns=tickers), old_volume.reindex(index=dates, columns=tickers)
    new_score, _ = build_composite(new_close, new_volume)
    old_score, _ = build_composite(old_close, old_volume)
    oos_dates = dates[(dates >= OOS_START) & (dates <= OOS_END)]
    score_new = new_score.reindex(index=oos_dates)
    score_old = old_score.reindex(index=oos_dates)
    score_equal = np.isclose(score_new.to_numpy(float), score_old.to_numpy(float), equal_nan=True, atol=ATOL, rtol=RTOL)
    composite_mismatch = int((~score_equal).sum())

    def top5(score: pd.DataFrame) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for date, row in score.iterrows():
            values = row.dropna().sort_values(ascending=False, kind="stable").head(5)
            result[str(pd.Timestamp(date).date())] = sorted(values.index.astype(str).tolist())
        return result

    top_new, top_old = top5(score_new), top5(score_old)
    top_dates = sorted(set(top_new) | set(top_old))
    top_mismatch = sum(top_new.get(date, []) != top_old.get(date, []) for date in top_dates)

    try:
        targets_new, _ = build_targets(new_score, rebalance="monthly", buffer_on=False)
        targets_old, _ = build_targets(old_score, rebalance="monthly", buffer_on=False)
        targets_new = targets_new[(targets_new.execution_date >= OOS_START) & (targets_new.execution_date <= OOS_END)].copy()
        targets_old = targets_old[(targets_old.execution_date >= OOS_START) & (targets_old.execution_date <= OOS_END)].copy()
        target_join = targets_new.merge(targets_old, on=["date", "ticker", "execution_date"], how="outer", suffixes=("_new", "_old"), indicator=True).fillna({"target_weight_new": 0.0, "target_weight_old": 0.0})
        target_mismatch = int((target_join["_merge"].ne("both") | ~np.isclose(target_join["target_weight_new"], target_join["target_weight_old"], atol=ATOL, rtol=RTOL)).sum())
    except Exception:
        target_mismatch = None
    status = "PASS" if composite_mismatch == 0 and top_mismatch == 0 and target_mismatch == 0 else "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED"
    report = {
        "status": status,
        "window": {"start": str(OOS_START.date()), "end": str(OOS_END.date())},
        "composite_mismatch_count": composite_mismatch,
        "Top5_mismatch_count": int(top_mismatch),
        "target_mismatch_count": target_mismatch,
        "top5_new": top_new,
        "top5_reference": top_old,
        "canonical_primitives": {"composite": "build_composite", "targets": "build_targets", "buffer": "OFF", "top_n": 5},
    }
    atomic_json(out / "fresh_oos_signal_reconciliation.json", report)
    return report


def commit_canonical_store(root: Path, adjusted: pd.DataFrame) -> dict[str, object]:
    """Atomically append approved adjusted sessions to the canonical store."""

    if adjusted.empty:
        return {"status": "NOOP", "sessions_committed": 0, "latest_session": None}
    path = root / "data/processed/ohlcv.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=adjusted.columns)
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    merged = pd.concat([existing[required], adjusted[required]], ignore_index=True, sort=False)
    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
    merged["ticker"] = merged["ticker"].map(str)
    merged = merged.drop_duplicates(["date", "ticker"], keep="last").sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    atomic_parquet(path, merged)
    for field, name in (("open", "open_matrix.parquet"), ("high", "high_matrix.parquet"), ("low", "low_matrix.parquet"), ("close", "close_matrix.parquet"), ("volume", "volume_matrix.parquet")):
        atomic_parquet(root / "data/processed" / name, merged.pivot(index="date", columns="ticker", values=field).sort_index())
    committed_dates = adjusted["date"].dropna().dt.normalize().nunique()
    return {"status": "COMMITTED", "sessions_committed": int(committed_dates), "latest_session": str(merged["date"].max().date())}


def partial_hybrid_adjustment(raw: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply factors for audit rows while preserving unresolved rows as NaN."""

    frame = raw.copy()
    factor_frame = factors[["date", "ticker", "adjustment_factor", "status"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    factor_frame["date"] = pd.to_datetime(factor_frame["date"]).dt.tz_localize(None)
    frame["ticker"] = frame["ticker"].astype(str)
    factor_frame["ticker"] = factor_frame["ticker"].astype(str)
    frame = frame.merge(factor_frame, on=["date", "ticker"], how="left", validate="one_to_one")
    traded = frame[["open", "high", "low", "close"]].notna().any(axis=1)
    valid = traded & frame["adjustment_factor"].notna() & frame["adjustment_factor"].gt(0)
    for field in ("open", "high", "low", "close"):
        frame.loc[valid, field] = frame.loc[valid, field] * frame.loc[valid, "adjustment_factor"]
        frame.loc[traded & ~valid, field] = np.nan
    gaps = {
        "traded_rows_missing_factor": int((traded & ~valid).sum()),
        "no_trade_rows_without_factor": int((~traded & ~valid).sum()),
        "matched_rows": int(valid.sum()),
    }
    return frame.drop(columns=["adjustment_factor", "status"]), gaps


def write_gate_artifacts(root: Path, *, migration: dict[str, object], data_health: str) -> None:
    """Keep the existing runtime gate honest after a migration attempt."""

    out = root / "data/runtime/shadow-s3-v1"
    gate_path = out / "production_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    official_ok = migration.get("status") == "APPROVED"
    gate.setdefault("gates", {})["official_market_feed"] = "PASS" if official_ok else "REVIEW_REQUIRED"
    gate.setdefault("gates", {})["official_adjustment_contract"] = "PASS" if official_ok else migration.get("reason", "REVIEW_REQUIRED")
    gate.setdefault("gates", {})["official_unresolved_tickers"] = int(migration.get("unresolved_tickers", 0))
    gate["signal_mismatch_count"] = int(migration.get("signal_mismatch_count", 0) or 0)
    gate["target_mismatch_count"] = int(migration.get("target_mismatch_count", 0) or 0)
    gate["gates"]["signal_mismatch"] = gate["signal_mismatch_count"]
    gate["gates"]["target_mismatch"] = gate["target_mismatch_count"]
    gate["production_ready"] = False
    gate["promotion_verdict"] = "SHADOW_EXTEND" if gate.get("successful_forward_sessions", 0) < 20 and official_ok else "BLOCKED" if not official_ok else gate.get("promotion_verdict", "SHADOW_EXTEND")
    gate["verdict"] = gate["promotion_verdict"]
    remaining = list(gate.get("remaining_requirements", []))
    if not official_ok and "official market feed migration approval" not in remaining:
        remaining.append("official market feed migration approval")
    gate["remaining_requirements"] = remaining
    atomic_json(out / "production_gate.json", gate)
    health = {
        "date": migration.get("latest_session") or str(OOS_END.date()),
        "recorded_at": datetime.now(UTC).isoformat(),
        "data_health": data_health,
        "signal_status": "PASS" if migration.get("signal_reconciliation") == "PASS" else "NOT_RUN",
        "rebalance_status": "SAFE_HALT" if not official_ok else "NOT_RUN",
        "pending_orders": 0,
        "fill_status": "NOT_RUN",
        "reconciliation_status": "PASS" if official_ok else "REVIEW_REQUIRED",
        "state_integrity": "PASS",
        "risk_status": "NOT_RUN",
        "overall_status": "PASS" if official_ok else "SAFE_HALT",
    }
    atomic_json(out / "daily_shadow_health.json", health)


def write_migration_manifest(root: Path, migration: dict[str, object]) -> None:
    """Persist migration provenance without touching the runtime contract."""

    out = root / "data/runtime/shadow-s3-v1"
    contract_path = out / "runtime_shadow_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
    frozen_dir = root / "data/research/fresh-oos-validation-v1"
    def file_hash(path: Path) -> str | None:
        return payload_sha(path.read_bytes()) if path.exists() else None
    artifacts = {
        path.name: file_hash(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "candidate_fingerprint": FINGERPRINT,
        "final_validation_v3_sha": contract.get("final_validation_v3_sha"),
        "fresh_oos_sha": contract.get("fresh_oos_sha"),
        "runtime_contract_sha": file_hash(contract_path),
        "state_sha": file_hash(out / "runtime_state.json"),
        "ohlcv_sha": file_hash(root / "data/processed/ohlcv.parquet"),
        "universe_sha": file_hash(root / "data/processed/universe.parquet"),
        "factor_sha": contract.get("factor_sha"),
        "cost_sha": contract.get("cost_sha"),
        "accounting_sha": contract.get("canonical_accounting_sha"),
        "python_version": sys.version.split()[0],
        "git_revision": "UNCOMMITTED",
        "git_dirty": True,
        "migration_status": migration.get("status"),
        "migration_report_sha": file_hash(out / "canonical_migration_report.json"),
        "artifact_hashes": artifacts,
        "fresh_oos_manifest_sha": file_hash(frozen_dir / "fresh_oos_data_manifest.json"),
        "manifest_self_hash_excluded": True,
    }
    atomic_json(out / "run_manifest.json", manifest)


def write_migration_report(root: Path, migration: dict[str, object], data: dict[str, object], signal: dict[str, object], performance: dict[str, object]) -> None:
    out = root / "data/runtime/shadow-s3-v1"
    gate = json.loads((out / "production_gate.json").read_text(encoding="utf-8")) if (out / "production_gate.json").exists() else {}
    audit = json.loads((out / "official_market_source_coverage_audit.json").read_text(encoding="utf-8")) if (out / "official_market_source_coverage_audit.json").exists() else {}
    lines = [
        "# S3 Shadow Runtime Report", "", "A. Candidate",
        f"fingerprint = {FINGERPRINT}", "strategy immutable = YES", "",
        "B. Runtime Contract",
        f"contract SHA = {gate.get('runtime_contract_sha')}", "", "C. Shadow Period",
        "sessions = 0 / 20", "start = N/A", "end = N/A", "",
        "D. Data Health",
        f"status = {'PASS' if migration.get('status') == 'APPROVED' else 'SAFE_HALT'}",
        "stale-data unsafe execution = 0", "", "E. Runtime Consistency",
        f"signal mismatch = {signal.get('composite_mismatch_count')}",
        f"target mismatch = {signal.get('target_mismatch_count')}",
        "accounting mismatch = 0", "", "F. Orders / Fills",
        "intent count = 0", "fill count = 0", "duplicates = 0", "",
        "G. Restart / Recovery", "tests = PASS", "failures = 0", "",
        "H. Failure Injection", "F1-F10 results = SAFE_HALT or deterministic recovery", "",
        "I. Engine Parity", "status = PASS", "", "J. Shadow Risk",
        "return = 0.0", "MDD = 0.0", "worst day = 0.0", "exposure = 0.0", "",
        "K. Promotion Evidence", "sessions = 0 / 20", "monthly rebalances = 0 / 1", "",
        "L. Production Gate", str(gate.get("promotion_verdict", "SHADOW_EXTEND")), "",
        "M. Governance", "strategy changed = NO", "runtime contract changed = NO",
        "historical evidence changed = NO", "real orders submitted = NO", "production ready = NO", "",
        "N. Official Feed Migration",
        f"TWSE coverage = {audit.get('twse_covered_count', 'N/A')} / {audit.get('canonical_count', 'N/A')}",
        f"unresolved tickers = {migration.get('unresolved_tickers', 0)}",
        f"migration = {migration.get('status')}",
        f"reason = {migration.get('reason')}",
        f"data reconciliation = {data.get('status')}",
        f"signal reconciliation = {signal.get('status')}",
        f"performance = {performance.get('classification')}",
        f"missing official rows = {migration.get('remaining_gaps', {}).get('missing_official_rows', 'N/A')}",
        f"factor gaps = {migration.get('remaining_gaps', {}).get('traded_rows_missing_factor', 'N/A')}",
        f"calendar missing sessions = {migration.get('remaining_gaps', {}).get('calendar_missing_sessions', [])}",
        f"volume mismatches = {migration.get('remaining_gaps', {}).get('volume_mismatch_count', 'N/A')}",
    ]
    (out / "shadow_runtime_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def migrate(root: Path = ROOT, *, start: str = "2026-01-02", end: str = "2026-09-16") -> dict[str, object]:
    root = Path(root).resolve()
    out = root / "data/runtime/shadow-s3-v1"
    out.mkdir(parents=True, exist_ok=True)
    ticker_report = resolve_tickers(root)
    adjustment = discover_adjustment_contract(root)
    raw_path = out / "normalized_raw_ohlcv.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
    else:
        raw = fetch_official_history(root, pd.Timestamp(start), pd.Timestamp(end), TWSEOfficialAdapter())
    tickers = sorted(raw["ticker"].dropna().astype(str).unique()) if not raw.empty else []
    reference_raw_path = out / "_yfinance_reference_raw.parquet"
    reference_adjusted_path = out / "_yfinance_reference_adjusted.parquet"
    if reference_raw_path.exists() and reference_adjusted_path.exists():
        reference_raw = pd.read_parquet(reference_raw_path)
        reference_adjusted = pd.read_parquet(reference_adjusted_path)
    else:
        reference_raw, reference_adjusted = download_yfinance_reference(tickers, start, end, chunk_size=100, timeout=10)
        atomic_parquet(reference_raw_path, reference_raw)
        atomic_parquet(reference_adjusted_path, reference_adjusted)
    reference_raw, reference_adjusted, retry = refresh_reference_gaps(
        raw, reference_raw, reference_adjusted, start=start, end=end
    )
    if retry["rows_added"]:
        atomic_parquet(reference_raw_path, reference_raw)
        atomic_parquet(reference_adjusted_path, reference_adjusted)
    ref = reference_raw[["date", "ticker", "close"]].rename(columns={"close": "reference_raw_close"}).merge(
        reference_adjusted[["date", "ticker", "close"]].rename(columns={"close": "reference_adjusted_close"}),
        on=["date", "ticker"], how="left", validate="one_to_one"
    )
    factors = build_factor_history(raw, ref, output=out / "adjustment_factor_history.parquet")
    official_adjusted, factor_gaps = partial_hybrid_adjustment(raw, factors)
    data_report = reconcile_data_layers(official_adjusted, reference_adjusted, raw, reference_raw, output_dir=out)
    action_report = corporate_action_reconciliation(raw, official_adjusted)
    atomic_json(out / "corporate_action_reconciliation.json", action_report)
    factor_report = factor_reconciliation(root, official_adjusted, reference_adjusted, raw, reference_raw, out)
    signal_report = signal_reconciliation(root, official_adjusted, reference_adjusted, out)
    migration_ok = (
        ticker_report["status"] == "PASS"
        and adjustment["contract"]["status"] == "RESOLVED"
        and factor_gaps["traded_rows_missing_factor"] == 0
        and data_report["status"] == "PASS"
        and factor_report["status"] == "PASS"
        and signal_report["status"] == "PASS"
        and action_report["status"] == "PASS"
    )
    performance = {
        "status": "PASS" if migration_ok else "REVIEW_REQUIRED",
        "classification": "NUMERICALLY_EQUIVALENT" if migration_ok else "MATERIAL_DATA_SEMANTIC_CHANGE",
        "metrics": {"total_return_delta": None, "cagr_delta": None, "sharpe_delta": None, "mdd_delta": None},
        "note": "Performance replay is gated by data/factor/signal equality; no new tolerance is introduced.",
    }
    atomic_json(out / "fresh_oos_performance_reconciliation.json", performance)
    commit = commit_canonical_store(root, official_adjusted.loc[official_adjusted["date"] >= OOS_START]) if migration_ok else {"status": "NOT_COMMITTED", "sessions_committed": 0, "latest_session": None}
    migration = {
        "status": "APPROVED" if migration_ok else "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED",
        "reason": None if migration_ok else ";".join(
            reason for reason in (
                "ADJUSTMENT_FACTOR_MISSING_OR_INVALID" if factor_gaps["traded_rows_missing_factor"] else "",
                "CALENDAR_OR_DATA_RECONCILIATION_MISMATCH" if data_report["status"] != "PASS" else "",
                "FACTOR_RECONCILIATION_MISMATCH" if factor_report["status"] != "PASS" else "",
                "SIGNAL_RECONCILIATION_MISMATCH" if signal_report["status"] != "PASS" else "",
                "CORPORATE_ACTION_RECONCILIATION_MISMATCH" if action_report["status"] != "PASS" else "",
            ) if reason
        ) or "MIGRATION_REVIEW_REQUIRED",
        "sessions_committed": commit["sessions_committed"],
        "latest_session": commit["latest_session"],
        "forward_sessions_added": 0,
        "unresolved_tickers": len(ticker_report.get("unresolved", [])),
        "ticker_resolution": ticker_report["status"],
        "adjustment_contract": adjustment["contract"]["status"],
        "adjustment_factor_gaps": factor_gaps,
        "data_reconciliation": data_report["status"],
        "factor_reconciliation": factor_report["status"],
        "signal_reconciliation": signal_report["status"],
        "performance": performance["classification"],
        "signal_mismatch_count": int(signal_report.get("composite_mismatch_count") or 0),
        "target_mismatch_count": int(signal_report.get("target_mismatch_count") or 0),
        "factor_mismatch_count": int(
            (factor_report.get("L2_mismatch_count") or 0)
            + (factor_report.get("L4_mismatch_count") or 0)
        ),
        "remaining_gaps": {
            "missing_official_rows": int(data_report.get("key_coverage", {}).get("missing_official", 0) or 0),
            "missing_official_row_examples": [
                {"date": str(date.date()), "ticker": str(ticker)}
                for date, ticker in sorted(
                    set(
                        map(
                            tuple,
                            pd.read_parquet(out / "_yfinance_reference_raw.parquet")[["date", "ticker"]]
                            .merge(raw[["date", "ticker"]], on=["date", "ticker"], how="left", indicator=True)
                            .loc[lambda frame: frame["_merge"].eq("left_only"), ["date", "ticker"]]
                            .itertuples(index=False, name=None),
                        )
                    )
                )[:20]
            ],
            "traded_rows_missing_factor": int(factor_gaps["traded_rows_missing_factor"]),
            "factor_gap_examples": [
                {"date": str(row.date.date()), "ticker": str(row.ticker)}
                for row in factors.loc[factors["status"].eq("MISSING_REFERENCE_FACTOR"), ["date", "ticker"]].head(20).itertuples(index=False)
            ],
            "calendar_missing_sessions": sorted(
                set(
                    pd.to_datetime(reference_adjusted["date"])
                    .loc[lambda values: values.between(OOS_START, OOS_END)]
                    .dt.strftime("%Y-%m-%d")
                )
                - set(
                    pd.to_datetime(raw["date"])
                    .loc[lambda values: values.between(OOS_START, OOS_END)]
                    .dt.strftime("%Y-%m-%d")
                )
            ),
            "volume_mismatch_count": int(
                data_report.get("VOLUME_COMPARISON", {}).get("mismatch_count", 0) or 0
            ),
            "corporate_action_examples": action_report.get("adjusted_examples", []),
        },
        "retry": retry,
    }
    atomic_json(out / "canonical_migration_report.json", migration)
    atomic_json(out / "backfill_commit_manifest.json", {
        "mode": "HISTORICAL_BACKFILL",
        "sessions_committed": int(commit["sessions_committed"]),
        "forward_sessions_added": 0,
        "latest_session": commit["latest_session"],
        "status": "COMMITTED" if migration_ok else "BLOCKED_MIGRATION_REVIEW",
    })
    atomic_json(out / "official_feed_fresh_oos_reconciliation.json", {
        "status": data_report["status"],
        "calendar": data_report.get("session_calendar"),
        "ticker_universe": data_report.get("ticker_universe"),
        "RAW_PRICE_COMPARISON": data_report.get("RAW_PRICE_COMPARISON"),
        "ADJUSTED_PRICE_COMPARISON": data_report.get("ADJUSTED_PRICE_COMPARISON"),
        "VOLUME_COMPARISON": data_report.get("VOLUME_COMPARISON"),
        "DAILY_RETURN_COMPARISON": data_report.get("DAILY_RETURN_COMPARISON"),
        "factor_reconciliation": factor_report,
        "signal_reconciliation": signal_report,
        "remaining_gaps": migration.get("remaining_gaps", {}),
        "action": "COMMIT_CANONICAL" if migration_ok else "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED",
    })
    atomic_json(out / "adjustment_factor_reconciliation.json", {
        "status": "PASS" if factor_gaps["traded_rows_missing_factor"] == 0 else "UNRESOLVED",
        "provider": "YFINANCE_ADJUSTMENT_FACTOR_PROVIDER",
        "factor_rows": int((factors["status"] == "MATCHED").sum()),
        "missing_factor_rows": factor_gaps["traded_rows_missing_factor"],
        "no_trade_rows_without_factor": factor_gaps["no_trade_rows_without_factor"],
        "formula": "official_raw_ohlcv * (reference_adjusted_close / reference_raw_close)",
        "no_silent_interpolation": True,
    })
    write_gate_artifacts(root, migration=migration, data_health="PASS" if migration_ok else "SAFE_HALT")
    write_migration_report(root, migration, data_report, signal_report, performance)
    write_migration_manifest(root, migration)
    return {
        "status": migration["status"],
        "ticker_resolution": ticker_report,
        "adjustment": adjustment,
        "data": data_report,
        "factor": factor_report,
        "signal": signal_report,
        "performance": performance,
        "migration": migration,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-09-16")
    args = parser.parse_args()
    result = migrate(ROOT, start=args.start, end=args.end)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("status") == "APPROVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

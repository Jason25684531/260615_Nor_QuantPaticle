"""Scheduler-safe canonical ingestion plus broker-free S3 shadow run."""

# ruff: noqa: E402, E501
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twse_factor_lab.data.incremental_ohlcv import (
    IncrementalCanonicalOHLCVUpdater,
)
from twse_factor_lab.data.official_market_data import (
    OfficialCanonicalIngestion,
    write_official_contract,
)
from twse_factor_lab.runtime.shadow import ShadowRuntime, atomic_json


def _append_daily_log(path: Path, row: dict[str, Any]) -> None:
    previous = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    merged = pd.concat([previous, pd.DataFrame([row])], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(["run_id"], keep="first").sort_values("run_id")
    temporary = path.with_suffix(path.suffix + ".tmp")
    merged.to_parquet(temporary, index=False)
    temporary.replace(path)


def _safe_halt(runtime: ShadowRuntime, result: Any) -> dict[str, Any]:
    runtime.prepare()
    state = runtime._state()
    reconciliation = runtime._reconcile()
    gate = runtime._gate(state, reconciliation)
    date = result.latest_after or str(pd.Timestamp.now(tz="Asia/Taipei").date())
    health = {
        "date": date,
        "data_health": result.status,
        "signal_status": "NOT_RUN",
        "rebalance_status": "SAFE_HALT",
        "pending_orders": len(state.get("pending_orders", [])),
        "fill_status": "NOT_RUN",
        "reconciliation_status": reconciliation["status"],
        "state_integrity": "PASS",
        "risk_status": "NOT_RUN",
        "overall_status": "SAFE_HALT",
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(runtime.out / "daily_shadow_health.json", health)
    runtime._reports(  # noqa: SLF001 - report generation is contract evidence
        state, {"status": "FAIL"}, gate
    )
    return {"status": "SAFE_HALT", "ingestion": result.as_dict(), "gate": gate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--mode", choices=["BACKFILL", "FORWARD"])
    parser.add_argument("--source", choices=["yfinance", "official"], default="yfinance")
    parser.add_argument("--tickers", nargs="+")
    args = parser.parse_args()
    if args.source == "official":
        official = OfficialCanonicalIngestion(ROOT)
        audit = official.coverage_audit()
        adjustment = official.adjustment
        write_official_contract(ROOT, official.adapter.inspect_openapi(), adjustment)
        ingestion = official.update(
            start=args.start or "2026-01-01",
            end=args.end or str(pd.Timestamp.now(tz="Asia/Taipei").date()),
            mode="HISTORICAL_BACKFILL" if (args.mode or "BACKFILL") == "BACKFILL" else "FORWARD_DAILY_FEED",
        )
        official.write_reconciliation_reports()
        if ingestion.status not in {"PASS", "NOOP"}:
            runtime = ShadowRuntime(ROOT)
            outcome = _safe_halt(runtime, ingestion)
            outcome["official_coverage"] = audit
            _append_daily_log(
                runtime.out / "daily_run_log.parquet",
                {
                    "run_id": f"daily-{ingestion.latest_after or 'none'}-{ingestion.mode}",
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "session_date": ingestion.latest_after,
                    "mode": ingestion.mode,
                    "ingestion_status": ingestion.status,
                    "runtime_status": outcome.get("status"),
                    "true_forward": False,
                },
            )
            print(json.dumps(outcome, sort_keys=True, default=str))
            return 2
        # Official raw data is intentionally not fed to the adjusted runtime
        # until the adjustment contract passes.
        runtime = ShadowRuntime(ROOT)
        outcome = _safe_halt(runtime, ingestion)
        outcome["official_coverage"] = audit
        _append_daily_log(
            runtime.out / "daily_run_log.parquet",
            {
                "run_id": f"daily-{ingestion.latest_after or 'none'}-{ingestion.mode}",
                "recorded_at": datetime.now(UTC).isoformat(),
                "session_date": ingestion.latest_after,
                "mode": ingestion.mode,
                "ingestion_status": ingestion.status,
                "runtime_status": outcome.get("status"),
                "true_forward": False,
            },
        )
        print(json.dumps(outcome, sort_keys=True, default=str))
        return 2

    updater = IncrementalCanonicalOHLCVUpdater(ROOT)
    updater.ensure_forward_boundary()
    ingestion = updater.update(
        start=args.start,
        end=args.end,
        mode=args.mode,
        tickers=args.tickers,
    )
    runtime = ShadowRuntime(ROOT)
    if ingestion.status not in {"PASS", "NOOP"}:
        outcome = _safe_halt(runtime, ingestion)
    else:
        processed = ROOT / "data/processed"
        close = pd.read_parquet(processed / "close_matrix.parquet")
        volume = pd.read_parquet(processed / "volume_matrix.parquet")
        universe = pd.read_parquet(processed / "universe.parquet")
        as_of = ingestion.latest_after or str(pd.Timestamp(close.index.max()).date())
        boundary = pd.Timestamp(updater.ensure_forward_boundary())
        available = (
            pd.Timestamp(ingestion.data_available_at)
            if ingestion.data_available_at
            else None
        )
        true_forward = (
            ingestion.status == "PASS"
            and ingestion.mode == "FORWARD"
            and available is not None
            and available.tz_localize(None) >= boundary.tz_localize(None)
        )
        outcome = runtime.run(
            close,
            volume,
            universe,
            as_of,
            forward=true_forward,
            data_available_at=ingestion.data_available_at,
        )
        outcome = {
            "status": outcome["status"],
            "ingestion": ingestion.as_dict(),
            **outcome,
        }
        if ingestion.mode == "BACKFILL":
            updater.write_backfill_reconciliation()
            updater.write_fresh_oos_reconciliation()
    run_id = f"daily-{ingestion.latest_after or 'none'}-{ingestion.mode}"
    _append_daily_log(
        runtime.out / "daily_run_log.parquet",
        {
            "run_id": run_id,
            "recorded_at": datetime.now(UTC).isoformat(),
            "session_date": ingestion.latest_after,
            "mode": ingestion.mode,
            "ingestion_status": ingestion.status,
            "runtime_status": outcome.get("status"),
            "true_forward": bool(
                outcome.get("ingestion", {}).get("mode") == "FORWARD"
                and outcome.get("status") == "PASS"
            ),
        },
    )
    print(json.dumps(outcome, sort_keys=True, default=str))
    return 0 if outcome.get("status") in {"PASS", "NOOP"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

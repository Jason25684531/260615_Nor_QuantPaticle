"""Incrementally update the canonical processed OHLCV store."""

# ruff: noqa: E402, E501
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twse_factor_lab.data.incremental_ohlcv import (
    IncrementalCanonicalOHLCVUpdater,
)
from twse_factor_lab.data.official_market_data import OfficialCanonicalIngestion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--mode", choices=["BACKFILL", "FORWARD"])
    parser.add_argument("--source", choices=["yfinance", "official"], default="yfinance")
    parser.add_argument("--tickers", nargs="+", help="Optional canonical ticker subset")
    args = parser.parse_args()
    if args.source == "official":
        updater = OfficialCanonicalIngestion(ROOT)
        audit = updater.coverage_audit()
        adjustment = updater.adjustment
        result = updater.update(
            start=args.start or "2026-01-01",
            end=args.end or str(pd.Timestamp.now(tz="Asia/Taipei").date()),
            mode="HISTORICAL_BACKFILL" if (args.mode or "BACKFILL") == "BACKFILL" else "FORWARD_DAILY_FEED",
        )
        updater.write_reconciliation_reports()
        inspection = updater.adapter.inspect_openapi()
        from twse_factor_lab.data.official_market_data import write_official_contract

        write_official_contract(ROOT, inspection, adjustment)
        updater_out = result.as_dict() | {"coverage": audit, "adjustment": adjustment}
        print(json.dumps(updater_out, sort_keys=True, default=str))
        return 0 if result.status in {"PASS", "NOOP"} else 2

    updater = IncrementalCanonicalOHLCVUpdater(ROOT)
    updater.ensure_forward_boundary()
    result = updater.update(
        start=args.start,
        end=args.end,
        mode=args.mode,
        tickers=args.tickers,
    )
    health_path = updater.out / "data_health_log.csv"
    health_row = {
        "run_at": result.data_available_at,
        "trading_date": result.latest_after,
        "latest_data_date": result.latest_after,
        "status": "PASS" if result.status in {"PASS", "NOOP"} else "FAIL",
        "data_ready": result.status in {"PASS", "NOOP"},
        "issues": ";".join(result.issues),
    }
    pd.DataFrame([health_row]).to_csv(
        health_path,
        index=False,
        mode="a",
        header=not health_path.exists(),
        lineterminator="\n",
    )
    if result.mode == "BACKFILL":
        updater.write_backfill_reconciliation()
        updater.write_fresh_oos_reconciliation()
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0 if result.status in {"PASS", "NOOP"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

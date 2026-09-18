"""Safe scheduler-ready Fundamental production recommendation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twse_factor_lab.application.commands.fundamental_final import run_final_validation
from twse_factor_lab.application.support import repository_root
from twse_factor_lab.production.final_runtime import (
    AtomicRecommendationStore,
    CanonicalFundamentalRuntimeProvider,
    run_daily_fundamental,
)
from twse_factor_lab.production.fundamental import build_current_promotion_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--write-recommendations", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = repository_root(args.root)
    if not args.as_of_date:
        result = run_final_validation(root)
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    try:
        provider = CanonicalFundamentalRuntimeProvider.from_repository(root)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": str(exc),
                    "broker_order_submission": "DISABLED",
                }
            )
        )
        return 2
    import pandas as pd

    calendar_path = root / "data/processed/ohlcv.parquet"
    if calendar_path.exists():
        calendar = pd.read_parquet(calendar_path, columns=["date"])
        sessions = sorted(
            pd.to_datetime(calendar["date"]).dt.strftime("%Y-%m-%d").unique().tolist()
        )
    else:
        sessions = []
    result = run_daily_fundamental(
        provider=provider,
        as_of_date=args.as_of_date,
        sessions=sessions,
        evidence=build_current_promotion_evidence(),
        write_recommendations=args.write_recommendations
        and not (args.dry_run or args.validate_only),
        store=AtomicRecommendationStore(root / "data/production/recommendations"),
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["status"] in {"PASS", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

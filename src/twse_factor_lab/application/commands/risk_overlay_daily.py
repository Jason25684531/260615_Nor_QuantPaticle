"""Generate the historical daily/T+1 MA60 breadth overlay comparison."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from twse_factor_lab.analysis.risk_overlay_daily import (
    DailyRiskOverlayError,
    evaluate_daily_risk_overlay,
)
from twse_factor_lab.application.support import repository_root
from twse_factor_lab.reporting.performance_adapter import FrozenStrategyChanged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = evaluate_daily_risk_overlay(repository_root(args.root))
    except (DailyRiskOverlayError, FrozenStrategyChanged, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result["classification"],
                "benchmark": result["benchmark"]["status"],
                "fresh_oos": result["fresh_oos_status"],
                "production_ready": result["production_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

"""Generate the frozen Fundamental strategy performance tear sheet."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from twse_factor_lab.application.support import repository_root
from twse_factor_lab.reporting.performance_adapter import FrozenStrategyChanged
from twse_factor_lab.reporting.performance_tearsheet import (
    generate_performance_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--period",
        choices=("all", "backtest", "historical-oos", "fresh-oos"),
        default="all",
    )
    parser.add_argument(
        "--format", choices=("html", "pdf", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = generate_performance_report(
            repository_root(args.root),
            period=args.period,
            output_format=args.format,
            force=args.force,
        )
    except FrozenStrategyChanged:
        print(json.dumps({"status": "BLOCKED", "reason": "FROZEN_STRATEGY_CHANGED"}))
        return 2
    except (FileExistsError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": report.summary["report_status"],
                "output": str(report.output_dir),
                "performance_evidence": report.summary["performance_evidence"],
                "fresh_oos_status": report.summary["fresh_oos_status"],
                "production_ready": report.summary["promotion_gate"][
                    "production_ready"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = ["main"]

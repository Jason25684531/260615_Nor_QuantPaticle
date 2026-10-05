"""Generate frozen-strategy canonical historical evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from twse_factor_lab.analysis.historical_evidence import (
    HistoricalEvidenceError,
    generate_historical_evidence,
)
from twse_factor_lab.application.support import repository_root
from twse_factor_lab.reporting.performance_tearsheet import (
    generate_performance_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--skip-performance-report", action="store_true")
    args = parser.parse_args(argv)
    root = repository_root(args.root)
    try:
        evidence = generate_historical_evidence(root)
        if not args.skip_performance_report:
            generate_performance_report(root, force=True)
    except HistoricalEvidenceError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "historical_evidence": evidence["historical_evidence"][
                    "classification"
                ],
                "historical_oos": evidence["historical_oos"]["status"],
                "fresh_oos": evidence["fresh_oos"]["fresh_oos_status"],
                "production_ready": evidence["production_gate"][
                    "production_ready"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

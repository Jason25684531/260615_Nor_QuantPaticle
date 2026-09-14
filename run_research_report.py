"""CLI for the read-only research reporting layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twse_factor_lab.reporting.runner import generate_research_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-id", required=True)
    parser.add_argument("--root", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument(
        "--output-mode", choices=("auto", "research-cycle", "external"), default="auto"
    )
    args = parser.parse_args()
    result = generate_research_report(args.root, args.research_id, args.output_mode)
    print(
        json.dumps(
            {
                "research_id": result.research_id,
                "output_mode": result.output_mode,
                "output_dir": str(result.output_dir),
                "figures": [
                    {"figure_id": item.figure_id, "status": item.status}
                    for item in result.figures
                ],
                "platform_verdict": result.model.acceptance.data.get(
                    "research_platform_verdict"
                ),
                "strategy_verdict": result.model.acceptance.data.get(
                    "strategy_verdict"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

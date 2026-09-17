"""CLI for the read-only research reporting layer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from twse_factor_lab.reporting.runner import generate_research_report


def _repository_root() -> Path:
    """Find the checkout root without depending on the current directory."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-id", required=True)
    parser.add_argument("--root", default=_repository_root(), type=Path)
    parser.add_argument(
        "--output-mode", choices=("auto", "research-cycle", "external"), default="auto"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


__all__ = ["build_parser", "main"]

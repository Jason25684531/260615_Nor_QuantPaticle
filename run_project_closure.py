"""Generate the final S3 production-readiness closure evidence."""

# ruff: noqa: E501, I001

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twse_factor_lab.project_closure import build_closure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true", help="run pytest, ruff, OpenSpec, and RC1")
    args = parser.parse_args()
    result = build_closure(args.root, verify=args.verify)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["project_closed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

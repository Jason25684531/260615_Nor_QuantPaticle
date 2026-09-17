# ruff: noqa: E402, I001

"""Run the frozen S3 Fresh OOS validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twse_factor_lab.acceptance.fresh_oos_validation import run_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="refresh adjusted OHLCV via canonical yfinance client",
    )
    result = run_validation(ROOT, download=parser.parse_args().download)
    print(result["output"])
    print(result["acceptance"]["fresh_oos_status"])

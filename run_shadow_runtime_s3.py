"""Run frozen S3 shadow execution.  This program cannot send broker orders."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twse_factor_lab.runtime.shadow import ShadowRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of")
    parser.add_argument("--historical-replay", action="store_true")
    parser.add_argument("--failure-injection-report", action="store_true")
    args = parser.parse_args()
    processed = ROOT / "data/processed"
    runtime = ShadowRuntime(ROOT)
    if args.failure_injection_report:
        print(runtime.failure_injection_report())
        return
    close = pd.read_parquet(processed / "close_matrix.parquet")
    volume = pd.read_parquet(processed / "volume_matrix.parquet")
    universe = pd.read_parquet(processed / "universe.parquet")
    as_of = args.as_of or str(pd.Timestamp(close.index.max()).date())
    print(
        runtime.run(close, volume, universe, as_of, forward=not args.historical_replay)
    )


if __name__ == "__main__":
    main()

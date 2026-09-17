# ruff: noqa: E501

"""Run the immutable S3 Custom/Vectorbt/Backtrader parity audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twse_factor_lab.acceptance.engine_parity_audit import run_audit  # noqa: E402

CHANGE = "audit-engine-parity-and-numerical-tolerance-v1"
OPENSPEC = "openspec.cmd" if sys.platform == "win32" else "openspec"


def _check(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return "PASS" if completed.returncode == 0 else f"FAIL:{' '.join(command)}"


def _baseline() -> dict[str, str]:
    return {
        "pytest": _check([sys.executable, "-m", "pytest"]),
        "ruff": _check([sys.executable, "-m", "ruff", "check", "."]),
        "openspec_change": _check([OPENSPEC, "validate", CHANGE, "--strict"]),
        "openspec_all": _check([OPENSPEC, "validate", "--all", "--strict"]),
        "rc1": _check([sys.executable, "run_rc1_offline_e2e.py"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="run hard gates before replay")
    args = parser.parse_args()
    baseline = _baseline() if args.verify else {}
    result = run_audit(ROOT, baseline=baseline)
    print(result["output"])
    print(result["classification"], result["recommendation"])
    if args.verify and baseline.get("openspec_all", "").startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

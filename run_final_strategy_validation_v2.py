# ruff: noqa: E501

"""Run the frozen S3 Final Strategy Validation v2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twse_factor_lab.acceptance.final_validation_v2 import run_validation  # noqa: E402

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
        "openspec": _check(
            [
                OPENSPEC,
                "validate",
                "add-final-strategy-validation-v2-parity-contract-v1",
                "--strict",
            ]
        ),
        "openspec_all": _check([OPENSPEC, "validate", "--all", "--strict"]),
        "rc1": _check([sys.executable, "run_rc1_offline_e2e.py"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="run hard gates first")
    args = parser.parse_args()
    result = run_validation(ROOT, baseline=_baseline() if args.verify else None)
    print(result["output"])
    print(result["acceptance"]["final_verdict"])
    print(result["acceptance"]["acceptance_label"])


if __name__ == "__main__":
    main()

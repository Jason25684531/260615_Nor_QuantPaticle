from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class FrozenInputError(ValueError):
    pass


def load_inputs(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / "strategy_freeze_manifest.json").read_text())
    d4 = json.loads((root / "d4_acceptance_handoff.json").read_text())
    if manifest.get("git_commit") != d4.get("strategy_freeze_id"):
        raise FrozenInputError("freeze id does not match D4 handoff")
    if (manifest["date_range"]["start"], manifest["date_range"]["end"]) != (
        d4["is_range"][0],
        d4["oos_range"][1],
    ):
        raise FrozenInputError("manifest date range does not cover D4 range")
    processed = root / "data" / "processed"
    hashes = {}
    aliases = {"engine_comparison": "backtest_engine_comparison"}
    for name, expected in manifest["artifact_hashes"].items():
        path = processed / f"{aliases.get(name, name)}.parquet"
        actual = (
            hashlib.sha256(path.read_bytes()).hexdigest().upper()
            if path.exists()
            else ""
        )
        if actual != str(expected).upper():
            raise FrozenInputError(f"frozen artifact hash mismatch: {name}")
        hashes[name] = actual
    return {"manifest": manifest, "d4": d4, "artifact_hashes": hashes}


def validate_oos(oos_start: str, oos_end: str, d4: dict[str, Any]) -> tuple[str, str]:
    expected = tuple(d4["oos_range"])
    if (oos_start, oos_end) != expected:
        raise FrozenInputError("OOS range must equal D4 handoff")
    return expected

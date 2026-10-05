from __future__ import annotations

import json
from pathlib import Path


def test_benchmark_is_explicitly_unavailable_and_not_fabricated() -> None:
    root = Path("data/research/fundamental-production-final-v1/historical-evidence")
    manifest = json.loads((root / "canonical_backtest_manifest.json").read_text())
    validation = json.loads((root / "historical_strategy_validation.json").read_text())
    assert manifest["benchmark_status"] == "UNAVAILABLE"
    assert validation["benchmark"]["status"] == "UNAVAILABLE"
    assert validation["strategy_evidence"].get("excess_return") is None


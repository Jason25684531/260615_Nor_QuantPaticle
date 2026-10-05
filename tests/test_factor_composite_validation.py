from __future__ import annotations

import json
from pathlib import Path


def test_composite_validation_is_the_frozen_equal_weight_signal() -> None:
    root = Path("data/research/fundamental-production-final-v1/historical-evidence")
    payload = json.loads((root / "factor_composite_validation.json").read_text())
    assert payload["factor"] == "COMPOSITE"
    assert payload["primary_horizon"] == 60
    assert payload["ic_observations"] >= 12
    comparison = (root / "factor_comparison.csv").read_text(encoding="utf-8")
    assert "COMPOSITE" in comparison


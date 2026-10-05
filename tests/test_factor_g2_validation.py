from __future__ import annotations

import json
from pathlib import Path


def test_g2_validation_contains_declared_statistics() -> None:
    path = Path(
        "data/research/fundamental-production-final-v1/historical-evidence/"
        "factor_g2_validation.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["factor"] == "G2"
    assert payload["ic_observations"] > 0
    assert "ic_positive_ratio" in payload
    assert "top_bottom_spread" in payload
    assert "annual_ic" in payload

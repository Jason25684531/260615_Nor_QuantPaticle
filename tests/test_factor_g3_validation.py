from __future__ import annotations

import json
from pathlib import Path


def test_g3_validation_contains_declared_statistics() -> None:
    path = Path(
        "data/research/fundamental-production-final-v1/historical-evidence/"
        "factor_g3_validation.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["factor"] == "G3"
    assert payload["ic_observations"] > 0
    assert payload["bootstrap_ic_ci"]["status"] == "AVAILABLE"

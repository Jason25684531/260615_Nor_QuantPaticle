from __future__ import annotations

import json
from pathlib import Path

from twse_factor_lab.production.fundamental import (
    STRATEGY_FINGERPRINT,
    STRATEGY_ID,
)


def test_canonical_backtest_manifest_binds_frozen_identity() -> None:
    path = Path(
        "data/research/fundamental-production-final-v1/historical-evidence/canonical_backtest_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["strategy_id"] == STRATEGY_ID
    assert manifest["strategy_fingerprint"] == STRATEGY_FINGERPRINT
    assert manifest["pit_audit_status"] == "PASS"
    assert manifest["historical_label"] == "HISTORICAL_ROBUSTNESS"


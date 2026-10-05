from __future__ import annotations

from pathlib import Path

from twse_factor_lab.analysis.historical_evidence import (
    historical_inventory,
    load_policy,
)


def test_inventory_selects_longest_legal_window() -> None:
    root = Path(__file__).parents[1]
    inventory = historical_inventory(root, load_policy(root))
    assert inventory["usable_start"] == "2021-01-04"
    assert inventory["usable_end"] == "2025-12-31"
    assert inventory["trading_sessions"] > 750
    assert inventory["pit_available"] is True
    assert inventory["benchmark_available"] is False


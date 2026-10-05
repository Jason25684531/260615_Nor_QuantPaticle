from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from twse_factor_lab.analysis.canonical_benchmark import _returns

ROOT = Path("data/research/fundamental-production-final-v1")
BENCHMARK = ROOT / "benchmark"


def test_benchmark_identity_and_window() -> None:
    spec = json.loads((BENCHMARK / "canonical_benchmark_spec.json").read_text())
    frame = pd.read_parquet(BENCHMARK / "canonical_benchmark_returns.parquet")
    assert spec["benchmark_id"] == "twii_taiex_broad_market_price_index_v1"
    assert spec["source_series_id"] == "^TWII"
    assert spec["start_date"] == "2021-01-04"
    assert spec["end_date"] == "2025-12-31"
    assert frame["date"].is_unique
    assert len(frame) == 1212


def test_benchmark_policy_rejects_proxy_and_synthetic_sources() -> None:
    policy = json.loads((BENCHMARK / "canonical_benchmark_policy.json").read_text())
    assert policy["etf_proxy_policy"] == "PROHIBITED"
    assert policy["synthetic_survivor_benchmark_policy"] == "PROHIBITED"
    assert policy["selected_source"]["source_series_id"] == "^TWII"


def test_benchmark_return_convention_is_explicit() -> None:
    audit = json.loads(
        (BENCHMARK / "benchmark_return_convention_audit.json").read_text()
    )
    assert audit["strategy_return_type"] == "adjusted_price_return"
    assert audit["benchmark_return_type"] == "index_price_return"
    assert audit["compatibility_status"] == "DEFINITION_DIFFERENCE"


def test_benchmark_no_forward_fill() -> None:
    calendar = pd.DatetimeIndex(pd.to_datetime(["2021-01-04", "2021-01-05"]))
    source = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2021-01-04"]), "benchmark_level": [100.0]}
    )
    with pytest.raises(ValueError, match="INSUFFICIENT_BENCHMARK_COVERAGE"):
        _returns(source, calendar)


def test_governance_and_relative_metrics_are_preserved() -> None:
    validation = json.loads((BENCHMARK / "benchmark_validation.json").read_text())
    summary = json.loads((ROOT / "performance/performance_summary.json").read_text())
    gate = json.loads((ROOT / "production_final_gate.json").read_text())
    assert validation["status"] == "PASS"
    assert validation["governance"]["historical_evidence"] == "HISTORICALLY_SUPPORTIVE"
    assert validation["governance"]["fresh_oos"] == "INSUFFICIENT_DATA"
    assert validation["governance"]["production_gate"] == "BLOCKED"
    assert (
        summary["periods"]["backtest"]["metrics"]["benchmark_return"]["status"]
        == "AVAILABLE"
    )
    assert gate["production_ready"] is False


def test_benchmark_outputs_are_idempotent_identity_bound() -> None:
    first = json.loads((BENCHMARK / "canonical_benchmark_spec.json").read_text())
    second = json.loads((BENCHMARK / "canonical_benchmark_manifest.json").read_text())
    assert first["benchmark_fingerprint"] == second["benchmark_fingerprint"]
    assert second["observations"] == 1212

"""Focused A1-A24 migration safety checks."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from twse_factor_lab.data.official_migration import (
    apply_hybrid_adjustment,
    build_factor_history,
    corporate_action_reconciliation,
    discover_adjustment_contract,
    reconcile_data_layers,
)


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "ticker": ["1101", "1101"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.0, 11.0],
            "volume": [100.0, 110.0],
        }
    )


def test_a3_a4_contract_discovery_and_separation(tmp_path):
    report = discover_adjustment_contract(tmp_path)
    assert report["discovery"]["status"] == "RESOLVED"
    assert report["contract"]["raw_layer"] != report["contract"]["adjusted_layer"]
    assert report["contract"]["candidate_fingerprint"].startswith("36b6fd")


def test_a5_a7_factor_history_is_deterministic_and_unique(tmp_path):
    raw = _raw()
    reference = raw[["date", "ticker", "close"]].rename(
        columns={"close": "reference_raw_close"}
    )
    reference["reference_adjusted_close"] = reference["reference_raw_close"] * 2
    first = build_factor_history(raw, reference)
    second = build_factor_history(raw, reference)
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["date", "ticker"]).any()
    assert first.adjustment_factor.eq(2).all()


def test_a6_hybrid_formula_and_missing_factor_fail_closed():
    raw = _raw()
    factors = pd.DataFrame(
        {"date": raw.date, "ticker": raw.ticker, "adjustment_factor": [2.0, 2.0]}
    )
    adjusted = apply_hybrid_adjustment(raw, factors)
    assert adjusted.close.tolist() == [20.0, 22.0]
    with pytest.raises(ValueError, match="ADJUSTMENT_FACTOR_MISSING"):
        apply_hybrid_adjustment(raw, factors.iloc[:1])


def test_a8_a11_exact_data_reconciliation(tmp_path):
    raw = _raw()
    report = reconcile_data_layers(raw, raw, raw, raw, output_dir=tmp_path)
    assert report["status"] == "PASS"
    assert report["session_calendar"]["status"] == "PASS"
    assert report["ADJUSTED_PRICE_COMPARISON"]["close"]["mismatch_count"] == 0
    assert (tmp_path / "fresh_oos_data_reconciliation.json").exists()


def test_a12_a16_semantic_drift_is_not_hidden(tmp_path):
    old = _raw()
    new = old.copy()
    new.loc[1, "close"] = 99.0
    report = reconcile_data_layers(new, old, new, old, output_dir=tmp_path)
    assert report["status"] == "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED"
    assert report["ADJUSTED_PRICE_COMPARISON"]["close"]["mismatch_count"] == 1


def test_a17_corporate_action_report_is_explicit():
    raw = _raw()
    adjusted = raw.copy()
    adjusted.loc[1, "close"] = 5.5
    report = corporate_action_reconciliation(raw, adjusted)
    assert report["formula"]
    assert report["volume_adjusted"] is False


def test_a18_a21_failed_migration_does_not_commit(tmp_path):
    report = reconcile_data_layers(
        _raw(), _raw().assign(close=[10.0, 12.0]), _raw(), _raw(), output_dir=tmp_path
    )
    assert report["status"] == "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED"
    assert not (tmp_path / "data/processed/ohlcv.parquet").exists()


def test_a22_a24_contract_artifact_is_json(tmp_path):
    result = discover_adjustment_contract(tmp_path)
    path = tmp_path / "data/runtime/shadow-s3-v1/adjustment_contract.json"
    assert json.loads(path.read_text())["status"] == result["contract"]["status"]

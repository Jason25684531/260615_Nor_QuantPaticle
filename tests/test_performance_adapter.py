from __future__ import annotations

import json

import pandas as pd
import pytest

from twse_factor_lab.reporting.performance_adapter import (
    PerformanceData,
    PerformanceDataError,
    _validate_canonical_manifest,
    load_frozen_identity,
    load_repository_performance_data,
)


def test_returns_are_utc_sorted_unique_and_not_filled():
    values = pd.Series([0.01, 0.02], index=["2024-01-02", "2024-01-04"])
    data = PerformanceData.from_frames("backtest", values)
    assert str(data.returns.index.tz) == "UTC"
    assert data.returns.index.day.tolist() == [2, 4]
    assert len(data.returns) == 2


def test_pyfolio_views_preserve_true_inputs(sample_performance_data):
    returns, positions, transactions = sample_performance_data.to_pyfolio_inputs()
    assert returns.index.equals(positions.index)
    assert "cash" in positions
    assert list(transactions.columns) == ["symbol", "amount", "price"]


@pytest.mark.parametrize(
    "values, message",
    [
        (
            pd.Series([0.01, 0.02], index=["2024-01-02", "2024-01-02"]),
            "duplicate",
        ),
        (
            pd.Series(
                [0.01, float("nan")],
                index=pd.date_range("2024-01-02", periods=2),
            ),
            "NaN",
        ),
    ],
)
def test_invalid_returns_are_rejected(values: pd.Series, message: str):
    with pytest.raises(PerformanceDataError, match=message):
        PerformanceData.from_frames("BACKTEST", values)


def test_repository_does_not_reuse_unbound_old_artifacts(performance_root):
    result = load_repository_performance_data(performance_root)
    assert result["BACKTEST"].status == "INSUFFICIENT_DATA"
    assert result["HISTORICAL_OOS"].status == "INSUFFICIENT_DATA"
    assert result["FRESH_OOS"].reason == (
        "NO_LEGAL_UNTOUCHED_WINDOW_WITH_REQUIRED_HISTORY"
    )


def test_canonical_manifest_fingerprint_mismatch_is_rejected(tmp_path):
    identity = load_frozen_identity(".")
    manifest = tmp_path / "canonical_backtest_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy_id": identity.strategy_id,
                "strategy_fingerprint": "changed",
                "strategy_config_hash": identity.digest,
                "pit_audit_status": "PASS",
                "artifact_sha256": {"returns.parquet": "unused"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PerformanceDataError, match="FROZEN_STRATEGY_CHANGED"):
        _validate_canonical_manifest(
            tmp_path, {"canonical_manifest": manifest.name}, identity
        )

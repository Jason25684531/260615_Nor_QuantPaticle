from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from twse_factor_lab.backtest.robustness import (
    ManifestValidationError,
    align_held_exposures,
    chronological_split,
    classify_robustness,
    hac_lag,
    load_frozen_manifest,
    verify_artifact_hashes,
)


def test_manifest_missing_required_field_fails_fast(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="selected_factors"):
        load_frozen_manifest(path)


def test_manifest_validates_each_required_field(tmp_path):
    manifest = json.loads(Path("strategy_freeze_manifest.json").read_text())
    for field in [
        "selected_factors",
        "factor_weights",
        "top_n",
        "buffer",
        "rebalance",
        "breadth",
        "strategy",
        "cost_model",
        "date_range",
        "git_commit",
        "artifact_hashes",
    ]:
        broken = manifest.copy()
        broken.pop(field)
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ManifestValidationError, match=field):
            load_frozen_manifest(path)


def test_hash_verification_returns_pass_and_warn(tmp_path):
    artifact = tmp_path / "x.parquet"
    artifact.write_bytes(b"fixture")
    digest = sha256(b"fixture").hexdigest()
    manifest = {"artifact_hashes": {"x": digest}}
    assert verify_artifact_hashes(manifest, tmp_path) == {"x": "PASS"}
    artifact.write_bytes(b"changed")
    assert verify_artifact_hashes(manifest, tmp_path) == {"x": "WARN"}


def test_chronological_split_is_deterministic_and_ordered():
    dates = pd.bdate_range("2024-01-01", periods=10)
    first = chronological_split(dates, "2024-01-01", "2024-01-31")
    second = chronological_split(dates, "2024-01-01", "2024-01-31")
    assert first.is_dates.equals(second.is_dates)
    assert first.is_dates[-1] < first.oos_dates[0]
    assert len(first.is_dates) == 7


def test_negative_mdd_uses_magnitude_not_signed_comparison():
    baseline = {"sharpe": 1.0, "max_drawdown": -0.10}
    stable = {"nobs": 252, "sharpe": 0.8, "max_drawdown": -0.15}
    fragile = {"nobs": 252, "sharpe": 0.8, "max_drawdown": -0.16}
    assert classify_robustness(stable, baseline, 252) == "STABLE"
    assert classify_robustness(fragile, baseline, 252) == "DEGRADED"


def test_alignment_excludes_return_realized_before_execution():
    dates = pd.bdate_range("2024-01-01", periods=3)
    signals = pd.DataFrame({"factor_a": [1.0, 2.0]}, index=dates[:2])
    execution = pd.Series([dates[1], dates[2]], index=dates[:2])
    returns = pd.Series([0.99, 0.01, 0.02], index=dates)
    aligned = align_held_exposures(signals, execution, returns)
    assert aligned["return"].tolist() == [0.01, 0.02]
    assert 0.99 not in aligned["return"].tolist()


def test_hac_automatic_rule_is_recordable():
    assert hac_lag(100, "automatic") == 4
    assert hac_lag(100, 3) == 3

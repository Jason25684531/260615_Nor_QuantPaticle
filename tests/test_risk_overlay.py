import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import twse_factor_lab.analysis.risk_overlay as risk_overlay
from twse_factor_lab.analysis.risk_overlay import (
    BREADTH_THRESHOLD,
    EXPOSURE_HIGH,
    EXPOSURE_LOW,
    apply_rebalance_exposure,
    calculate_ma60_breadth,
    classify_risk_overlay,
    discover_canonical_benchmark,
    evaluate_risk_overlay,
    exposure_for_breadth,
)


def test_fixed_step_rule_includes_boundary_as_risk_off():
    assert exposure_for_breadth(0.41) == EXPOSURE_HIGH
    assert exposure_for_breadth(BREADTH_THRESHOLD) == EXPOSURE_LOW
    assert exposure_for_breadth(None) == EXPOSURE_LOW


def test_ma60_breadth_does_not_use_future_prices():
    dates = pd.date_range("2021-01-01", periods=65, freq="D")
    close = pd.DataFrame({"A": np.full(65, 10.0), "B": np.full(65, 10.0)}, index=dates)
    universe = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "ticker": ["A", "B"] * len(dates),
            "is_eligible": True,
        }
    )
    original = calculate_ma60_breadth(close, universe)
    changed = close.copy()
    changed.loc[dates[-1], "A"] = 1000.0
    revised = calculate_ma60_breadth(changed, universe)
    assert original.loc[original.trade_date == dates[-2], "breadth"].iloc[
        0
    ] == pytest.approx(revised.loc[revised.trade_date == dates[-2], "breadth"].iloc[0])


def test_empty_usable_universe_is_not_fabricated():
    dates = pd.date_range("2021-01-01", periods=2, freq="D")
    close = pd.DataFrame({"A": [10.0, 10.0]}, index=dates)
    universe = pd.DataFrame(
        {"date": dates, "ticker": ["A", "A"], "is_eligible": [True, True]}
    )
    result = calculate_ma60_breadth(close, universe)
    assert result["breadth"].isna().all()


def test_exposure_changes_only_at_rebalance_execution_and_carries():
    sessions = pd.date_range("2024-01-01", periods=6, freq="D")
    events = pd.DataFrame(
        {
            "signal_date": [sessions[0], sessions[0], sessions[3], sessions[3]],
            "execution_date": [sessions[1], sessions[1], sessions[4], sessions[4]],
            "ticker": ["A", "B", "A", "B"],
            "target_weight": [0.6, 0.4, 0.7, 0.3],
        }
    )
    breadth = pd.DataFrame(
        {"trade_date": sessions, "breadth": [0.6, 0.2, 0.2, 0.2, 0.8, 0.8]}
    )
    scaled, state, _ = apply_rebalance_exposure(events, breadth, sessions)
    assert scaled.loc[scaled.signal_date == sessions[0], "exposure"].iloc[0] == 1.0
    assert scaled.loc[scaled.signal_date == sessions[3], "exposure"].iloc[0] == 0.5
    assert state.loc[sessions[2]] == 1.0
    assert state.loc[sessions[4]] == 0.5
    assert scaled.loc[
        scaled.signal_date == sessions[3], "target_weight"
    ].sum() == pytest.approx(0.5)


def test_selection_and_relative_weights_are_only_scaled():
    sessions = pd.date_range("2024-01-01", periods=2, freq="D")
    events = pd.DataFrame(
        {
            "signal_date": [sessions[0], sessions[0]],
            "execution_date": [sessions[1], sessions[1]],
            "ticker": ["A", "B"],
            "target_weight": [0.6, 0.4],
        }
    )
    breadth = pd.DataFrame({"trade_date": sessions, "breadth": [0.2, 0.2]})
    scaled, _, _ = apply_rebalance_exposure(events, breadth, sessions)
    assert scaled.ticker.tolist() == ["A", "B"]
    assert scaled.target_weight.iloc[0] / scaled.target_weight.iloc[1] == pytest.approx(
        1.5
    )


def test_classification_policy_is_fixed():
    rows = [
        ("cagr", 0.10, 0.08),
        ("sharpe", 0.50, 0.50),
        ("max_drawdown", -0.40, -0.33),
        ("annualized_volatility", 0.30, 0.27),
    ]
    assert classify_risk_overlay(
        pd.DataFrame(rows, columns=["metric", "base", "overlay"])
    ) == ("RISK_OVERLAY_SUPPORTIVE")


def test_repository_benchmark_is_explicitly_unavailable():
    result = discover_canonical_benchmark(".")
    assert result["status"] == "UNAVAILABLE"


def test_existing_canonical_benchmark_is_reused(tmp_path):
    manifest_path = tmp_path / risk_overlay.BASE_MANIFEST
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"benchmark_status": "AVAILABLE", "benchmark_id": "TEST"})
    )
    benchmark_path = tmp_path / risk_overlay.BENCHMARK_PATH
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "daily_return": [0.0, 0.01],
        }
    ).to_parquet(benchmark_path, index=False)
    result = discover_canonical_benchmark(tmp_path)
    assert result["status"] == "AVAILABLE"
    assert result["benchmark_id"] == "TEST"


def test_historical_evaluation_is_additive_and_fresh_oos_safe():
    result = evaluate_risk_overlay(".")
    assert result["base_strategy_fingerprint"] == (
        "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f"
    )
    assert result["strategy_modified"] is False
    assert result["fresh_oos_status"] == "UNCHANGED / INSUFFICIENT_DATA"
    assert result["production_ready"] is False


def test_frozen_identity_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(
        risk_overlay,
        "load_frozen_identity",
        lambda root: SimpleNamespace(
            strategy_id=risk_overlay.STRATEGY_ID,
            strategy_fingerprint="changed",
        ),
    )
    with pytest.raises(
        risk_overlay.FrozenStrategyChanged, match="FROZEN_STRATEGY_CHANGED"
    ):
        risk_overlay.evaluate_risk_overlay(".")


def test_evaluation_outputs_are_idempotent_and_complete():
    first = evaluate_risk_overlay(".")
    second = evaluate_risk_overlay(".")
    assert first["comparison"] == second["comparison"]
    output = risk_overlay.Path(
        "data/research/fundamental-production-final-v1/risk-overlay"
    )
    assert {path.name for path in output.glob("*.json")} >= {
        "breadth_overlay_summary.json",
        "risk_overlay_validation.json",
    }
    assert len(list((output / "charts").glob("*.png"))) == 4

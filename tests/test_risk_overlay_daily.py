import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import twse_factor_lab.analysis.risk_overlay_daily as daily
from twse_factor_lab.analysis.risk_overlay import (
    calculate_ma60_breadth,
    exposure_for_breadth,
)


def _events(sessions):
    return pd.DataFrame(
        {
            "signal_date": [sessions[0], sessions[0], sessions[2], sessions[2]],
            "execution_date": [sessions[1], sessions[1], sessions[3], sessions[3]],
            "ticker": ["A", "B", "A", "B"],
            "target_weight": [0.6, 0.4, 0.7, 0.3],
        }
    )


def test_daily_breadth_no_lookahead_and_threshold_boundary():
    dates = pd.date_range("2024-01-01", periods=65, freq="D")
    close = pd.DataFrame({"A": 10.0, "B": 10.0}, index=dates)
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
    original_value = original.loc[original.trade_date == dates[-2], "breadth"].iloc[0]
    revised_value = revised.loc[revised.trade_date == dates[-2], "breadth"].iloc[0]
    assert original_value == pytest.approx(revised_value)
    assert exposure_for_breadth(0.40) == daily.EXPOSURE_LOW
    assert exposure_for_breadth(0.40001) == daily.EXPOSURE_HIGH


def test_daily_signal_executes_next_session_and_repeated_state_has_no_transition():
    sessions = pd.date_range("2024-01-01", periods=6, freq="D")
    breadth = pd.DataFrame(
        {"trade_date": sessions, "breadth": [0.8, 0.8, 0.2, 0.2, 0.8, 0.8]}
    )
    weights, history, audit = daily._daily_targets(_events(sessions), breadth, sessions)
    assert history.loc[0, "execution_date"] == sessions[1]
    assert audit.loc[2, "breadth_signal_date"] == sessions[1]
    assert audit.loc[2, "new_exposure"] == daily.EXPOSURE_HIGH
    assert audit.loc[3, "exposure_transition"] == "1.0_TO_0.5"
    assert audit.loc[4, "exposure_transition"] == "NO_CHANGE"
    assert audit.loc[5, "exposure_transition"] == "0.5_TO_1.0"
    assert weights.loc[sessions[2]].equals(weights.loc[sessions[1]])
    assert weights.loc[sessions[3]].sum() == pytest.approx(0.5)
    assert set(weights.columns) == {"A", "B"}


def test_daily_targets_preserve_selection_and_relative_weights_on_non_rebalance():
    sessions = pd.date_range("2024-01-01", periods=5, freq="D")
    breadth = pd.DataFrame(
        {"trade_date": sessions, "breadth": [0.8, 0.2, 0.2, 0.8, 0.8]}
    )
    _, _, audit = daily._daily_targets(_events(sessions), breadth, sessions)
    non_rebalance = audit.loc[~audit["is_rebalance"]]
    assert non_rebalance["selected_tickers_changed"].eq(False).all()
    assert non_rebalance["relative_weights_preserved"].all()


def test_daily_events_require_signal_before_execution():
    sessions = pd.date_range("2024-01-01", periods=3, freq="D")
    weights = pd.DataFrame(
        [[0.0, 0.0], [0.6, 0.4], [0.3, 0.2]],
        index=sessions,
        columns=["A", "B"],
    )
    events = daily._daily_events(weights, sessions)
    valid = events["signal_date"].notna()
    assert (
        events.loc[valid, "signal_date"] < events.loc[valid, "execution_date"]
    ).all()


def test_daily_evaluation_is_identity_bound_and_outputs_required_artifacts():
    result = daily.evaluate_daily_risk_overlay(".")
    assert result["base_strategy_fingerprint"] == (
        "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f"
    )
    assert result["base_replay_matches_canonical"] is True
    assert result["base_metrics_match_previous"] is True
    assert result["previous_reb60_overlay"]["classification"] == "ADVERSE"
    assert result["fresh_oos_status"] == "UNCHANGED / INSUFFICIENT_DATA"
    assert result["production_ready"] is False
    output = daily.Path(
        "data/research/fundamental-production-final-v1/risk-overlay-daily"
    )
    assert len(list((output / "charts").glob("*.png"))) == 5
    assert (output / "2024_2025_drawdown_overlay_audit.csv").exists()
    audit = pd.read_parquet(output / "daily_overlay_trade_audit.parquet")
    assert audit["relative_weights_preserved"].all()
    assert result["exposure_activity"]["transition_count"] > 0
    assert (
        result["daily"]["metrics"]["transaction_cost"]
        > result["base"]["metrics"]["transaction_cost"]
    )
    non_rebalance = audit.loc[~audit["is_rebalance"]]
    traded = non_rebalance["turnover"] > 1e-12
    assert non_rebalance.loc[traded, "exposure_transition"].ne("NO_CHANGE").all()
    signal_column = (
        audit["signal_date"] if "signal_date" in audit else audit["breadth_signal_date"]
    )
    assert signal_column.isna().any()


def test_daily_fingerprint_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(
        daily,
        "load_frozen_identity",
        lambda root: SimpleNamespace(
            strategy_id=daily.STRATEGY_ID,
            strategy_fingerprint="changed",
        ),
    )
    with pytest.raises(daily.FrozenStrategyChanged, match="FROZEN_STRATEGY_CHANGED"):
        daily.evaluate_daily_risk_overlay(".")


def test_daily_policy_gap_is_rejected(tmp_path):
    path = tmp_path / daily.PREVIOUS_VALIDATION_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "overlay_id": "market_breadth_ma60_040_half_exposure_v1",
                "classification": "ADVERSE",
                "classification_policy": {"changed": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(daily.DailyRiskOverlayError, match="CLASSIFICATION_POLICY_GAP"):
        daily._policy_payload(tmp_path)


def test_daily_output_is_idempotent_for_core_metrics():
    first = daily.evaluate_daily_risk_overlay(".")
    second = daily.evaluate_daily_risk_overlay(".")
    assert first["comparison"] == second["comparison"]
    assert first["exposure_activity"] == second["exposure_activity"]
    assert (
        first["previous_overlay_artifact_sha256"]
        == second["previous_overlay_artifact_sha256"]
    )

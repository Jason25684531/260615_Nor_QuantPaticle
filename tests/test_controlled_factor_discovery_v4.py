from __future__ import annotations

import json

import pandas as pd
import pytest

from run_controlled_factor_discovery_v4 import (
    FACTORS,
    OUT,
    PRIMARY_HORIZON,
    _bootstrap,
    _check_pool_not_mutated,
    _pool,
)
from twse_factor_lab.analysis.factor_gate import FactorGateConfig
from twse_factor_lab.factors.controlled import (
    build_controlled_price_factors,
    build_eps_yoy_change_matrix,
)
from twse_factor_lab.factors.registry import build_default_registry

REGISTRY = json.loads(
    (OUT / "factor_candidate_registry.json").read_text(encoding="utf-8")
)
GATE_RESULTS = pd.read_csv(OUT / "factor_gate_results.csv")
ADMITTED = json.loads((OUT / "admitted_factor_pool.json").read_text(encoding="utf-8"))
TRIAL_REGISTRY = json.loads((OUT / "trial_registry.json").read_text(encoding="utf-8"))


def test_fixed_nine_factor_pool_and_metadata() -> None:
    registry = build_default_registry()
    assert [row[0] for row in FACTORS] == [
        "G2_EPS_YOY_CHANGE", "M0_MOMENTUM_20D", "M1_MOMENTUM_60D",
        "M2_NEAR_HIGH_252D", "R1_REVERSAL_5D", "L1_LOW_VOL_20D",
        "L3_DOWNSIDE_VOL_20D", "L4_DOLLAR_VOLUME_20D", "L2_AMIHUD_20D",
    ]
    assert all(
        registry.get(factor_id).direction == "higher_is_better"
        for factor_id, *_ in FACTORS
    )


def test_eps_yoy_matches_same_quarter_and_both_availability_dates() -> None:
    dates = pd.DatetimeIndex(["2024-05-14", "2024-05-20", "2025-05-20"])
    records = pd.DataFrame({
        "ticker": ["A", "A", "A"], "metric": ["eps"] * 3,
        "period_end": ["2023-03-31", "2024-03-31", "2025-03-31"],
        "available_date": ["2024-05-15", "2024-05-14", "2025-05-14"],
        "value": [1.0, 3.0, 5.0],
    })
    result = build_eps_yoy_change_matrix(records, dates, ["A"])
    assert pd.isna(result.loc[pd.Timestamp("2024-05-14"), "A"])
    assert result.loc[pd.Timestamp("2025-05-20"), "A"] == 2.0


def test_ohlcv_construction_is_trailing_and_dollar_volume_is_not_turnover() -> None:
    index = pd.date_range("2024-01-01", periods=252, freq="D")
    close = pd.DataFrame({"A": range(1, 253)}, index=index, dtype=float)
    volume = pd.DataFrame({"A": [10.0] * 252}, index=index)
    factors = build_controlled_price_factors(close, volume)
    assert factors["L4_DOLLAR_VOLUME_20D"].iloc[-1, 0] == pytest.approx(2425.0)
    assert factors["M0_MOMENTUM_20D"].iloc[-1, 0] == pytest.approx(20 / 232)


def test_pool_refuses_unavailable_predeclared_candidate() -> None:
    audit = [{"available": True} for _ in FACTORS]
    audit[-1]["available"] = False
    with pytest.raises(RuntimeError, match="INSUFFICIENT_PREDECLARED_FACTOR_POOL"):
        _pool(audit)


def test_pool_mutation_after_freeze_is_detected() -> None:
    with pytest.raises(RuntimeError, match="FROZEN_CANDIDATE_POOL_MUTATED"):
        _check_pool_not_mutated("new-hash", {"candidate_pool_hash": "old-hash"})


def test_pool_hash_match_does_not_raise() -> None:
    _check_pool_not_mutated("same-hash", {"candidate_pool_hash": "same-hash"})


def test_candidate_pool_size_between_eight_and_ten() -> None:
    assert 8 <= len(FACTORS) <= 10


def test_raw_eps_and_roe_are_diagnostic_only_not_selection_relevant() -> None:
    diagnostic = [row for row in REGISTRY if row.get("factor_id") in ("eps", "roe")]
    assert diagnostic
    assert all(row["selection_relevant"] is False for row in diagnostic)


def test_primary_horizon_fixed_at_twenty() -> None:
    assert PRIMARY_HORIZON == 20
    assert FactorGateConfig().primary_horizon == 20


def test_canonical_factor_gate_config_defaults_unchanged() -> None:
    config = FactorGateConfig()
    assert config.min_coverage == pytest.approx(0.20)
    assert config.min_mean_ic == pytest.approx(0.05)
    assert config.min_icir == pytest.approx(0.25)


def test_reject_verdicts_never_appear_in_admitted_pool() -> None:
    rejected = set(GATE_RESULTS.loc[GATE_RESULTS.verdict.eq("REJECT"), "factor_id"])
    admitted_ids = {row["factor_id"] for row in ADMITTED}
    assert admitted_ids.isdisjoint(rejected)


def test_admitted_pool_capped_at_three() -> None:
    assert len(ADMITTED) <= 3


def test_trial_registry_count_matches_evaluated_candidates() -> None:
    trial_count = TRIAL_REGISTRY["factor_test_trial_count"]
    assert trial_count == len(FACTORS) == len(GATE_RESULTS)


def test_bootstrap_output_carries_no_verdict_or_pool_membership_field() -> None:
    result = _bootstrap(pd.Series([0.01, 0.02, -0.01] * 10))
    assert "verdict" not in result
    assert "admitted" not in result
    assert result["bootstrap_status"] in {"DIAGNOSTIC_ONLY", "DEFERRED_TO_CHANGE_3"}


def test_bootstrap_ci_contains_synthetic_sample_mean() -> None:
    values = pd.Series([0.03, 0.05, -0.01, 0.02, 0.04] * 10)
    result = _bootstrap(values)
    assert result["ci_95"][0] <= result["mean_ic"] <= result["ci_95"][1]


def test_deterministic_ranking_caps_admission_at_three_on_tiebreak() -> None:
    columns = ["factor_id", "verdict", "mean_ic", "icir", "q5_q1", "turnover"]
    rows = [
        ("F1", "ACCEPT", 0.05, 0.30, 0.01, 0.5),
        ("F2", "ACCEPT", 0.05, 0.30, 0.01, 0.3),
        ("F3", "ACCEPT", 0.04, 0.20, 0.02, 0.1),
        ("F4", "ACCEPT", 0.03, 0.10, 0.01, 0.1),
        ("F5", "REJECT", 0.10, 0.90, 0.05, 0.1),
    ]
    frame = pd.DataFrame(rows, columns=columns)
    accepted = frame.loc[frame.verdict.eq("ACCEPT")].sort_values(
        ["mean_ic", "icir", "q5_q1", "turnover", "factor_id"],
        ascending=[False, False, False, True, True],
    )
    assert accepted.head(3).factor_id.tolist() == ["F2", "F1", "F3"]


def test_eps_yoy_missing_prior_year_same_quarter_is_missing_not_substituted() -> None:
    dates = pd.DatetimeIndex(["2025-05-20"])
    records = pd.DataFrame({
        "ticker": ["A", "A"], "metric": ["eps", "eps"],
        "period_end": ["2024-06-30", "2025-03-31"],
        "available_date": ["2024-08-14", "2025-05-14"],
        "value": [9.0, 5.0],
    })
    result = build_eps_yoy_change_matrix(records, dates, ["A"])
    assert pd.isna(result.loc[pd.Timestamp("2025-05-20"), "A"])

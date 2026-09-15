"""Tests C1-C20 for composite-factor-admission-v1."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from run_composite_factor_admission_v1 import OUT, SOURCE
from twse_factor_lab.analysis.composite_gate import (
    CompositeCandidateGateConfig,
    CompositeFactorGateConfig,
    build_composite_score,
    build_redundancy_report,
    classify_composite_candidate,
    compare_composite_to_components,
    compute_median_rank_correlation,
    greedy_select_components,
    rank_candidates,
)
from twse_factor_lab.analysis.factor_gate import FactorGateConfig

GATE_RESULTS = pd.read_csv(SOURCE / "factor_gate_results.csv")
ADMITTED = json.loads(
    (SOURCE / "admitted_factor_pool.json").read_text(encoding="utf-8")
)
ADMISSION = json.loads((OUT / "composite_admission.json").read_text(encoding="utf-8"))
TRIAL_REGISTRY = json.loads((OUT / "trial_registry.json").read_text(encoding="utf-8"))
COMPOSITE_DEFINITION = json.loads(
    (OUT / "composite_definition.json").read_text(encoding="utf-8")
)
GATE_SUMMARY = json.loads(
    (OUT / "composite_factor_gate_summary.json").read_text(encoding="utf-8")
)
RUN_MANIFEST = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))


def _base_kwargs(**overrides):
    kwargs = dict(
        coverage=0.25,
        mean_ic=0.03,
        icir=0.20,
        q5_q1=0.01,
        valid_year_count=3,
        positive_year_ratio=1.0,
        config=CompositeCandidateGateConfig(),
    )
    kwargs.update(overrides)
    return kwargs


# 7.1 -----------------------------------------------------------------------


def test_c1_historical_strong_verdicts_byte_identical() -> None:
    assert set(GATE_RESULTS["verdict"]) == {"REJECT"}
    assert len(GATE_RESULTS) == 9
    assert ADMITTED == []


def test_c2_canonical_factor_gate_config_unchanged() -> None:
    config = FactorGateConfig()
    assert config.min_coverage == pytest.approx(0.20)
    assert config.min_mean_ic == pytest.approx(0.05)
    assert config.min_icir == pytest.approx(0.25)
    assert config.primary_horizon == 20


# 7.2 -----------------------------------------------------------------------


def test_c3_composite_candidate_gate_thresholds_fixed() -> None:
    config = CompositeCandidateGateConfig()
    assert config.min_coverage == pytest.approx(0.20)
    assert config.min_mean_ic == pytest.approx(0.025)
    assert config.min_icir == pytest.approx(0.15)
    assert config.primary_horizon == 20
    assert config.min_valid_year_count == 3
    assert config.min_positive_year_ratio == pytest.approx(0.50)


def test_c4_coverage_below_threshold_rejects() -> None:
    status, failed = classify_composite_candidate(**_base_kwargs(coverage=0.19))
    assert status == "REJECT"
    assert any("coverage" in reason for reason in failed)


def test_c5_mean_ic_below_threshold_rejects() -> None:
    status, failed = classify_composite_candidate(**_base_kwargs(mean_ic=0.02))
    assert status == "REJECT"
    assert any("mean_ic" in reason for reason in failed)


def test_c6_icir_below_threshold_rejects() -> None:
    status, failed = classify_composite_candidate(**_base_kwargs(icir=0.10))
    assert status == "REJECT"
    assert any("icir" in reason for reason in failed)


def test_c7_negative_ic_cannot_candidate() -> None:
    status, failed = classify_composite_candidate(
        **_base_kwargs(mean_ic=-0.05, icir=0.30)
    )
    assert status == "REJECT"
    assert any("mean_ic" in reason for reason in failed)


def test_c8_year_stability_below_half_rejects() -> None:
    status, failed = classify_composite_candidate(
        **_base_kwargs(positive_year_ratio=0.33)
    )
    assert status == "REJECT"
    assert any("positive_year_ratio" in reason for reason in failed)


def test_c8_passing_candidate_accepts() -> None:
    status, failed = classify_composite_candidate(**_base_kwargs())
    assert status == "COMPOSITE_CANDIDATE"
    assert failed == []


# 7.3 -----------------------------------------------------------------------


def test_c9_candidate_pool_capped_at_three() -> None:
    calibration_report = [
        {
            "factor_id": fid,
            "composite_candidate_status": "COMPOSITE_CANDIDATE",
            "mean_ic": mean_ic,
            "icir": 0.20,
            "q5_q1": 0.01,
            "coverage": 0.25,
        }
        for fid, mean_ic in [
            ("A", 0.05),
            ("B", 0.04),
            ("C", 0.03),
            ("D", 0.02),
            ("E", 0.01),
        ]
    ]
    ranked = rank_candidates(calibration_report)
    assert len(ranked) == 3
    assert [row["factor_id"] for row in ranked] == ["A", "B", "C"]
    assert [row["selection_rank"] for row in ranked] == [1, 2, 3]


def test_c10_redundancy_threshold_is_point_eight() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    tickers = [f"T{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    identical = pd.DataFrame(rng.normal(size=(40, 8)), index=dates, columns=tickers)
    duplicate = identical * 3.0 + 1.0  # same rank order every day -> corr == 1.0
    independent = pd.DataFrame(rng.normal(size=(40, 8)), index=dates, columns=tickers)

    matrices = {"A": identical, "B": duplicate, "C": independent}
    matrix, report = build_redundancy_report(["A", "B", "C"], matrices)

    assert matrix.loc["A", "B"] == pytest.approx(1.0)
    high = {(p["factor_a"], p["factor_b"]) for p in report["high_redundancy_pairs"]}
    assert ("A", "B") in high
    assert ("A", "C") not in high
    assert report["threshold"] == pytest.approx(0.80)


def test_c11_greedy_selection_stops_below_two_when_all_redundant() -> None:
    ranked = [{"factor_id": fid} for fid in ("A", "B", "C")]
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.95, 0.90], [0.95, 1.0, 0.92], [0.90, 0.92, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    selected = greedy_select_components(ranked, correlation_matrix)
    assert selected == ["A"]
    assert len(selected) < 2  # would trigger INSUFFICIENT_NON_REDUNDANT_FACTORS


def test_c11_greedy_selection_keeps_complementary_pair() -> None:
    ranked = [{"factor_id": fid} for fid in ("A", "B")]
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    assert greedy_select_components(ranked, correlation_matrix) == ["A", "B"]


def test_c12_equal_weights_only() -> None:
    n = len(COMPOSITE_DEFINITION["components"])
    assert n >= 2
    for weight in COMPOSITE_DEFINITION["weights"].values():
        assert weight == pytest.approx(1.0 / n)
    assert sum(COMPOSITE_DEFINITION["weights"].values()) == pytest.approx(1.0)


def test_c13_missing_component_excluded_complete_case() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    tickers = ["A", "B", "C"]
    factor_x = pd.DataFrame([[1, 2, 3]] * 3, index=dates, columns=tickers, dtype=float)
    factor_y = pd.DataFrame([[3, 2, 1]] * 3, index=dates, columns=tickers, dtype=float)
    factor_y.loc[dates[0], "B"] = np.nan

    composite = build_composite_score({"X": factor_x, "Y": factor_y}, ["X", "Y"])

    assert pd.isna(composite.loc[dates[0], "B"])
    # Percentile ranks over three complete tickers are 2/3 for the middle
    # ticker in each component; complete-case handling does not reweight it.
    assert composite.loc[dates[1], "B"] == pytest.approx(2 / 3)


# 7.4 -----------------------------------------------------------------------


def test_c14_composite_factor_gate_uses_horizon_twenty() -> None:
    assert CompositeFactorGateConfig().primary_horizon == 20


def test_c15_diagnostic_horizons_cannot_alter_verdict() -> None:
    horizon_1 = next(
        h for h in GATE_SUMMARY["diagnostics"]["horizon_metrics"] if h["horizon"] == 1
    )
    assert horizon_1["signal_pass"] is False
    assert horizon_1["stability_pass"] is False
    assert GATE_SUMMARY["diagnostics"]["selected_horizon"] == 20
    assert GATE_SUMMARY["verdict"] == "PASS"


def test_c16_composite_factor_gate_thresholds() -> None:
    config = CompositeFactorGateConfig()
    assert config.min_coverage == pytest.approx(0.20)
    assert config.min_mean_ic == pytest.approx(0.03)
    assert config.min_icir == pytest.approx(0.20)


def test_c17_diagnostic_comparison_cannot_mutate_gate_membership() -> None:
    result = compare_composite_to_components(
        {"mean_ic": 0.03, "icir": 0.20},
        {"A": {"mean_ic": 0.02, "icir": 0.15}},
    )
    forbidden = {"verdict", "composite_gate_verdict", "ready_for_change_2", "admitted"}
    assert forbidden.isdisjoint(result.keys())


# 7.5 -----------------------------------------------------------------------


def test_c18_no_strategy_artifacts_created() -> None:
    forbidden_terms = (
        "backtest",
        "topn",
        "top_n",
        "rebalance",
        "cost",
        "portfolio",
        "pyfolio",
        "walkforward",
        "walk_forward",
        "psr",
        "dsr",
    )
    names = [path.name.lower() for path in OUT.iterdir()]
    for name in names:
        assert not any(term in name for term in forbidden_terms), name


def test_c19_trial_registry_post_hoc_flag_present() -> None:
    assert TRIAL_REGISTRY["trials"]
    for trial in TRIAL_REGISTRY["trials"]:
        assert trial["post_hoc"] is True
        assert trial["confirmatory"] is False


def test_c20_frozen_evidence_unchanged_after_run() -> None:
    for name, recorded_hash in RUN_MANIFEST["source_evidence_hashes"].items():
        current_hash = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        assert current_hash == recorded_hash


def test_median_rank_correlation_perfect_positive() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    a = pd.DataFrame(
        [[1, 2, 3, 4, 5]] * 5, index=dates, columns=list("ABCDE"), dtype=float
    )
    b = a * 2.0 + 7.0
    assert compute_median_rank_correlation(a, b) == pytest.approx(1.0)

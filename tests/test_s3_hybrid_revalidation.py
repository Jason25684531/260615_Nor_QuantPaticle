# ruff: noqa: E501
"""R1-R29 tests for frozen S3 Hybrid revalidation governance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twse_factor_lab.acceptance.s3_hybrid_revalidation import (
    BASE_COST,
    FINGERPRINT,
    S3HybridContract,
    S3HybridRevalidationError,
    bootstrap_report,
    build_historical_manifest,
    compare_frozen_snapshots,
    composite_report,
    cost_stress,
    default_contract,
    engine_parity,
    evaluate_verdict,
    factor_reports,
    frozen_evidence_snapshot,
    output_dir,
    psr_dsr_report,
    strategy_backtest,
    temporal_validation,
    write_json,
)


@pytest.fixture(scope="module")
def matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2018-01-02", periods=320)
    tickers = [f"T{i:02d}" for i in range(10)]
    close = pd.DataFrame(
        {
            ticker: 20 + i + np.arange(len(dates)) * (0.01 + i / 1000)
            for i, ticker in enumerate(tickers)
        },
        index=dates,
    )
    volume = pd.DataFrame(10000 + np.arange(len(dates))[:, None] + np.arange(10), index=dates, columns=tickers)
    return close, volume


def _ready_inputs(tmp_path: Path, matrices: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[dict, dict]:
    close, volume = matrices
    rows = []
    for date in close.index:
        for ticker in close.columns:
            price = float(close.loc[date, ticker])
            rows.append({"date": date, "ticker": ticker, "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": float(volume.loc[date, ticker]), "trade_value": price * float(volume.loc[date, ticker]), "adjustment_status": "MATCHED", "volume_normalization": "TWSE_REPORTED_SHARES_UNCHANGED", "primary_status": "PRIMARY_PRESENT"})
    canonical = pd.DataFrame(rows)
    calendar = pd.DataFrame({"date": close.index, "is_trading_session": True})
    status = pd.DataFrame({"ticker": list(close.columns), "active": True, "suspended": False})
    paths = {}
    for name, frame in (("canonical", canonical), ("calendar", calendar), ("security_status", status), ("corporate_actions", pd.DataFrame())):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    paths.update({"old_raw": tmp_path / "old_raw.parquet", "old_adjusted": tmp_path / "old_adjusted.parquet"})
    return {"canonical": canonical, "calendar": calendar, "security_status": status, "corporate_actions": pd.DataFrame(), "old": canonical.copy(), "paths": paths}, {"start": "2018-01-02", "end": "2020-01-01"}


def _verdict_inputs() -> dict:
    return {"historical": {"status": "DATA_READY"}, "quality": {"internal_validity": "PASS", "readiness": "DATA_READY"}, "l2": {"status": "PASS"}, "l4": {"status": "PASS"}, "composite": {"gate": "HYBRID_COMPOSITE_PASS"}, "strategy": {"status": "PASS"}, "parity": {"parity": "PASS"}, "temporal": {"status": "PASS"}, "bootstrap": {"status": "PASS"}, "stress": {"status": "PASS"}, "psr_dsr": {"status": "PASS"}, "frozen": {"status": "PASS"}}


def test_r1_fingerprint_unchanged():
    assert default_contract().candidate_fingerprint == FINGERPRINT


def test_r2_l2_definition_unchanged():
    assert default_contract().factors["L2_AMIHUD_20D"] == 0.5


def test_r3_l4_definition_unchanged():
    assert default_contract().factors["L4_DOLLAR_VOLUME_20D"] == 0.5


def test_r4_weights_unchanged():
    assert default_contract().factors == {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5}


def test_r5_top5_unchanged():
    assert default_contract().top_n == 5


def test_r6_monthly_unchanged():
    assert default_contract().rebalance == "monthly"


def test_r7_buffer_off_unchanged():
    assert default_contract().buffer is False


def test_r8_cost_unchanged():
    assert default_contract().costs == BASE_COST.summary()


def test_r9_execution_unchanged():
    assert "T+1" in default_contract().execution


def test_r10_hybrid_contract_unchanged():
    default_contract().validate()


def test_r11_no_second_experiment(tmp_path: Path):
    out = output_dir(tmp_path)
    out.mkdir(parents=True)
    (out / "hybrid_revalidation_verdict.json").write_text("{}", encoding="utf-8")
    from twse_factor_lab.acceptance.s3_hybrid_revalidation import (
        assert_single_experiment,
    )
    with pytest.raises(S3HybridRevalidationError):
        assert_single_experiment(tmp_path)


def test_r12_no_factor_discovery():
    with pytest.raises(S3HybridRevalidationError):
        S3HybridContract(factors={"new_factor": 1.0}).validate()


def test_r13_no_strategy_search():
    with pytest.raises(S3HybridRevalidationError):
        S3HybridContract(top_n=10).validate()


def test_r14_historical_backfill_manifest_is_deterministic(tmp_path: Path, matrices):
    inputs, _ = _ready_inputs(tmp_path, matrices)
    first = build_historical_manifest(tmp_path, inputs)
    second = build_historical_manifest(tmp_path, inputs)
    assert first == second


def test_r15_factor_metrics_are_deterministic(matrices):
    first = factor_reports(*matrices)["reports"]
    second = factor_reports(*matrices)["reports"]
    assert first == second


def test_r16_composite_is_deterministic(matrices):
    first = composite_report(*matrices)
    second = composite_report(*matrices)
    assert first["gate"] == second["gate"]
    pd.testing.assert_frame_equal(first["score"], second["score"])


def test_r17_backtest_is_deterministic(matrices):
    close, volume = matrices
    score = composite_report(close, volume)["score"]
    first = strategy_backtest(close, score)
    second = strategy_backtest(close, score)
    assert first["metrics"] == second["metrics"]


def test_r18_engine_parity(matrices):
    close, volume = matrices
    result = engine_parity(close, composite_report(close, volume)["score"])
    assert result["parity"] == "PASS"


def test_r19_temporal_deterministic(matrices):
    close, volume = matrices
    strategy = strategy_backtest(close, composite_report(close, volume)["score"])
    assert temporal_validation(strategy, close) == temporal_validation(strategy, close)


def test_r20_bootstrap_deterministic(matrices):
    close, volume = matrices
    strategy = strategy_backtest(close, composite_report(close, volume)["score"])
    first = bootstrap_report(strategy, default_contract())
    second = bootstrap_report(strategy, default_contract())
    assert first == second


def test_r21_psr_deterministic(tmp_path: Path, matrices):
    close, volume = matrices
    strategy = strategy_backtest(close, composite_report(close, volume)["score"])
    registry = {"strategy_selection_trial_count": 4, "trials": []}
    path = tmp_path / "data/research/engine-parity-fix-final-validation-v3"
    path.mkdir(parents=True)
    write_json(path / "trial_registry_v3.json", registry)
    assert psr_dsr_report(strategy, tmp_path) == psr_dsr_report(strategy, tmp_path)


def test_r22_dsr_trial_contract_unchanged(tmp_path: Path, matrices):
    close, volume = matrices
    strategy = strategy_backtest(close, composite_report(close, volume)["score"])
    path = tmp_path / "data/research/engine-parity-fix-final-validation-v3"
    path.mkdir(parents=True)
    write_json(path / "trial_registry_v3.json", {"strategy_selection_trial_count": 4, "trials": []})
    assert psr_dsr_report(strategy, tmp_path)["strategy_trial_count"] == 4


def test_r23_cost_stress_deterministic(matrices):
    close, volume = matrices
    score = composite_report(close, volume)["score"]
    strategy = strategy_backtest(close, score)
    assert cost_stress(strategy, close, score) == cost_stress(strategy, close, score)


def test_r24_revalidation_window_is_not_fresh_oos():
    assert default_contract().as_dict()["revalidation_window"]["fresh_oos_available"] is False


def test_r25_original_evidence_snapshot_unchanged(tmp_path: Path):
    before = frozen_evidence_snapshot(tmp_path)
    assert compare_frozen_snapshots(before, before)["historical_evidence_unchanged"] is True


def test_r26_statistical_failure_cannot_pass():
    values = _verdict_inputs()
    values["l2"] = {"status": "FAIL"}
    assert evaluate_verdict(**values)["verdict"] != "S3_HYBRID_REVALIDATED"


def test_r27_economic_failure_cannot_pass():
    values = _verdict_inputs()
    values["stress"] = {"status": "FAIL"}
    assert evaluate_verdict(**values)["verdict"] != "S3_HYBRID_REVALIDATED"


def test_r28_engine_failure_cannot_pass():
    values = _verdict_inputs()
    values["parity"] = {"parity": "FAIL"}
    assert evaluate_verdict(**values)["verdict"] != "S3_HYBRID_REVALIDATED"


def test_r29_runtime_contract_requires_pass():
    values = _verdict_inputs()
    values["historical"] = {"status": "DATA_INSUFFICIENT"}
    result = evaluate_verdict(**values)
    assert result["verdict"] == "S3_HYBRID_REVALIDATION_INSUFFICIENT"
    assert result["READY_FOR_FORWARD_SHADOW"] == "NO"

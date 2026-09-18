# ruff: noqa: E501
"""H1-H20 governance tests for the single hybrid revalidation experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from twse_factor_lab.data.hybrid_revalidation import (
    DataFrameSource,
    HybridCanonicalBuilder,
    HybridRevalidationError,
    assert_single_experiment,
    corporate_action_normalizer,
    default_contract,
    evaluate_verdict,
    factor_reconciliation,
    output_dir,
    signal_reconciliation,
)


def _frames(periods: int = 60) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-02", periods=periods)
    tickers = ["1101", "2330", "6669", "9999", "1000", "2000"]
    raw_rows = []
    yf_rows = []
    for day, date in enumerate(dates):
        for number, ticker in enumerate(tickers):
            close = 10.0 + number + day / 100
            raw_rows.append({"date": date, "ticker": ticker, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000 + number + day, "trade_value": close * (1000 + number + day)})
            yf_rows.append({"date": date, "ticker": ticker, "open": close, "high": close + 1, "low": close - 1, "close": close, "raw_close": close, "adjusted_close": close, "volume": 1000 + number + day})
    raw = pd.DataFrame(raw_rows)
    yfinance = pd.DataFrame(yf_rows)
    calendar = pd.DataFrame({"date": dates, "is_trading_session": True})
    status = pd.DataFrame({"ticker": tickers, "active": True})
    return raw, yfinance, calendar, status


def _build(tmp_path: Path, **kwargs):
    raw, yfinance, calendar, status = _frames()
    raw = kwargs.pop("raw", raw)
    yfinance = kwargs.pop("yfinance", yfinance)
    return HybridCanonicalBuilder(
        tmp_path,
        DataFrameSource(raw, "TWSE_OFFICIAL", "TWSE_OFFICIAL"),
        DataFrameSource(yfinance, "YFINANCE", "YFINANCE"),
        calendar=kwargs.pop("calendar", calendar),
        security_status=kwargs.pop("status", status),
        **kwargs,
    ).build()


def _failed_report(**overrides):
    data = {"calendar": {"status": "PASS"}}
    factor = {"status": "PASS"}
    signal = {"top5_mismatch_count": 0, "target_mismatch_count": 0}
    values = {
        "contract_status": "PASS",
        "data_report": data,
        "factor_report": factor,
        "signal_report": signal,
        "performance_report": {"classification": "NUMERICALLY_EQUIVALENT"},
        "gap_fill_governance": True,
        "adjustment_status": "PASS",
        "volume_status": "PASS",
    }
    values.update(overrides)
    return evaluate_verdict(**values)


def test_h1_source_roles_are_fixed():
    assert default_contract().source_roles["RAW_PRICE_AUTHORITY"] == "TWSE_OFFICIAL"
    with pytest.raises(HybridRevalidationError):
        HybridCanonicalBuilder(Path("."), DataFrameSource(pd.DataFrame(), "YFINANCE", "YFINANCE"), DataFrameSource(pd.DataFrame(), "TWSE_OFFICIAL", "TWSE_OFFICIAL"))


def test_h2_single_experiment_is_enforced(tmp_path: Path):
    out = output_dir(tmp_path)
    out.mkdir(parents=True)
    (out / "hybrid_revalidation_verdict.json").write_text("{}", encoding="utf-8")
    with pytest.raises(HybridRevalidationError, match="SECOND_EXPERIMENT_FORBIDDEN"):
        assert_single_experiment(tmp_path)


def test_h3_no_outcome_directed_source_switching():
    contract = default_contract()
    contract_bad = type(contract)(source_roles={**contract.source_roles, "VOLUME_AUTHORITY": "YFINANCE"})
    with pytest.raises(HybridRevalidationError, match="SOURCE_ROLE_CONTRACT_MISMATCH"):
        contract_bad.validate()


def test_h4_gap_fill_is_logged(tmp_path: Path):
    raw, yfinance, calendar, status = _frames(2)
    raw = raw[~((raw["date"] == raw["date"].iloc[0]) & raw["ticker"].eq("1101"))]
    result = _build(tmp_path, raw=raw, yfinance=yfinance, calendar=calendar, status=status)
    assert len(result["gap_log"]) == 5
    assert set(result["gap_log"]["field"]) >= {"open", "close", "volume"}


def test_h5_suspended_rows_are_not_gap_filled(tmp_path: Path):
    raw, yfinance, calendar, status = _frames(1)
    raw = raw.iloc[0:0]
    status.loc[status["ticker"].eq("1101"), "suspended"] = True
    result = _build(tmp_path, raw=raw, yfinance=yfinance, calendar=calendar, status=status)
    row = result["canonical"].loc[result["canonical"]["ticker"].eq("1101")].iloc[0]
    assert row["primary_status"] == "SUSPENDED"
    assert result["gap_log"].loc[result["gap_log"]["ticker"].eq("1101")].empty


def test_h6_delisted_rows_are_not_gap_filled(tmp_path: Path):
    raw, yfinance, calendar, status = _frames(1)
    raw = raw.iloc[0:0]
    status.loc[status["ticker"].eq("1101"), "active"] = False
    result = _build(tmp_path, raw=raw, yfinance=yfinance, calendar=calendar, status=status)
    row = result["canonical"].loc[result["canonical"]["ticker"].eq("1101")].iloc[0]
    assert row["primary_status"] == "DELISTED"
    assert result["gap_log"].loc[result["gap_log"]["ticker"].eq("1101")].empty


def test_h7_adjustment_is_deterministic(tmp_path: Path):
    first = _build(tmp_path / "one")
    second = _build(tmp_path / "two")
    pd.testing.assert_frame_equal(first["canonical"], second["canonical"])
    assert first["canonical"]["adjustment_factor"].eq(1.0).all()


def test_h8_volume_normalization_is_deterministic(tmp_path: Path):
    result = _build(tmp_path)
    assert result["canonical"]["volume_normalization"].eq("TWSE_REPORTED_SHARES_UNCHANGED").all()
    assert result["manifest"]["volume_status"] == "TWSE reported shares unchanged"


def test_h9_corporate_action_rule_is_generic():
    dates = pd.date_range("2026-01-02", periods=2)
    raw = pd.DataFrame({"date": dates, "ticker": ["6669", "6669"], "close": [100.0, 50.0]})
    adjusted = pd.DataFrame({"date": dates, "ticker": ["6669", "6669"], "close": [100.0, 100.0]})
    factors = pd.DataFrame({"date": dates, "ticker": ["6669", "6669"], "adjustment_factor": [1.0, 2.0]})
    result = corporate_action_normalizer(raw, adjusted, factors)
    assert result["generic_rule"] is True
    assert result["ticker_specific_rules"] == []
    assert result["event_count"] == 1


def test_h10_fresh_oos_window_is_immutable():
    with pytest.raises(HybridRevalidationError, match="FRESH_OOS_WINDOW_IMMUTABLE"):
        type(default_contract())(start_date="2026-01-03").validate()


def test_h11_s3_is_immutable():
    with pytest.raises(HybridRevalidationError, match="S3_STRATEGY_IMMUTABLE"):
        type(default_contract())(top_n=10).validate()


def test_h12_l2_reuses_the_canonical_implementation(tmp_path: Path):
    raw, yfinance, calendar, status = _frames()
    built = _build(tmp_path, raw=raw, yfinance=yfinance, calendar=calendar, status=status)
    old_close = built["canonical"].pivot(index="date", columns="ticker", values="close")
    old_volume = built["canonical"].pivot(index="date", columns="ticker", values="volume")
    _, report = factor_reconciliation(old_close, old_volume, old_close, old_volume)
    assert report["L2_mismatch_count"] == 0


def test_h13_l4_reuses_the_canonical_implementation(tmp_path: Path):
    raw, yfinance, calendar, status = _frames()
    built = _build(tmp_path, raw=raw, yfinance=yfinance, calendar=calendar, status=status)
    close = built["canonical"].pivot(index="date", columns="ticker", values="close")
    volume = built["canonical"].pivot(index="date", columns="ticker", values="volume")
    _, report = factor_reconciliation(close, volume, close, volume)
    assert report["L4_mismatch_count"] == 0


def test_h14_top5_comparison_is_deterministic(tmp_path: Path):
    result = _build(tmp_path)
    close = result["canonical"].pivot(index="date", columns="ticker", values="close")
    volume = result["canonical"].pivot(index="date", columns="ticker", values="volume")
    first, _, _ = signal_reconciliation(close, volume, close, volume)
    second, _, _ = signal_reconciliation(close, volume, close, volume)
    assert first == second


def test_h15_target_comparison_is_deterministic(tmp_path: Path):
    result = _build(tmp_path)
    close = result["canonical"].pivot(index="date", columns="ticker", values="close")
    volume = result["canonical"].pivot(index="date", columns="ticker", values="volume")
    report, old_targets, hybrid_targets = signal_reconciliation(close, volume, close, volume)
    pd.testing.assert_frame_equal(old_targets, hybrid_targets)
    assert report["target_mismatch_count"] == 0


def test_h16_performance_gate_is_deterministic():
    report = _failed_report()
    assert report["gates"]["performance"] is True
    assert report["verdict"] == "CANONICAL_HYBRID_DATA_APPROVED"


def test_h17_frozen_evidence_is_not_written(tmp_path: Path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text('{"frozen": true}\n', encoding="utf-8")
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    _build(tmp_path)
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before


def test_h18_pass_cannot_have_top5_mismatch():
    result = _failed_report(signal_report={"top5_mismatch_count": 1, "target_mismatch_count": 0})
    assert result["verdict"] == "HYBRID_REVALIDATION_FAILED"
    assert result["READY_FOR_FORWARD_SHADOW"] is False


def test_h19_pass_cannot_have_target_mismatch():
    result = _failed_report(signal_report={"top5_mismatch_count": 0, "target_mismatch_count": 1})
    assert result["verdict"] == "HYBRID_REVALIDATION_FAILED"


def test_h20_fail_cannot_trigger_automatic_second_experiment(tmp_path: Path):
    out = output_dir(tmp_path)
    out.mkdir(parents=True)
    (out / "hybrid_revalidation_verdict.json").write_text(
        json.dumps({"verdict": "HYBRID_REVALIDATION_FAILED"}), encoding="utf-8"
    )
    with pytest.raises(HybridRevalidationError):
        assert_single_experiment(tmp_path)

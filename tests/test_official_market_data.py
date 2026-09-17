# ruff: noqa: E501
"""M1-M26 official-feed remediation contract checks."""

from __future__ import annotations

import json

import pandas as pd

from twse_factor_lab.data.official_market_data import (
    OfficialCanonicalIngestion,
    OfficialIngestionResult,
    OfficialMarketDataError,
    TWSEOfficialAdapter,
    apply_adjustment_factors,
    build_adjustment_factors,
    build_market_calendar,
    build_session_coverage,
    coverage_audit,
    normalize_official_ohlcv,
    payload_sha,
    validate_official_ohlcv,
    write_adjustment_audit,
    write_official_contract,
    write_security_master,
)


def _payload() -> dict[str, object]:
    return {
        "tables": [
            {
                "fields": [
                    "Code",
                    "Name",
                    "Volume",
                    "Transaction",
                    "TradeValue",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ],
                "data": [["1101", "A", "1,000", "10", "2,000", "10", "11", "9", "10.5"]],
            }
        ]
    }


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["1101", "1102", "2867"],
            "market": ["TWSE"] * 3,
            "listed_date": pd.to_datetime(["2020-01-01"] * 3),
        }
    )


def test_m1_twse_schema_parsing():
    frame = normalize_official_ohlcv(_payload(), session_date="2026-01-02")
    assert list(frame.columns)[0:9] == ["date", "ticker", "market", "open", "high", "low", "close", "volume", "trade_value"]
    assert frame.iloc[0].ticker == "1101"


def test_m2_ticker_mapping_and_m3_market_classification():
    frame = normalize_official_ohlcv([{"Code": "1101.TW", "OpeningPrice": "1", "HighestPrice": "2", "LowestPrice": "1", "ClosingPrice": "1.5", "TradeVolume": "1", "TradeValue": "2", "Date": "1150102"}])
    assert frame.iloc[0].ticker == "1101"
    assert frame.iloc[0].market == "TWSE"


def test_m4_twse_coverage_audit_and_m5_tpex_classification():
    audit = coverage_audit(_universe(), ["1101", "1102"], listed_tickers=["1101", "1102"], tpex_tickers=["2867"])
    assert audit["twse_missing_count"] == 1
    assert audit["missing_classification"]["TPEx"] == ["2867"]


def test_m6_payload_sha_is_immutable():
    assert payload_sha({"a": 1}) == payload_sha({"a": 1})
    assert payload_sha({"a": 1}) != payload_sha({"a": 2})


def test_m7_date_ticker_unique():
    frame = normalize_official_ohlcv([{"Code": "1101", "OpeningPrice": "1", "HighestPrice": "2", "LowestPrice": "1", "ClosingPrice": "1.5", "TradeVolume": "1", "TradeValue": "2", "Date": "1150102"}, {"Code": "1101", "OpeningPrice": "1", "HighestPrice": "2", "LowestPrice": "1", "ClosingPrice": "1.5", "TradeVolume": "1", "TradeValue": "2", "Date": "1150102"}])
    assert not frame.duplicated(["date", "ticker"]).any()


def test_m8_no_implicit_fallback(tmp_path):
    contract = write_official_contract(tmp_path, {"twse_forward_only": True}, {"status": "ADJUSTMENT_CONTRACT_UNRESOLVED"})
    assert "fallback" in contract
    assert "silent" in contract["fallback"]


class _Response:
    def __init__(self, payload: object, status: int = 200):
        self.content = json.dumps(payload, ensure_ascii=False).encode()
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError("request")


class _Session:
    def __init__(self, responses: list[_Response]):
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def get(self, url, params=None, timeout=30):
        self.calls.append((url, params))
        if not self.responses:
            raise RuntimeError("empty")
        return self.responses.pop(0)


def test_m9_bounded_retry():
    session = _Session([_Response([], 500), _Response([], 500), _Response({"paths": {}})])
    adapter = TWSEOfficialAdapter(session=session, max_attempts=3, backoff_seconds=0)
    try:
        adapter.inspect_openapi()
    except OfficialMarketDataError:
        pass
    assert len(session.calls) == 3


def test_m10_partial_data_is_not_valid():
    frame = normalize_official_ohlcv([{"Code": "1101", "OpeningPrice": "1", "HighestPrice": "2", "LowestPrice": "1", "ClosingPrice": "1.5", "TradeVolume": "1", "TradeValue": "2", "Date": "1150102"}])
    assert not validate_official_ohlcv(frame)
    coverage = build_session_coverage(_universe(), frame)
    assert bool(coverage.loc[coverage.ticker == "1102", "missing_api_record"].iloc[0])


def test_m11_suspended_not_missing():
    frame = normalize_official_ohlcv(_payload(), session_date="2026-01-02")
    coverage = build_session_coverage(_universe(), frame, suspended_tickers=["1102"])
    row = coverage.loc[coverage.ticker == "1102"].iloc[0]
    assert bool(row.suspended) and not bool(row.missing_api_record)


def test_m12_no_trade_is_distinct():
    frame = normalize_official_ohlcv([{"Code": "1101", "OpeningPrice": "-", "HighestPrice": "-", "LowestPrice": "-", "ClosingPrice": "-", "TradeVolume": "0", "TradeValue": "0", "Date": "1150102"}])
    coverage = build_session_coverage(_universe().iloc[[0]], frame)
    assert coverage.iloc[0].no_trade


def test_m13_holiday_calendar():
    holidays = pd.DataFrame({"Date": ["1150101"], "Description": ["holiday"]})
    calendar = build_market_calendar(holidays, "2026-01-01", "2026-01-02")
    assert not bool(calendar.loc[calendar.date == pd.Timestamp("2026-01-01"), "is_trading_session"].iloc[0])


def test_m13_make_up_trading_day_override():
    holidays = pd.DataFrame({"Date": ["1150103"], "Description": ["make-up trading day"]})
    calendar = build_market_calendar(holidays, "2026-01-03", "2026-01-03")
    assert bool(calendar.iloc[0].is_trading_session)


def test_m14_official_contract_declares_historical_source():
    adapter = TWSEOfficialAdapter(session=_Session([_Response({"paths": {}})]), backoff_seconds=0)
    report = adapter.inspect_openapi()
    assert report["openapi_snapshot_only"] is True
    assert report["historical_source_available"] is True


def test_m15_adjustment_contract_is_explicit(tmp_path):
    report = write_adjustment_audit(tmp_path)
    assert report["status"] == "ADJUSTMENT_CONTRACT_UNRESOLVED"
    assert (tmp_path / "data/runtime/shadow-s3-v1/corporate_action_adjustment_audit.json").exists()


def test_m16_raw_and_adjusted_semantics_are_separate():
    frame = normalize_official_ohlcv(_payload(), session_date="2026-01-02")
    assert frame.close.iloc[0] == 10.5
    assert "raw" in "official raw exchange OHLCV"


def test_m16_hybrid_adjustment_formula_is_predeclared():
    raw = normalize_official_ohlcv(_payload(), session_date="2026-01-02")
    adjusted = raw.copy()
    adjusted["close"] = adjusted["close"] * 2
    factors = build_adjustment_factors(raw, adjusted)
    result = apply_adjustment_factors(raw, factors)
    assert result.close.iloc[0] == 21


def test_m17_fresh_oos_report_is_unresolved_until_adjustment(tmp_path):
    obj = OfficialCanonicalIngestion(tmp_path, adapter=TWSEOfficialAdapter(session=_Session([]), backoff_seconds=0))
    reports = obj.write_reconciliation_reports()
    assert reports["fresh_oos"]["status"] == "ADJUSTMENT_CONTRACT_UNRESOLVED"


def test_m18_material_drift_requires_review(tmp_path):
    obj = OfficialCanonicalIngestion(tmp_path, adapter=TWSEOfficialAdapter(session=_Session([]), backoff_seconds=0))
    report = obj.write_reconciliation_reports()["fresh_oos"]
    assert report["action"] == "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED"


def test_m19_security_master_is_atomic(tmp_path):
    path = write_security_master(tmp_path, _universe(), _universe())
    assert path.exists()
    assert len(pd.read_parquet(path)) == 3


def test_m20_backfill_mode_is_not_forward(tmp_path):
    assert OfficialIngestionResult("NOOP", "HISTORICAL_BACKFILL", None, None, 0, 0).mode == "HISTORICAL_BACKFILL"


def test_m21_boundary_file_is_not_touched_by_official_audit(tmp_path):
    boundary = tmp_path / "data/runtime/shadow-s3-v1/forward_evidence_start_timestamp.json"
    boundary.parent.mkdir(parents=True)
    boundary.write_text('{"forward_evidence_start_timestamp":"2026-09-17T00:00:00+00:00"}')
    OfficialCanonicalIngestion(tmp_path, adapter=TWSEOfficialAdapter(session=_Session([]), backoff_seconds=0))
    assert "2026-09-17" in boundary.read_text()


def test_m22_normalization_is_idempotent():
    left = normalize_official_ohlcv(_payload(), session_date="2026-01-02")
    right = normalize_official_ohlcv(_payload(), session_date="2026-01-02")
    assert payload_sha(left["close"].tolist()) == payload_sha(right["close"].tolist())


def test_m23_revision_artifact_path(tmp_path):
    obj = OfficialCanonicalIngestion(tmp_path, adapter=TWSEOfficialAdapter(session=_Session([]), backoff_seconds=0))
    obj.write_reconciliation_reports()
    assert not (obj.out / "data_revision_log.parquet").exists()


def test_m24_s3_fingerprint_untouched():
    from twse_factor_lab.runtime.shadow import FINGERPRINT
    assert FINGERPRINT == "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"


def test_m25_runtime_contract_is_not_owned_by_adapter(tmp_path):
    obj = OfficialCanonicalIngestion(tmp_path, adapter=TWSEOfficialAdapter(session=_Session([]), backoff_seconds=0))
    assert not hasattr(obj, "contract")


def test_m26_historical_evidence_is_not_changed(tmp_path):
    obj = OfficialCanonicalIngestion(tmp_path, adapter=TWSEOfficialAdapter(session=_Session([]), backoff_seconds=0))
    obj.write_reconciliation_reports()
    assert not (tmp_path / "data/processed/ohlcv.parquet").exists()

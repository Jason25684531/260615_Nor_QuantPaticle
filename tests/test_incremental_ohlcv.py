"""D1-D20 ingestion and forward-boundary checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from twse_factor_lab.data.incremental_ohlcv import (
    IncrementalCanonicalOHLCVUpdater,
    resolve_latest_completed_session,
)
from twse_factor_lab.data.yfinance_client import OhlcvDownloadResult
from twse_factor_lab.runtime.shadow import ShadowRuntime

ROOT = Path(__file__).resolve().parents[1]


def _rows(dates: list[str], tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["1101", "1102"]
    rows = []
    for date in dates:
        for index, ticker in enumerate(tickers):
            price = 10.0 + index + len(rows) / 100
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price + 0.5,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


class FakeClient:
    def __init__(self, frame: pd.DataFrame, failed: list[str] | None = None):
        self.frame = frame
        self.failed = failed or []
        self.calls: list[dict[str, object]] = []

    def download_ohlcv(self, **kwargs: object) -> OhlcvDownloadResult:
        self.calls.append(kwargs)
        start, end = pd.Timestamp(kwargs["start"]), pd.Timestamp(kwargs["end"])
        data = self.frame.loc[
            (pd.to_datetime(self.frame.date) >= start)
            & (pd.to_datetime(self.frame.date) <= end)
        ].copy()
        return OhlcvDownloadResult(data, self.failed)


def _root(tmp_path: Path) -> Path:
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    _rows(["2025-01-02"]).to_parquet(processed / "ohlcv.parquet", index=False)
    pd.DataFrame(
        {"ticker": ["1101", "1102"], "listed_date": ["2020-01-01"] * 2}
    ).to_parquet(processed / "universe.parquet", index=False)
    return tmp_path


def _updater(
    tmp_path: Path, frame: pd.DataFrame, **kwargs: object
) -> IncrementalCanonicalOHLCVUpdater:
    return IncrementalCanonicalOHLCVUpdater(
        _root(tmp_path),
        client=FakeClient(frame),
        calendar=pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"]),
        now=datetime(2025, 1, 7, 15, tzinfo=UTC),
        **kwargs,
    )


def test_d1_downloads_only_missing_sessions(tmp_path):
    updater = _updater(tmp_path, _rows(["2025-01-03"]))
    result = updater.update(end="2025-01-03", mode="BACKFILL")
    assert result.status == "PASS"
    assert updater.client.calls[0]["start"] == "2025-01-03"


def test_d2_duplicate_update_is_idempotent(tmp_path):
    updater = _updater(tmp_path, _rows(["2025-01-03"]))
    updater.update(end="2025-01-03", mode="BACKFILL")
    path = updater.ohlcv_path
    first = path.read_bytes()
    updater.update(end="2025-01-03", mode="BACKFILL")
    assert path.read_bytes() == first


def test_d3_atomic_write_leaves_old_store_on_validation_failure(tmp_path):
    updater = _updater(tmp_path, _rows(["2025-01-03"]).assign(close=-1))
    before = updater.ohlcv_path.read_bytes()
    assert updater.update(end="2025-01-03", mode="BACKFILL").status == "SAFE_HALT"
    assert updater.ohlcv_path.read_bytes() == before


def test_d4_partial_data_cannot_commit(tmp_path):
    frame = _rows(["2025-01-03"], ["1101"])
    updater = _updater(tmp_path, frame)
    result = updater.update(end="2025-01-03", mode="BACKFILL")
    assert result.status == "PARTIAL_INGESTION"


def test_d5_stale_data_safe_halt(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=3)
    close = pd.DataFrame(10.0, index=dates, columns=["1101"])
    volume = pd.DataFrame(1000.0, index=dates, columns=["1101"])
    universe = pd.DataFrame({"ticker": ["1101"], "listed_date": ["2019-01-01"]})
    runtime = ShadowRuntime(ROOT, tmp_path / "test-d5")
    assert runtime.health(close, volume, universe, dates[-1], True)["status"] == "FAIL"


def test_d6_weekend_does_not_create_session():
    assert (
        resolve_latest_completed_session(
            ["2025-01-03", "2025-01-06"], now="2025-01-04T15:00:00+08:00"
        ).date()
        == datetime(2025, 1, 3).date()
    )


def test_d7_holiday_calendar_is_authoritative():
    result = resolve_latest_completed_session(
        ["2025-01-02", "2025-01-06"], now="2025-01-05T15:00:00+08:00"
    )
    assert result.date() == datetime(2025, 1, 2).date()


def test_d8_current_session_before_close_is_not_ready():
    result = resolve_latest_completed_session(
        ["2025-01-06", "2025-01-07"], now="2025-01-07T12:00:00+08:00"
    )
    assert result.date() == datetime(2025, 1, 6).date()


def test_d9_backfill_is_not_forward(tmp_path):
    updater = _updater(tmp_path, _rows(["2025-01-03"]))
    assert updater.update(end="2025-01-03", mode="BACKFILL").mode == "BACKFILL"
    assert not (updater.out / "runtime_state.json").exists()


def test_d10_replay_flag_is_not_forward():
    assert ShadowRuntime._initial_state()["forward_sessions"] == []


def test_d11_forward_timestamp_is_fixed(tmp_path):
    updater = _updater(
        tmp_path,
        _rows(["2025-01-03"]),
        forward_evidence_start_timestamp="2025-01-01T00:00:00+00:00",
    )
    first = updater.ensure_forward_boundary()
    assert updater.ensure_forward_boundary() == first


def test_d12_same_forward_date_has_one_provenance_row(tmp_path):
    updater = _updater(tmp_path, _rows(["2025-01-03"]))
    updater.now = datetime(2025, 1, 3, 15, tzinfo=UTC)
    updater.update(end="2025-01-03", mode="FORWARD")
    updater.update(end="2025-01-03", mode="FORWARD")
    assert len(pd.read_parquet(updater.out / "market_data_ingestion_log.parquet")) == 1


def test_d13_revision_is_recorded(tmp_path):
    updater = _updater(tmp_path, _rows(["2025-01-03"]))
    updater.update(end="2025-01-03", mode="BACKFILL")
    revised = _rows(["2025-01-03"])
    revised.loc[0, "close"] = 10.0
    updater.client.frame = revised
    updater.update(start="2025-01-03", end="2025-01-03", mode="BACKFILL")
    assert (updater.out / "data_revision_log.json").exists()


def test_d14_forward_revision_requires_review(tmp_path):
    updater = _updater(tmp_path, _rows(["2025-01-03"]))
    updater.now = datetime(2025, 1, 3, 15, tzinfo=UTC)
    updater.update(end="2025-01-03", mode="FORWARD")
    state = {"forward_sessions": ["2025-01-03"]}
    (updater.out / "runtime_state.json").write_text(json.dumps(state), encoding="utf-8")
    revised = _rows(["2025-01-03"])
    revised.loc[0, "close"] = 10.0
    updater.client.frame = revised
    result = updater.update(start="2025-01-03", end="2025-01-03", mode="FORWARD")
    assert "FORWARD_DATA_REVISION_DETECTED" in result.issues


def test_d15_fresh_oos_report_is_written(tmp_path):
    updater = _updater(tmp_path, _rows(["2025-01-03"]))
    updater.write_fresh_oos_reconciliation()
    assert (updater.out / "fresh_oos_data_reconciliation.json").exists()


def test_d16_candidate_fingerprint_is_not_owned_by_ingestion():
    assert ShadowRuntime._initial_state()["candidate_fingerprint"]


def test_d17_runtime_contract_remains_immutable(tmp_path):
    runtime = ShadowRuntime(ROOT, tmp_path / "runtime")
    contract = runtime.prepare()
    assert (
        contract["candidate_fingerprint"]
        == "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"
    )


def test_d18_strategy_not_present_in_updater():
    assert not hasattr(IncrementalCanonicalOHLCVUpdater, "rank")


def test_d19_execution_semantics_are_runtime_owned(tmp_path):
    assert (
        "T+1"
        in ShadowRuntime(ROOT, tmp_path / "test-d19").contract()["execution_semantics"]
    )


def test_d20_historical_fresh_freeze_files_exist():
    assert (
        ROOT / "data/research/fresh-oos-validation-v1/fresh_oos_contract.json"
    ).exists()
    assert (
        ROOT
        / "data/research/engine-parity-fix-final-validation-v3"
        / "final_validation_v3_contract.json"
    ).exists()

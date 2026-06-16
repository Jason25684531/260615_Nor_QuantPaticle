import pandas as pd
import pytest

from run_data_pipeline import (
    build_quality_report,
    download_ohlcv_with_retries,
    ohlcv_settings_from_config,
    select_ohlcv_tickers,
)
from twse_factor_lab.data.yfinance_client import OhlcvDownloadResult


def test_ticker_limit_is_read_from_nested_config_not_hardcoded_to_30():
    settings = ohlcv_settings_from_config(
        {
            "ohlcv_ticker_limit": 30,
            "ohlcv": {
                "ticker_limit": 100,
                "batch_size": 20,
                "retry": 3,
                "sleep_seconds": 0,
                "fail_fast": False,
            },
        }
    )

    assert settings.ticker_limit == 100
    assert settings.batch_size == 20


def test_select_ohlcv_tickers_uses_configurable_limit():
    universe = pd.DataFrame({"ticker": [f"{idx:04d}" for idx in range(1, 151)]})

    tickers = select_ohlcv_tickers(universe, ticker_limit=100)

    assert len(tickers) == 100
    assert tickers[0] == "0001"
    assert tickers[-1] == "0100"


def test_failed_tickers_are_logged_and_pipeline_continues_when_not_fail_fast():
    calls: list[tuple[str, ...]] = []

    class FakeClient:
        def download_ohlcv(self, tickers, start, end):
            batch = tuple(tickers)
            calls.append(batch)
            rows = [
                {
                    "date": pd.Timestamp("2024-01-02"),
                    "ticker": ticker,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 100,
                }
                for ticker in batch
                if ticker != "0002"
            ]
            return OhlcvDownloadResult(
                pd.DataFrame(rows), ["0002"] if "0002" in batch else []
            )

    result = download_ohlcv_with_retries(
        client=FakeClient(),
        tickers=["0001", "0002", "0003"],
        start="2024-01-01",
        end="2024-01-31",
        batch_size=2,
        retry=1,
        sleep_seconds=0,
        fail_fast=False,
    )

    assert calls == [("0001", "0002"), ("0002",), ("0003",)]
    assert result.failed_tickers == ["0002"]
    assert sorted(result.data["ticker"].unique().tolist()) == ["0001", "0003"]


def test_download_fail_fast_raises_for_failed_tickers():
    class FakeClient:
        def download_ohlcv(self, tickers, start, end):
            return OhlcvDownloadResult(pd.DataFrame(), list(tickers))

    with pytest.raises(RuntimeError, match="OHLCV download failed"):
        download_ohlcv_with_retries(
            client=FakeClient(),
            tickers=["0001"],
            start="2024-01-01",
            end="2024-01-31",
            batch_size=1,
            retry=1,
            sleep_seconds=0,
            fail_fast=True,
        )


def test_data_quality_report_includes_week3_coverage_metrics():
    universe = pd.DataFrame({"ticker": ["0001", "0002", "0003", "0004"]})
    valuation = pd.DataFrame({"ticker": ["0001"], "pe": [10.0]})
    ohlcv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker": ["0001", "0003"],
            "close": [1.0, 2.0],
        }
    )

    report = build_quality_report(
        universe=universe,
        valuation=valuation,
        ohlcv=ohlcv,
        failed_tickers=["0002"],
        configured_ticker_limit=3,
        ohlcv_requested_tickers=3,
    )

    assert "universe_total_tickers: 4" in report
    assert "ohlcv_requested_tickers: 3" in report
    assert "ohlcv_successful_tickers: 2" in report
    assert "ohlcv_failed_tickers: 1" in report
    assert "ohlcv_coverage_ratio: 0.5000" in report
    assert "configured_ticker_limit: 3" in report
    assert "actual_ohlcv_ticker_count: 2" in report
    assert "failed_yfinance_tickers: 0002" in report

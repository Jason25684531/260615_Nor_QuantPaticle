import json

import pandas as pd
import pytest

import run_data_pipeline as pipeline
from twse_factor_lab.data.yfinance_client import OhlcvDownloadResult


def test_pipeline_writes_research_universe_coverage_and_provenance(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config" / "strategy.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
data:
  start_date: "2024-01-01"
  end_date: "2024-01-05"
  ohlcv:
    ticker_limit: 1
    batch_size: 1
    retry: 0
    sleep_seconds: 0
    fail_fast: true
twse:
  base_url: "https://example.com"
paths:
  universe: "data/universe.parquet"
  valuation: "data/valuation.parquet"
  ohlcv: "data/ohlcv.parquet"
  research_universe: "data/research_universe.parquet"
  universe_coverage: "data/universe_coverage.parquet"
  data_quality_report: "reports/quality.md"
  manifest: "data/manifest.json"
universe:
  membership: current_listed_only
  liquidity:
    enabled: true
    window: 2
    measure: median
    minimum_traded_value: 50
research_quality:
  minimum_universe_coverage: 0.9
research:
  in_sample: {start: "2024-01-01", end: "2024-01-02"}
  out_of_sample: {start: "2024-01-03", end: "2024-01-05"}
""".strip(),
        encoding="utf-8",
    )

    class FakeTwseClient:
        def __init__(self, **_):
            pass

        def fetch_dataframe(self, endpoint):
            if endpoint == "listed_companies":
                return pd.DataFrame(
                    {
                        "Code": ["1101", "1102"],
                        "ListedDate": ["2024-01-01", "2024-01-01"],
                    }
                )
            return pd.DataFrame({"Code": ["1101"], "PE": [10.0]})

    class FakeYFinanceClient:
        def download_ohlcv(self, **_):
            return OhlcvDownloadResult(
                pd.DataFrame(
                    {
                        "date": pd.date_range("2024-01-01", periods=3),
                        "ticker": ["1101"] * 3,
                        "open": [1.0] * 3,
                        "high": [2.0] * 3,
                        "low": [1.0] * 3,
                        "close": [1.0] * 3,
                        "volume": [100.0] * 3,
                    }
                ),
                [],
            )

    monkeypatch.setattr(pipeline, "TWSEClient", FakeTwseClient)
    monkeypatch.setattr(pipeline, "YFinanceClient", FakeYFinanceClient)

    with pytest.warns(UserWarning, match="PARTIAL"):
        outputs = pipeline.run_pipeline(config_path)

    research = pd.read_parquet(outputs["research_universe"])
    coverage = pd.read_parquet(outputs["universe_coverage"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    report = outputs["data_quality_report"].read_text(encoding="utf-8")

    assert not research.query("ticker == '1102'")["has_ohlcv"].any()
    assert coverage.iloc[-1]["coverage_ratio"] == 0.5
    assert "universe_scope: PARTIAL" in report
    assert "survivorship_disclosure" in report
    ohlcv_entry = next(
        item for item in manifest["artifacts"] if item["artifact_name"] == "ohlcv"
    )
    assert ohlcv_entry["price_adjustment"] == "auto_adjusted"
    assert ohlcv_entry["schema_version"] == "ohlcv-v2"

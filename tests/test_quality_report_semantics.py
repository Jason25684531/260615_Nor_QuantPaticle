from pathlib import Path

import pandas as pd

from run_data_pipeline import build_quality_report
from run_factor_pipeline import (
    build_factor_quality_report,
    run_factor_pipeline,
)
from twse_factor_lab.data.normalizer import normalize_universe
from twse_factor_lab.data.parquet_store import ParquetStore


def test_normalize_universe_defaults_market_to_twse_when_missing():
    raw = pd.DataFrame(
        {
            "Code": ["2330"],
            "Name": ["TSMC"],
            "Industry": ["Semiconductor"],
            "ListedDate": ["1994-09-05"],
        }
    )

    frame = normalize_universe(raw)

    assert frame.loc[0, "market"] == "TWSE"


def test_build_quality_report_includes_source_context_and_limitations():
    universe = pd.DataFrame(
        {
            "ticker": ["2330", "2317"],
            "company_name": ["A", "B"],
            "industry": ["Semi", "Elec"],
            "market": ["TWSE", "TWSE"],
            "listed_date": pd.to_datetime(["1994-09-05", "1991-02-20"]),
        }
    )
    valuation = pd.DataFrame(
        {
            "date": [pd.NaT],
            "ticker": ["2330"],
            "pe": [20.0],
            "pb": [3.0],
            "dividend_yield": [1.5],
        }
    )
    ohlcv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["2330", "2317"],
            "close": [1.5, 2.0],
        }
    )

    report = build_quality_report(
        universe=universe,
        valuation=valuation,
        ohlcv=ohlcv,
        failed_tickers=["9999"],
        configured_start_date="2018-01-01",
        configured_end_date="2025-12-31",
        ohlcv_ticker_subset_size=30,
        ohlcv_source="yfinance fallback",
        valuation_source="TWSE latest snapshot valuation endpoint",
    )

    assert "Configured date range: 2018-01-01 to 2025-12-31" in report
    assert "Actual OHLCV date range: 2024-01-02 to 2024-01-03" in report
    assert "OHLCV ticker subset size: 30" in report
    assert "Universe total count: 2" in report
    assert "market source" in report.lower()
    assert "valuation.date is empty" in report
    assert "survivorship bias" in report.lower()
    assert "9999" in report


def test_build_factor_quality_report_separates_composite_semantics():
    matrices = {
        "close": pd.DataFrame({"1101": [10.0]}, index=pd.to_datetime(["2024-01-02"])),
        "high": pd.DataFrame({"1101": [11.0]}, index=pd.to_datetime(["2024-01-02"])),
        "low": pd.DataFrame({"1101": [9.0]}, index=pd.to_datetime(["2024-01-02"])),
        "volume": pd.DataFrame({"1101": [100.0]}, index=pd.to_datetime(["2024-01-02"])),
    }
    price_volume_factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["1101"],
            "momentum_60d": [0.1],
            "low_volatility_20d": [0.2],
            "volume_ratio_5d_60d": [1.1],
        }
    )
    valuation_factors = pd.DataFrame(
        {
            "date": [pd.NaT],
            "as_of_date": pd.to_datetime(["2026-06-16"]),
            "ticker": ["1101"],
            "pb_inverse": [1.0],
            "pe_inverse": [0.5],
            "dividend_yield": [3.0],
            "is_snapshot": [True],
        }
    )
    composite_factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2026-06-16"]),
            "ticker": ["1101", "1101"],
            "composite_score": [0.8, 0.9],
            "composite_type": [
                "historical_price_volume",
                "latest_snapshot_mixed",
            ],
            "is_snapshot_component_used": [False, True],
        }
    )

    report = build_factor_quality_report(
        input_paths={"universe": Path("u"), "valuation": Path("v"), "ohlcv": Path("o")},
        universe=pd.DataFrame({"ticker": ["1101"]}),
        matrices=matrices,
        price_volume_factors=price_volume_factors,
        valuation_factors=valuation_factors,
        composite_factors=composite_factors,
        alignment_report={"is_aligned": True, "missing_ratio": 0.0},
        formatter_report={"is_ready": True},
        valuation_limitation="valuation inputs are latest snapshot data",
        low_volatility_method="atr_20d_over_close",
    )

    assert "historical_price_volume_composite" in report
    assert "latest_snapshot_mixed_composite" in report
    assert "historical_backtest_ready: true" in report
    assert "historical_backtest_ready: false" in report
    assert "snapshot_only_not_historical_ready" in report
    assert "latest_snapshot_mixed cannot be used for historical backtest" in report
    assert "low_volatility_method: atr_20d_over_close" in report


def test_run_factor_pipeline_reports_composite_breakdown_and_readiness(tmp_path):
    config_path = tmp_path / "config" / "strategy.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
data:
  start_date: "2018-01-01"
  end_date: "2025-12-31"
  ohlcv_ticker_limit: 30
twse:
  base_url: "https://example.com"
paths:
  universe: "data/processed/universe.parquet"
  valuation: "data/processed/valuation.parquet"
  ohlcv: "data/processed/ohlcv.parquet"
  close_matrix: "data/processed/close_matrix.parquet"
  high_matrix: "data/processed/high_matrix.parquet"
  low_matrix: "data/processed/low_matrix.parquet"
  volume_matrix: "data/processed/volume_matrix.parquet"
  factors_price_volume: "data/processed/factors_price_volume.parquet"
  factors_valuation_snapshot: "data/processed/factors_valuation_snapshot.parquet"
  factors_composite: "data/processed/factors_composite.parquet"
  factor_quality_report: "reports/factor_quality_summary.md"
  manifest: "data/processed/_manifest.json"
week2:
  manifest_schema_version: "1.0.0"
""".strip(),
        encoding="utf-8",
    )

    root = config_path.parent.parent
    store = ParquetStore()
    store.save(
        pd.DataFrame({"ticker": ["1101", "1102"], "market": ["TWSE", "TWSE"]}),
        root / "data/processed/universe.parquet",
    )
    store.save(
        pd.DataFrame(
            {
                "date": [pd.NaT, pd.NaT],
                "ticker": ["1101", "1102"],
                "pe": [10.0, 20.0],
                "pb": [1.0, 2.0],
                "dividend_yield": [3.0, 4.0],
            }
        ),
        root / "data/processed/valuation.parquet",
    )
    store.save(
        pd.DataFrame(
            {
                "date": list(pd.date_range("2024-01-01", periods=65, freq="D")) * 2,
                "ticker": ["1101"] * 65 + ["1102"] * 65,
                "open": list(range(1, 66)) + list(range(11, 76)),
                "high": list(range(2, 67)) + list(range(12, 77)),
                "low": list(range(1, 66)) + list(range(11, 76)),
                "close": list(range(1, 66)) + list(range(11, 76)),
                "volume": list(range(100, 165)) + list(range(200, 265)),
            }
        ),
        root / "data/processed/ohlcv.parquet",
    )

    outputs = run_factor_pipeline(config_path)
    report = Path(outputs["factor_quality_report"]).read_text(encoding="utf-8")

    assert "factors_composite_total_rows" in report
    assert "historical_price_volume_rows" in report
    assert "latest_snapshot_mixed_rows" in report
    assert "Alphalens readiness by factor type" in report
    assert "latest_snapshot_mixed_composite" in report

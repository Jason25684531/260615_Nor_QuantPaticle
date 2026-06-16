from pathlib import Path

import pandas as pd

from run_factor_pipeline import load_config, run_factor_pipeline
from twse_factor_lab.data.parquet_store import ParquetStore


def test_run_factor_pipeline_reads_week1_outputs_and_writes_week2_artifacts(tmp_path):
    config_path = tmp_path / "config" / "strategy.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
data:
  start_date: "2018-01-01"
  end_date: "2025-12-31"
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
        pd.DataFrame({"ticker": ["1101", "1102"]}),
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
    report_path = Path(outputs["factor_quality_report"])
    manifest_path = Path(outputs["manifest"])

    assert report_path.exists()
    assert manifest_path.exists()
    assert "valuation inputs are latest snapshot data" in report_path.read_text(
        encoding="utf-8"
    )


def test_load_config_keeps_week2_paths_available(tmp_path):
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        (
            "paths:\n"
            "  manifest: data/processed/_manifest.json\n"
            "week2:\n"
            "  manifest_schema_version: 1.0.0\n"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["paths"]["manifest"] == "data/processed/_manifest.json"

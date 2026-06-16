from pathlib import Path

import pandas as pd

from run_factor_analysis import load_config, run_factor_analysis
from twse_factor_lab.data.manifest import write_manifest
from twse_factor_lab.data.parquet_store import ParquetStore


def _build_factor_rows(dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, date in enumerate(dates):
        for rank, ticker in enumerate(tickers, start=1):
            signal = float(len(tickers) - rank + 1 + offset)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "momentum_60d": signal,
                    "low_volatility_20d": float(rank),
                    "volume_ratio_5d_60d": signal / 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_run_factor_analysis_writes_outputs_report_and_manifest(tmp_path):
    config_path = tmp_path / "config" / "strategy.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
paths:
  close_matrix: "data/processed/close_matrix.parquet"
  factors_price_volume: "data/processed/factors_price_volume.parquet"
  factors_composite: "data/processed/factors_composite.parquet"
  manifest: "data/processed/_manifest.json"
  factor_forward_returns: "data/processed/factor_forward_returns.parquet"
  factor_ic_summary: "data/processed/factor_ic_summary.parquet"
  factor_quantile_returns: "data/processed/factor_quantile_returns.parquet"
  factor_turnover: "data/processed/factor_turnover.parquet"
  factor_monotonicity: "data/processed/factor_monotonicity.parquet"
  factor_analysis_report: "reports/factor_analysis_report.md"
analysis:
  horizons: [1, 5, 10, 20]
  quantiles: 5
  factors:
    historical:
      - momentum_60d
      - low_volatility_20d
      - volume_ratio_5d_60d
    optional_composites:
      - historical_price_volume
    excluded_snapshot:
      - pb_inverse
      - pe_inverse
      - dividend_yield
      - latest_snapshot_mixed
week2:
  manifest_schema_version: "1.0.0"
""".strip(),
        encoding="utf-8",
    )

    root = config_path.parent.parent
    store = ParquetStore()
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    tickers = ["1101", "1102", "1103", "1104", "1105"]

    close_matrix = pd.DataFrame(
        {
            ticker: [100.0 + (idx * 10.0) + day * (idx + 1) for day in range(25)]
            for idx, ticker in enumerate(tickers)
        },
        index=dates,
    )
    factors_price_volume = _build_factor_rows(dates, tickers)
    historical_composite = factors_price_volume[["date", "ticker"]].copy()
    historical_composite["composite_score"] = factors_price_volume["momentum_60d"]
    historical_composite["composite_type"] = "historical_price_volume"
    historical_composite["is_snapshot_component_used"] = False
    snapshot_rows = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-16")] * len(tickers),
            "ticker": tickers,
            "composite_score": [0.5] * len(tickers),
            "composite_type": ["latest_snapshot_mixed"] * len(tickers),
            "is_snapshot_component_used": [True] * len(tickers),
        }
    )
    factors_composite = pd.concat(
        [historical_composite, snapshot_rows], ignore_index=True
    )

    store.save(
        close_matrix,
        root / "data/processed/close_matrix.parquet",
        include_index=True,
    )
    store.save(
        factors_price_volume, root / "data/processed/factors_price_volume.parquet"
    )
    store.save(factors_composite, root / "data/processed/factors_composite.parquet")
    write_manifest(
        [
            {
                "artifact_name": "factors_composite",
                "path": "seed",
                "rows": 1,
                "columns": 1,
            }
        ],
        root / "data/processed/_manifest.json",
    )

    outputs = run_factor_analysis(config_path)
    report = Path(outputs["factor_analysis_report"]).read_text(encoding="utf-8")
    manifest = Path(outputs["manifest"]).read_text(encoding="utf-8")
    ic_summary = store.load(outputs["factor_ic_summary"])

    assert Path(outputs["factor_analysis_report"]).exists()
    assert "snapshot_only_not_historical_ready" in report
    assert "latest_snapshot_mixed" in report
    assert "Generated Artifacts" in report
    assert "factor_forward_returns" in manifest
    assert "factor_ic_summary" in manifest
    assert "pb_inverse" not in ic_summary["factor"].tolist()


def test_load_config_preserves_analysis_defaults(tmp_path):
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        (
            "paths:\n"
            "  factor_analysis_report: reports/factor_analysis_report.md\n"
            "analysis:\n"
            "  quantiles: 5\n"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["analysis"]["quantiles"] == 5

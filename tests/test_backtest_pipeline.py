from pathlib import Path

import pandas as pd
import pytest

from run_backtest import check_backtest_readiness, run_backtest
from twse_factor_lab.data.manifest import write_manifest
from twse_factor_lab.data.parquet_store import ParquetStore


def _write_config(config_path: Path, min_ticker_count: int = 2) -> None:
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        f"""
paths:
  close_matrix: "data/processed/close_matrix.parquet"
  factors_composite: "data/processed/factors_composite.parquet"
  factor_ic_summary: "data/processed/factor_ic_summary.parquet"
  factor_quantile_returns: "data/processed/factor_quantile_returns.parquet"
  factor_turnover: "data/processed/factor_turnover.parquet"
  factor_monotonicity: "data/processed/factor_monotonicity.parquet"
  factor_scoreboard: "data/processed/factor_scoreboard.parquet"
  selected_factor_scores: "data/processed/selected_factor_scores.parquet"
  topn_positions: "data/processed/topn_positions.parquet"
  portfolio_weights: "data/processed/portfolio_weights.parquet"
  backtest_results: "data/processed/backtest_results.parquet"
  backtest_metrics: "data/processed/backtest_metrics.parquet"
  composite_factor_report: "reports/composite_factor_report.md"
  backtest_report: "reports/backtest_report.md"
  manifest: "data/processed/_manifest.json"
backtest:
  enabled: true
  min_ticker_count: {min_ticker_count}
  factor_name: historical_price_volume
  top_n: 2
  execution_lag_days: 1
  initial_cash: 1000000
  fees:
    buy_fee_rate: 0.001425
    sell_fee_rate: 0.001425
    transaction_tax_rate: 0.003
    slippage_rate: 0.001
  vectorbt:
    use_vectorbt: false
    allow_fallback: true
week2:
  manifest_schema_version: "1.0.0"
""".strip(),
        encoding="utf-8",
    )


def _seed_inputs(root: Path, ticker_count: int = 3) -> None:
    store = ParquetStore()
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    tickers = [f"000{idx}" for idx in range(1, ticker_count + 1)]
    close = pd.DataFrame(
        {
            ticker: [100.0 + day + idx for day in range(len(dates))]
            for idx, ticker in enumerate(tickers)
        },
        index=dates,
    )
    composite_rows = []
    for offset, date in enumerate(dates[:-1]):
        for idx, ticker in enumerate(tickers):
            composite_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "composite_score": float(ticker_count - idx + offset),
                    "composite_type": "historical_price_volume",
                    "is_snapshot_component_used": False,
                }
            )
    composite_rows.append(
        {
            "date": dates[0],
            "ticker": "9999",
            "composite_score": 999.0,
            "composite_type": "latest_snapshot_mixed",
            "is_snapshot_component_used": True,
        }
    )
    store.save(close, root / "data/processed/close_matrix.parquet", include_index=True)
    store.save(
        pd.DataFrame(composite_rows),
        root / "data/processed/factors_composite.parquet",
    )
    store.save(
        pd.DataFrame(
            {
                "factor": ["historical_price_volume"],
                "horizon": [20],
                "ic_mean": [0.03],
                "ic_std": [0.2],
                "ir": [0.15],
            }
        ),
        root / "data/processed/factor_ic_summary.parquet",
    )
    store.save(
        pd.DataFrame(
            {
                "factor": ["historical_price_volume", "historical_price_volume"],
                "horizon": [20, 20],
                "quantile": [1, 5],
                "mean_return": [0.01, 0.04],
            }
        ),
        root / "data/processed/factor_quantile_returns.parquet",
    )
    store.save(
        pd.DataFrame(
            {
                "factor": ["historical_price_volume"],
                "average_best_bucket_turnover": [0.2],
            }
        ),
        root / "data/processed/factor_turnover.parquet",
    )
    store.save(
        pd.DataFrame(
            {
                "factor": ["historical_price_volume"],
                "horizon": [20],
                "adjacent_agreement_ratio": [0.75],
                "monotonicity_pass": [False],
            }
        ),
        root / "data/processed/factor_monotonicity.parquet",
    )
    write_manifest([], root / "data/processed/_manifest.json")


def test_readiness_gate_fails_when_ticker_count_is_below_minimum():
    close = pd.DataFrame({"0001": [1.0, 2.0]})

    with pytest.raises(RuntimeError, match="not suitable for baseline Top N backtest"):
        check_backtest_readiness(close, min_ticker_count=2)


def test_run_backtest_pipeline_writes_artifacts_reports_and_manifest(tmp_path):
    config_path = tmp_path / "config" / "strategy.yaml"
    _write_config(config_path)
    _seed_inputs(tmp_path)

    outputs = run_backtest(config_path)

    assert Path(outputs["factor_scoreboard"]).exists()
    assert Path(outputs["selected_factor_scores"]).exists()
    assert Path(outputs["topn_positions"]).exists()
    assert Path(outputs["portfolio_weights"]).exists()
    assert Path(outputs["backtest_results"]).exists()
    assert Path(outputs["backtest_metrics"]).exists()
    report = Path(outputs["backtest_report"]).read_text(encoding="utf-8")
    manifest = Path(outputs["manifest"]).read_text(encoding="utf-8")
    positions = pd.read_parquet(outputs["topn_positions"])
    weights = pd.read_parquet(outputs["portfolio_weights"])

    assert "research backtest only" in report
    assert "historical_price_volume" in report
    assert "latest_snapshot_mixed" not in positions["ticker"].tolist()
    assert (weights["execution_date"] > weights["date"]).all()
    assert "backtest_results" in manifest
    assert "backtest_metrics" in manifest

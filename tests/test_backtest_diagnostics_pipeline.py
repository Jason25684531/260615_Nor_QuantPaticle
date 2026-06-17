"""Integration test for the backtest diagnostics pipeline using synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _make_close(n_days: int = 120, n_tickers: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    data = {}
    for i in range(n_tickers):
        ticker = f"T{i:03d}.TW"
        prices = (
            100.0 * (1 + np.random.default_rng(i).normal(0, 0.01, n_days)).cumprod()
        )
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_factors(close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in close.index[::5]:
        for ticker in close.columns:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "composite_score": float(hash(ticker + str(date)) % 100),
                    "composite_type": "historical_price_volume",
                    "is_snapshot_component_used": False,
                }
            )
    return pd.DataFrame(rows)


def _make_portfolio_weights(close: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    tickers = list(close.columns[:top_n])
    rows = []
    for date in close.index[1::5]:
        for ticker in tickers:
            rows.append(
                {
                    "date": close.index[0],
                    "execution_date": date,
                    "ticker": ticker,
                    "target_weight": 1.0 / top_n,
                    "execution_lag_days": 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Set up a minimal synthetic project in a temp directory."""
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)

    close = _make_close()
    factors = _make_factors(close)
    weights = _make_portfolio_weights(close)

    close.to_parquet(tmp_path / "data" / "processed" / "close_matrix.parquet")
    factors.to_parquet(tmp_path / "data" / "processed" / "factors_composite.parquet")
    weights.to_parquet(tmp_path / "data" / "processed" / "portfolio_weights.parquet")

    # Write minimal manifest
    (tmp_path / "data" / "processed" / "_manifest.json").write_text(
        json.dumps({"artifacts": []}), encoding="utf-8"
    )

    config = {
        "backtest": {
            "factor_name": "historical_price_volume",
            "top_n": 10,
            "top_n_grid": [10, 15],
            "rebalance_frequency": "weekly",
            "rebalance_frequency_grid": ["weekly", "monthly"],
            "execution_lag_days": 1,
            "initial_cash": 1_000_000,
            "fees": {
                "buy_fee_rate": 0.001425,
                "sell_fee_rate": 0.001425,
                "transaction_tax_rate": 0.003,
                "slippage_rate": 0.001,
            },
            "rules": {"hold_until_drop": False, "drop_rank_buffer": 30},
            "cost_sensitivity": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "no_cost",
                        "buy_fee_rate": 0.0,
                        "sell_fee_rate": 0.0,
                        "transaction_tax_rate": 0.0,
                        "slippage_rate": 0.0,
                    },
                    {
                        "name": "base_cost",
                        "buy_fee_rate": 0.001425,
                        "sell_fee_rate": 0.001425,
                        "transaction_tax_rate": 0.003,
                        "slippage_rate": 0.001,
                    },
                ],
            },
        },
        "paths": {
            "close_matrix": "data/processed/close_matrix.parquet",
            "factors_composite": "data/processed/factors_composite.parquet",
            "portfolio_weights": "data/processed/portfolio_weights.parquet",
            "rebalance_calendar": "data/processed/rebalance_calendar.parquet",
            "backtest_scenarios": "data/processed/backtest_scenarios.parquet",
            "topn_sensitivity": "data/processed/topn_sensitivity.parquet",
            "rebalance_sensitivity": "data/processed/rebalance_sensitivity.parquet",
            "backtest_turnover_diagnostics": (
                "data/processed/backtest_turnover_diagnostics.parquet"
            ),
            "backtest_engine_comparison": (
                "data/processed/backtest_engine_comparison.parquet"
            ),
            "backtest_realism_report": "reports/backtest_realism_report.md",
            "manifest": "data/processed/_manifest.json",
        },
        "week2": {"manifest_schema_version": "1.0.0"},
    }
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return tmp_path


class TestDiagnosticsPipeline:
    def test_all_artifacts_created(self, tmp_project: Path) -> None:
        from run_backtest_diagnostics import run_diagnostics

        config_path = tmp_project / "strategy.yaml"
        run_diagnostics(config_path)

        expected = [
            "data/processed/rebalance_calendar.parquet",
            "data/processed/backtest_scenarios.parquet",
            "data/processed/topn_sensitivity.parquet",
            "data/processed/rebalance_sensitivity.parquet",
            "data/processed/backtest_turnover_diagnostics.parquet",
            "data/processed/backtest_engine_comparison.parquet",
            "reports/backtest_realism_report.md",
        ]
        for rel_path in expected:
            full = tmp_project / rel_path
            assert full.exists(), f"Missing artifact: {rel_path}"

    def test_report_contains_sections(self, tmp_project: Path) -> None:
        from run_backtest_diagnostics import run_diagnostics

        config_path = tmp_project / "strategy.yaml"
        run_diagnostics(config_path)

        report = (tmp_project / "reports" / "backtest_realism_report.md").read_text(
            encoding="utf-8"
        )
        for section in [
            "## 1. Run Metadata",
            "## 2. Purpose",
            "## 3. Baseline Strategy Recap",
            "## 4. Universe Coverage",
            "## 5. Selected Factor",
            "## 6. Rebalance Frequency Results",
            "## 7. Cost Sensitivity Results",
            "## 8. Top N Sensitivity Results",
            "## 9. Turnover Diagnostics",
            "## 10. Buffer Rule Impact",
            "## 11. Engine Status / Cross-Check",
            "## 12. Interpretation",
            "## 13. Limitations",
            "## 14. Recommended Next Step",
            "## 15. Generated Artifacts",
        ]:
            assert section in report, f"Missing section: {section}"

    def test_report_contains_disclaimer(self, tmp_project: Path) -> None:
        from run_backtest_diagnostics import run_diagnostics

        config_path = tmp_project / "strategy.yaml"
        run_diagnostics(config_path)

        report = (tmp_project / "reports" / "backtest_realism_report.md").read_text(
            encoding="utf-8"
        )
        assert "research backtest only" in report
        assert "not investment advice" in report

    def test_snapshot_factor_excluded(self, tmp_project: Path) -> None:
        """Pipeline raises ValueError for snapshot valuation factor."""
        import yaml as _yaml

        config_path = tmp_project / "strategy.yaml"
        cfg = _yaml.safe_load(config_path.read_text())
        cfg["backtest"]["factor_name"] = "pb_inverse"
        config_path.write_text(_yaml.dump(cfg), encoding="utf-8")

        import pytest as _pytest

        from run_backtest_diagnostics import run_diagnostics

        with _pytest.raises(ValueError):
            run_diagnostics(config_path)

    def test_no_network_calls(
        self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure pipeline does not call yfinance or TWSE API."""
        import unittest.mock as mock

        config_path = tmp_project / "strategy.yaml"

        blocked = []

        def _block(*a: object, **kw: object) -> None:
            blocked.append(a)
            raise RuntimeError("Network call detected in test")

        with mock.patch("yfinance.download", _block):
            from run_backtest_diagnostics import run_diagnostics

            run_diagnostics(config_path)  # should complete without network

        assert not blocked

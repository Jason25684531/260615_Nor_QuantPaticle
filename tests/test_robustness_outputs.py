from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from twse_factor_lab.backtest.robustness import compute_metrics, fit_hac_regression

ROOT = Path(__file__).parents[1]
PROCESSED = ROOT / "data" / "processed"


def test_declared_grids_and_cost_math() -> None:
    config = yaml.safe_load((ROOT / "config" / "strategy.yaml").read_text())
    assert config["robustness"]["top_n"]["values"] == [10, 20, 30]
    cost = pd.read_parquet(PROCESSED / "cost_stress.parquet")
    assert set(cost["cost_multiplier"]) == {0.0, 0.5, 1.0, 2.0}
    for _, group in cost.groupby("sample"):
        zero = group.loc[group["cost_multiplier"] == 0.0, "total_return"].iloc[0]
        assert (group["cost_drag"] == zero - group["total_return"]).all()


def test_sensitivity_and_ablation_artifacts() -> None:
    rebalance = pd.read_parquet(PROCESSED / "rebalance_sensitivity_d4.parquet")
    assert set(rebalance["rebalance_frequency"]) == {"daily", "weekly", "monthly"}
    breadth = pd.read_parquet(PROCESSED / "breadth_sensitivity.parquet")
    assert set(breadth["breadth_threshold"]) == {0.35, 0.40, 0.45}
    assert set(pd.read_parquet(PROCESSED / "factor_ablation.parquet")["scenario"]) == {
        "FULL",
        "ABLATION_A",
        "ABLATION_B",
    }


def test_metrics_and_regression_artifacts() -> None:
    metrics = compute_metrics(pd.Series([0.01, -0.02, 0.03]))
    assert metrics["max_drawdown"] < 0 and metrics["calmar"] == metrics["calmar"]
    frame = pd.DataFrame(
        {"factor": [1.0, 2.0, 3.0, 4.0], "return": [2.0, 4.0, 6.0, 8.0]}
    )
    fitted = fit_hac_regression(frame, 0)
    assert fitted.loc[
        fitted["regressor"] == "factor", "coefficient"
    ].iloc[0] == pytest.approx(2.0)
    regression = pd.read_parquet(PROCESSED / "statsmodels_regression.parquet")
    assert {"regressor", "nobs", "lag", "notes"}.issubset(regression.columns)
    assert set(regression["notes"]) == {"LIMITED ATTRIBUTION"}


def test_scoreboard_reports_and_no_manifest_writer() -> None:
    scoreboard = pd.read_parquet(PROCESSED / "robustness_scoreboard.parquet")
    assert set(scoreboard["category"]) == {
        "BASELINE",
        "TOP_N",
        "BUFFER",
        "BREADTH",
        "REBALANCE",
        "COST",
        "ABLATION",
        "REGRESSION",
    }
    source = (ROOT / "run_robustness.py").read_text()
    assert 'strategy_freeze_manifest.json").write' not in source
    for path in [
        ROOT / "reports" / "d4_robustness_report.md",
        ROOT / "reports" / "statistical_diagnostics_report.md",
    ]:
        text = path.read_text()
        assert "PARTIAL-UNIVERSE" in text and "PIPELINE STATUS" in text

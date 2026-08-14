"""Render a compact performance report from canonical backtest artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from run_data_pipeline import resolve_path
from twse_factor_lab.analysis.pyfolio_adapter import to_pyfolio_inputs


def _config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _paths(config_path: str | Path, config: dict[str, Any]) -> dict[str, Path]:
    defaults = {
        "backtest_results": "data/processed/backtest_results.parquet",
        "backtest_metrics": "data/processed/backtest_metrics.parquet",
        "backtest_engine_comparison": (
            "data/processed/backtest_engine_comparison.parquet"
        ),
        "performance_report": "reports/performance_report.md",
        "performance_dir": "reports/performance",
    }
    return {
        name: resolve_path(config_path, value)
        for name, value in (defaults | (config.get("paths", {}) or {})).items()
    }


def _save_plot(path: Path, series: pd.Series, title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 3))
    series.plot(ax=axis)
    axis.set(title=title, ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _charts(results: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    returns, _, _ = to_pyfolio_inputs(results)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    window = min(20, max(2, len(returns)))
    volatility = returns.rolling(window).std(ddof=0) * np.sqrt(252)
    sharpe = returns.rolling(window).mean() / returns.rolling(window).std(ddof=0)
    monthly = returns.resample("ME").sum()
    _save_plot(
        output / "cumulative_return.png", equity - 1, "Cumulative Return", "return"
    )
    _save_plot(output / "drawdown.png", drawdown, "Drawdown", "drawdown")
    _save_plot(output / "rolling_sharpe.png", sharpe, "Rolling Sharpe", "sharpe")
    _save_plot(
        output / "rolling_volatility.png",
        volatility,
        "Rolling Volatility",
        "volatility",
    )
    _save_plot(output / "monthly_returns.png", monthly, "Monthly Returns", "return")
    _save_plot(
        output / "turnover.png",
        pd.Series(results["turnover"].to_numpy(), index=returns.index),
        "Turnover",
        "turnover",
    )


def run_performance_report(config_path: str | Path) -> dict[str, Path]:
    config = _config(config_path)
    paths = _paths(config_path, config)
    results = pd.read_parquet(paths["backtest_results"])
    metrics = pd.read_parquet(paths["backtest_metrics"]).iloc[0]
    comparison_path = paths["backtest_engine_comparison"]
    comparison = (
        pd.read_parquet(comparison_path) if comparison_path.exists() else pd.DataFrame()
    )
    _charts(results, paths["performance_dir"])
    try:
        import pyfolio as pf

        returns, positions, transactions = to_pyfolio_inputs(results)
        pf.create_full_tear_sheet(
            returns, positions=positions, transactions=transactions
        )
    except Exception:
        pass
    import pyfolio
    import vectorbt

    drawdown = (1 + results["returns"]).cumprod()
    drawdown = drawdown / drawdown.cummax() - 1
    monthly = (
        pd.Series(results["returns"].to_numpy(), index=pd.to_datetime(results["date"]))
        .resample("ME")
        .sum()
    )
    rolling_volatility = (
        pd.Series(results["returns"].to_numpy())
        .rolling(min(20, max(2, len(results))))
        .std(ddof=0)
        .iloc[-1]
    )
    parity = (
        "unavailable" if comparison.empty else ", ".join(comparison["status"].unique())
    )
    lines = [
        "# Performance Report",
        "",
        "## Engine",
        "",
        f"- Requested Engine: {metrics.requested_engine}",
        f"- Actual Engine: {metrics.actual_engine}",
        f"- Vectorbt Version: {vectorbt.__version__}",
        f"- Pyfolio Version: {pyfolio.__version__}",
        "",
        "## Headline Metrics",
        "",
    ]
    for name in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "turnover",
    ]:
        lines.append(f"- {name}: {metrics[name]}")
    lines += [
        "",
        "## Engine Parity Status",
        "",
        f"- {parity}",
        "",
        "## Top Drawdowns",
        "",
        f"- worst_drawdown: {drawdown.min()}",
        "",
        "## Rolling Risk Summary",
        "",
        f"- latest_rolling_volatility: {rolling_volatility}",
        "",
        "## Monthly Return Summary",
        "",
        f"- months: {len(monthly)}",
        f"- mean_monthly_return: {monthly.mean()}",
    ]
    paths["performance_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["performance_report"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a backtest performance report."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    for name, path in run_performance_report(args.config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

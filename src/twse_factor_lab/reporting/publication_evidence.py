"""Research-only, identity-bound publication evidence derivatives."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .performance_adapter import (
    PerformanceData,
    PerformanceDataError,
    load_frozen_identity,
    load_repository_performance_data,
)
from .performance_metrics import DEFAULT_ROLLING_WINDOW, calculate_metrics

RESEARCH_ID = "fundamental-production-final-v1"
STRATEGY_ID = "fundamental_g2g3_top5_reb60_score_weighted_v1"
FINGERPRINT = "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f"
PERIOD = "2021-01-04 to 2025-12-31"
TOLERANCE = 1e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite(series: pd.Series, name: str) -> pd.Series:
    result = pd.Series(series, copy=True, dtype=float)
    if not isinstance(result.index, pd.DatetimeIndex):
        raise PerformanceDataError(f"{name} must have a DatetimeIndex")
    if not result.index.is_monotonic_increasing or not result.index.is_unique:
        raise PerformanceDataError(f"{name} dates must be ascending and unique")
    if not np.isfinite(result.to_numpy()).all():
        raise PerformanceDataError(f"{name} contains non-finite values")
    return result


def build_pyfolio_inputs(data: PerformanceData) -> dict[str, Any]:
    """Return validated diagnostic inputs without changing canonical accounting."""

    if not data.available or data.returns is None:
        raise PerformanceDataError("BACKTEST canonical returns are unavailable")
    returns = _finite(data.returns, "returns")
    benchmark = data.benchmark_returns
    if benchmark is None:
        raise PerformanceDataError("canonical benchmark returns are unavailable")
    benchmark = _finite(benchmark, "benchmark_returns")
    if not returns.index.equals(benchmark.index):
        raise PerformanceDataError("benchmark dates must match returns without fill")
    returns, positions, transactions = data.to_pyfolio_inputs()
    returns = _finite(returns, "returns")
    if positions is not None and not positions.index.equals(returns.index):
        raise PerformanceDataError("positions dates must align exactly with returns")
    if transactions is not None and not transactions.empty:
        if not transactions.index.isin(returns.index).all():
            raise PerformanceDataError("transactions dates must be a returns subset")
    return {
        "returns": returns,
        "benchmark_returns": benchmark,
        "positions": positions,
        "transactions": transactions,
    }


def _pyfolio_diagnostics(returns: pd.Series) -> dict[str, Any]:
    """Run the installed API for disclosure, never for official metrics."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Module "zipline.assets" not found')
        from pyfolio import timeseries

        metrics = timeseries.perf_stats(returns)
    return {
        "status": "AVAILABLE",
        "role": "PYFOLIO_DIAGNOSTIC_ONLY",
        "metrics": {
            str(name): float(value)
            for name, value in metrics.items()
            if np.isfinite(float(value))
        },
    }


def _factor_frame(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = root / "data/research" / RESEARCH_ID / "historical-evidence"
    daily = pd.read_csv(base / "factor_validation.csv", parse_dates=["date"])
    quantiles = pd.read_csv(
        base / "factor_quantile_validation.csv", parse_dates=["date"]
    )
    summaries = {
        name: _json(base / f"factor_{name.lower()}_validation.json")
        for name in ("G2", "G3", "COMPOSITE")
    }
    return daily, quantiles, summaries


def _caption(title: str, source: str, limitation: str) -> str:
    return (
        f"**{title}.** Period: {PERIOD}. Source: `{source}`. "
        f"Strategy: `{STRATEGY_ID}`. "
        f"Limitation: {limitation}"
    )


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _factor_figures(
    daily: pd.DataFrame,
    quantiles: pd.DataFrame,
    summaries: dict[str, Any],
    output: Path,
) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    daily = daily.loc[daily["horizon"].eq(60)].copy()
    quantiles = quantiles.loc[quantiles["horizon"].eq(60)].copy()
    paths: dict[str, Path] = {}

    figure, axis = plt.subplots(figsize=(11, 5))
    for name in ("G2", "G3", "COMPOSITE"):
        rows = daily.loc[daily["factor"].eq(name)]
        item = summaries[name]
        axis.plot(rows["date"], rows["ic"], label=(
            f"{name} mean={item['rank_ic']:.3f}, n={item['ic_observations']}"
        ))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(title="F1. 60-Day Rank IC Time Series", ylabel="Spearman Rank IC")
    axis.legend(fontsize=8)
    paths["F1"] = output / "01_rank_ic_timeseries.png"
    _save(figure, paths["F1"])

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, name in zip(axes, ("G2", "G3", "COMPOSITE"), strict=True):
        rows = daily.loc[daily["factor"].eq(name), "ic"]
        item = summaries[name]
        interval = item["bootstrap_ic_ci"]
        axis.hist(rows, bins=12, color="#4C78A8", alpha=0.8)
        axis.axvline(item["rank_ic"], color="#E45756", label="mean")
        axis.axvline(item["ic_median"], color="#54A24B", label="median")
        axis.set_title(f"{name}\n95% CI [{interval['low']:.3f}, {interval['high']:.3f}]")
        axis.set_xlabel("Rank IC")
    axes[0].set_ylabel("Evaluation dates")
    axes[-1].legend(fontsize=8)
    paths["F2"] = output / "02_ic_distribution_bootstrap.png"
    _save(figure, paths["F2"])

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, name in zip(axes, ("G2", "G3", "COMPOSITE"), strict=True):
        values = (
            quantiles.loc[quantiles["factor"].eq(name)]
            .groupby("quantile", sort=True)["mean_forward_return"]
            .mean()
        )
        axis.bar([f"Q{number}" for number in values.index], values, color="#4C78A8")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{name}\nQ5−Q1={values.iloc[-1] - values.iloc[0]:.2%}")
        axis.set_xlabel("Factor quantile")
    axes[0].set_ylabel("Mean 60-day forward return")
    paths["F3"] = output / "03_quantile_forward_returns.png"
    _save(figure, paths["F3"])

    pivot = daily.pivot(index="date", columns="factor", values="coverage")
    eligible = daily.pivot(index="date", columns="factor", values="eligible_count")
    figure, left = plt.subplots(figsize=(11, 5))
    right = left.twinx()
    eligible.max(axis=1).plot(ax=left, color="black", label="Eligible universe")
    pivot[["G2", "G3", "COMPOSITE"]].rename(
        columns={"COMPOSITE": "Both-valid"}
    ).plot(ax=right)
    left.set(title="F4. Factor Coverage", ylabel="Eligible universe")
    right.set_ylabel("Valid ratio")
    right.set_ylim(0, 1)
    paths["F4"] = output / "04_factor_coverage.png"
    _save(figure, paths["F4"])
    return paths


def _factor_table(summaries: dict[str, Any], path: Path) -> None:
    columns = [
        "factor", "mean_rank_ic", "median_rank_ic", "ic_95ci_low",
        "ic_95ci_high", "positive_ic_ratio", "top_bottom_spread",
        "positive_year_ratio", "coverage", "observations", "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name in ("G2", "G3", "COMPOSITE"):
            item = summaries[name]
            interval = item.get("bootstrap_ic_ci", {})
            writer.writerow({
                "factor": name, "mean_rank_ic": item.get("rank_ic"),
                "median_rank_ic": item.get("ic_median"),
                "ic_95ci_low": interval.get("low"), "ic_95ci_high": interval.get("high"),
                "positive_ic_ratio": item.get("ic_positive_ratio"),
                "top_bottom_spread": item.get("top_bottom_spread"),
                "positive_year_ratio": item.get("positive_ic_year_ratio"),
                "coverage": item.get("coverage"), "observations": item.get("ic_observations"),
                "status": "AVAILABLE",
            })


def _evidence_tables(root: Path, metrics: Any, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)

    def lookup(name: str) -> Any:
        return metrics.metrics[name].value
    rows = [
        ("Total Return", "total_return", "benchmark_return"),
        ("CAGR", "cagr", "benchmark_cagr"),
        ("Sharpe", "sharpe", "benchmark_sharpe"),
        ("Sortino", "sortino", None),
        ("Annualized Volatility", "annualized_volatility", None),
        ("Max Drawdown", "max_drawdown", "benchmark_max_drawdown"),
        ("Turnover", "turnover", None), ("Transaction Cost", "transaction_cost", None),
        ("Excess Return", "excess_return", None), ("Excess CAGR", None, None),
        ("Alpha", "alpha", None), ("Beta", "beta", None),
        ("Tracking Error", "tracking_error", None),
        ("Information Ratio", "information_ratio", None),
    ]
    strategy_path = output / "strategy_benchmark_evidence.csv"
    with strategy_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Metric", "Strategy", "Benchmark", "Difference", "Canonical Source", "Status",
        ])
        writer.writeheader()
        for label, strategy_key, benchmark_key in rows:
            if label == "Excess CAGR":
                strategy = lookup("cagr") - lookup("benchmark_cagr")
                bench = None
            else:
                strategy = lookup(strategy_key) if strategy_key else None
                bench = lookup(benchmark_key) if benchmark_key else None
            difference = strategy - bench if strategy is not None and bench is not None else strategy
            writer.writerow({
                "Metric": label, "Strategy": strategy, "Benchmark": bench,
                "Difference": difference,
                "Canonical Source": "performance_metrics.csv; benchmark_comparison.csv",
                "Status": "AVAILABLE" if strategy is not None else "UNAVAILABLE",
            })
    final = root / "data/research" / RESEARCH_ID
    pit = _json(final / "historical-evidence/historical_pit_audit.json")
    fresh = _json(final / "fresh_oos_validation.json")
    validation = [
        ("PIT Integrity", pit["status"], "historical_pit_audit.json"),
        ("Future Leakage", pit["leakage"], "historical_pit_audit.json"),
        ("Historical Evidence", "HISTORICALLY_SUPPORTIVE", "historical_strategy_validation.json"),
        ("Benchmark", "AVAILABLE", "benchmark_validation.json"),
        ("Benchmark Definition Compatibility", "DEFINITION_DIFFERENCE", "benchmark_return_convention_audit.json"),
        ("MA60 REB60 Overlay", "ADVERSE", "risk_overlay_research_closure.json"),
        ("MA60 Daily Overlay", "ADVERSE", "risk_overlay_research_closure.json"),
        ("Risk Overlay Route", "CLOSED_NOT_ADOPTED", "risk_overlay_research_closure.json"),
        ("Survivorship Bias", "BLOCKED / NOT_FULLY_RESOLVED", "survivorship_bias_validation.json"),
        ("Historical OOS", "INSUFFICIENT_DATA", "performance_summary.json"),
        ("Fresh OOS", fresh["status"], "fresh_oos_validation.json"),
        ("Production Gate", "BLOCKED", "production_final_gate.json"),
    ]
    validation_path = output / "research_validation_status.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Validation", "Status", "Canonical Source"])
        writer.writerows(validation)
    return [strategy_path, validation_path]


def _fresh_oos_figure(root: Path, path: Path) -> None:
    audit = _json(root / "data/research" / RESEARCH_ID / "fresh_oos_eligibility_audit.json")
    current = [audit["observed"]["trading_days"], audit["observed"]["months"], audit["observed"]["rebalances"]]
    required = [audit["required"]["trading_days"], audit["required"]["months"], audit["required"]["rebalances"]]
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    axis.bar(x - 0.18, current, width=0.36, label="Observed")
    axis.bar(x + 0.18, required, width=0.36, label="Required")
    axis.set_xticks(x, ["Trading days", "Calendar months", "Legal rebalances"])
    axis.set(title="Fresh OOS Availability — Accumulation Status Only", ylabel="Count")
    axis.legend()
    axis.text(0.5, -0.25, "NOT PERFORMANCE EVALUATION; formal status: INSUFFICIENT_DATA", ha="center", transform=axis.transAxes)
    _save(figure, path)


def _copy_portfolio_sources(root: Path, output: Path) -> dict[str, Path]:
    source = root / "data/research" / RESEARCH_ID / "performance/charts"
    mapping = {
        "P1": ("02_strategy_vs_benchmark.png", "01_strategy_vs_benchmark.png"),
        "P2": ("04_drawdown_underwater.png", "02_drawdown_underwater.png"),
        "P3": ("05_top_drawdowns.png", "03_top_drawdowns.png"),
        "P4": ("07_rolling_sharpe.png", "04_rolling_sharpe.png"),
        "P5": ("08_rolling_volatility.png", "05_rolling_volatility.png"),
        "P6": ("09_rolling_beta.png", "06_rolling_beta.png"),
        "P7": ("10_monthly_returns_heatmap.png", "07_monthly_returns_heatmap.png"),
        "P8": ("11_annual_returns.png", "08_annual_returns.png"),
        "P9": ("13_daily_return_distribution.png", "09_daily_return_distribution.png"),
        "P10": ("14_qq_plot.png", "10_daily_return_qq.png"),
        "P11": ("15_gross_exposure.png", "11_gross_exposure.png"),
        "P12": ("16_net_exposure.png", "12_net_exposure.png"),
        "P13": ("17_position_concentration.png", "13_position_concentration.png"),
        "P14": ("18_turnover.png", "14_turnover.png"),
        "P15": ("19_transaction_cost.png", "15_transaction_cost.png"),
        "P16": ("20_round_trip_returns.png", "16_round_trip_returns.png"),
        "P17": ("21_round_trip_duration.png", "17_round_trip_duration.png"),
    }
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, (old, new) in mapping.items():
        target = output / new
        shutil.copy2(source / old, target)
        result[key] = target
    return result


def build_publication_evidence(root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    """Build only derivative publication assets from validated canonical evidence."""

    root = Path(root).resolve()
    identity = load_frozen_identity(root)
    if identity.strategy_id != STRATEGY_ID or identity.strategy_fingerprint != FINGERPRINT:
        raise PerformanceDataError("FROZEN_STRATEGY_CHANGED")
    data = load_repository_performance_data(root, identity)["BACKTEST"]
    inputs = build_pyfolio_inputs(data)
    metrics = calculate_metrics(data, rolling_window=DEFAULT_ROLLING_WINDOW)
    total = float((1.0 + inputs["returns"]).prod() - 1.0)
    canonical_total = float(metrics.metrics["total_return"].value)
    if not np.isclose(total, canonical_total, atol=TOLERANCE, rtol=0):
        raise PerformanceDataError("PYFOLIO_RETURN_PARITY_FAIL")
    drawdown = (1.0 + inputs["returns"]).cumprod().div(
        (1.0 + inputs["returns"]).cumprod().cummax()
    ).sub(1.0).min()
    final = root / "data/research" / RESEARCH_ID
    target = Path(output).resolve() if output else final / "publication-evidence"
    factor_dir, portfolio_dir = target / "factor", target / "portfolio"
    tables_dir, manifests_dir = target / "tables", target / "manifests"
    benchmark_dir = target / "benchmark"
    for directory in (factor_dir, portfolio_dir, tables_dir, manifests_dir, benchmark_dir):
        directory.mkdir(parents=True, exist_ok=True)
    daily, quantiles, summaries = _factor_frame(root)
    factors = _factor_figures(daily, quantiles, summaries, factor_dir)
    factor_table = tables_dir / "factor_evidence.csv"
    _factor_table(summaries, factor_table)
    tables = [factor_table, *_evidence_tables(root, metrics, tables_dir)]
    portfolio = _copy_portfolio_sources(root, portfolio_dir)
    fresh_path = benchmark_dir / "01_fresh_oos_availability.png"
    _fresh_oos_figure(root, fresh_path)
    source_paths = [
        final / "performance_source_manifest.json",
        final / "historical-evidence/canonical_backtest_manifest.json",
        final / "historical-evidence/canonical_backtest_returns.parquet",
        final / "historical-evidence/factor_validation.csv",
        final / "historical-evidence/factor_quantile_validation.csv",
        final / "historical-evidence/factor_g2_validation.json",
        final / "historical-evidence/factor_g3_validation.json",
        final / "historical-evidence/factor_composite_validation.json",
        final / "benchmark/benchmark_comparison.csv",
        final / "fresh_oos_eligibility_audit.json",
    ]
    pyfolio_manifest = {
        "pyfolio_available": True,
        "pyfolio_package": "pyfolio-reloaded",
        "pyfolio_version": importlib.metadata.version("pyfolio-reloaded"),
        "pyfolio_api_style": "pyfolio.timeseries / plotting (diagnostic only)",
        "role": "PYFOLIO_DIAGNOSTIC_ONLY",
        "returns": {"observations": len(inputs["returns"]), "sha256": _sha256(source_paths[2])},
        "benchmark_returns": {"observations": len(inputs["benchmark_returns"]), "aligned": True},
        "positions": {
            "status": "AVAILABLE" if inputs["positions"] is not None else "UNAVAILABLE",
            "rows": len(inputs["positions"]) if inputs["positions"] is not None else 0,
        },
        "transactions": {
            "status": "AVAILABLE" if inputs["transactions"] is not None else "UNAVAILABLE",
            "rows": len(inputs["transactions"]) if inputs["transactions"] is not None else 0,
            "synthetic": False,
        },
        "no_forward_fill": True,
        "diagnostics": _pyfolio_diagnostics(inputs["returns"]),
    }
    _write_json(manifests_dir / "pyfolio_input_manifest.json", pyfolio_manifest)
    parity = {
        "classification": "PASS_SAME_RETURN_SERIES",
        "canonical_total_return": canonical_total,
        "growth_of_one_return": total,
        "total_return_difference": total - canonical_total,
        "canonical_max_drawdown": metrics.metrics["max_drawdown"].value,
        "same_series_max_drawdown": float(drawdown),
        "canonical_volatility": metrics.metrics["annualized_volatility"].value,
        "same_series_volatility": float(inputs["returns"].std(ddof=0) * np.sqrt(252)),
        "definition_differences": [
            "Pyfolio diagnostics are not canonical accounting metrics.",
            "Pyfolio Sharpe and volatility use package-defined conventions.",
            "Strategy uses adjusted_price_return; TAIEX uses index_price_return (DEFINITION_DIFFERENCE).",
        ],
    }
    _write_json(manifests_dir / "pyfolio_parity_audit.json", parity)
    docs = root / "docs/research-publication/assets"
    figure_docs, table_docs = docs / "figures", docs / "tables"
    figure_docs.mkdir(parents=True, exist_ok=True)
    table_docs.mkdir(parents=True, exist_ok=True)
    outputs = {**factors, **portfolio, "OOS": fresh_path}
    for key, path in outputs.items():
        shutil.copy2(path, figure_docs / f"{key}_{path.name}")
    for path in tables:
        shutil.copy2(path, table_docs / path.name)
    artifacts = []
    for key, path in outputs.items():
        artifacts.append({
            "artifact_id": key, "file_path": str(path.relative_to(root)),
            "research_question": "RQ1" if key.startswith("F") else ("RQ3" if key == "OOS" else "RQ2"),
            "source_files": [str(item.relative_to(root)) for item in source_paths],
            "source_hashes": {str(item.relative_to(root)): _sha256(item) for item in source_paths},
            "strategy_fingerprint": FINGERPRINT, "period": PERIOD,
            "generator": "twse_factor_lab.reporting.publication_evidence",
            "generated_at": datetime.now(UTC).isoformat(),
            "canonical_or_derivative": "DERIVATIVE_EVIDENCE", "status": "GENERATED",
        })
    for path in tables:
        artifacts.append({
            "artifact_id": path.stem, "file_path": str(path.relative_to(root)),
            "research_question": "RQ1" if path.name == "factor_evidence.csv" else "RQ2/RQ3",
            "source_files": [str(item.relative_to(root)) for item in source_paths],
            "source_hashes": {str(item.relative_to(root)): _sha256(item) for item in source_paths},
            "strategy_fingerprint": FINGERPRINT, "period": PERIOD,
            "generator": "twse_factor_lab.reporting.publication_evidence",
            "generated_at": datetime.now(UTC).isoformat(),
            "canonical_or_derivative": "DERIVATIVE_EVIDENCE", "status": "GENERATED",
        })
    _write_json(target / "publication_evidence_manifest.json", {"artifacts": artifacts})
    captions = []
    for key, path in outputs.items():
        source = (
            "historical-evidence factor CSV/JSON"
            if key.startswith("F")
            else (
                "fresh_oos_eligibility_audit.json"
                if key == "OOS"
                else "performance/charts"
            )
        )
        limitation = (
            "PIT historical factor evidence; not future profitability."
            if key.startswith("F")
            else (
                "Accumulation status only, not performance evaluation."
                if key == "OOS"
                else "Historical BACKTEST only; TAIEX is price-index while strategy returns use adjusted prices (DEFINITION_DIFFERENCE)."
            )
        )
        captions.append(_caption(f"{key} {path.stem}", source, limitation))
    report = "\n".join([
        "# Publication Evidence Report", "",
        "- Strategy identity: `fundamental_g2g3_top5_reb60_score_weighted_v1`.",
        "- Canonical accounting remains authoritative; Pyfolio is `PYFOLIO_DIAGNOSTIC_ONLY`.",
        "- Fresh OOS accumulation has begun / operational evidence exists, but the predeclared formal evaluation minimum has not been satisfied (`INSUFFICIENT_DATA`).", "",
        *captions,
    ]) + "\n"
    (target / "publication_evidence_report.md").write_text(report, encoding="utf-8")
    return {"target": target, "outputs": outputs, "tables": tables, "parity": parity}


__all__ = ["build_publication_evidence", "build_pyfolio_inputs"]

"""Build the identity-bound broad-market benchmark for the frozen strategy."""

# ruff: noqa: E501,E702

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf

from twse_factor_lab.reporting.performance_adapter import (
    PerformanceData,
    load_frozen_identity,
    load_repository_performance_data,
)
from twse_factor_lab.reporting.performance_metrics import (
    DEFAULT_ROLLING_WINDOW,
    calculate_metrics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMESPACE = Path("data/research/fundamental-production-final-v1")
OUTPUT = NAMESPACE / "benchmark"
START = pd.Timestamp("2021-01-04")
END = pd.Timestamp("2025-12-31")
SOURCE_SERIES = "^TWII"
BENCHMARK_ID = "twii_taiex_broad_market_price_index_v1"
POLICY_VERSION = "canonical_benchmark_selection_v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    return hashlib.sha256(encoded).hexdigest()


def _download() -> pd.DataFrame:
    frame = yf.download(
        SOURCE_SERIES,
        start="2021-01-01",
        end="2026-01-02",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError("NO_AUTHORITATIVE_BROAD_MARKET_SOURCE")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    column = "Close" if "Close" in frame.columns else "Adj Close"
    frame = frame.rename_axis("trade_date").reset_index()[["trade_date", column]]
    frame.columns = ["trade_date", "benchmark_level"]
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["benchmark_level"] = pd.to_numeric(frame["benchmark_level"], errors="coerce")
    return frame.loc[(frame["trade_date"] >= START) & (frame["trade_date"] <= END)].copy()


def _calendar(root: Path) -> pd.DatetimeIndex:
    returns = pd.read_parquet(
        root / NAMESPACE / "historical-evidence/canonical_backtest_returns.parquet"
    )
    dates = pd.to_datetime(returns["date"], errors="coerce").dt.normalize()
    return pd.DatetimeIndex(dates)


def _validate(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict[str, Any]:
    duplicate = int(frame["trade_date"].duplicated().sum())
    invalid_dates = int(frame["trade_date"].isna().sum())
    invalid_values = int((~np.isfinite(frame["benchmark_level"].to_numpy())).sum())
    non_positive = int((frame["benchmark_level"] <= 0).sum())
    expected = pd.DatetimeIndex(calendar)
    actual = pd.DatetimeIndex(frame["trade_date"])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    ascending = bool(actual.is_monotonic_increasing)
    aligned_count = len(expected.intersection(actual))
    return {
        "start_date": expected.min().date().isoformat(),
        "end_date": expected.max().date().isoformat(),
        "expected_sessions": len(expected),
        "available_sessions": aligned_count,
        "missing_sessions": [d.date().isoformat() for d in missing],
        "extra_sessions": [d.date().isoformat() for d in extra],
        "duplicate_sessions": duplicate,
        "invalid_dates": invalid_dates,
        "invalid_values": invalid_values,
        "non_positive_levels": non_positive,
        "coverage_ratio": len(expected.intersection(actual)) / len(expected),
        "status": "PASS"
        if not (duplicate or invalid_dates or invalid_values or non_positive or len(missing) or not ascending)
        else "BLOCKED",
    }


def _returns(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    frame = frame.set_index("trade_date").reindex(calendar)
    if frame["benchmark_level"].isna().any():
        raise ValueError("INSUFFICIENT_BENCHMARK_COVERAGE")
    frame["benchmark_return"] = frame["benchmark_level"].pct_change().fillna(0.0)
    frame["daily_return"] = frame["benchmark_return"]
    if not np.isfinite(frame["benchmark_return"].to_numpy()).all():
        raise ValueError("BENCHMARK_DATA_INTEGRITY_FAILURE")
    return frame.reset_index(names="trade_date")


def _metric_value(metrics: Any, name: str) -> Any:
    item = metrics.metrics[name]
    return item.value if item.status == "AVAILABLE" else None


def _comparison(strategy: Any, benchmark: Any) -> pd.DataFrame:
    names = (
        ("total_return", "total_return"),
        ("cagr", "cagr"),
        ("annualized_volatility", "annualized_volatility"),
        ("sharpe", "sharpe"),
        ("sortino", "sortino"),
        ("max_drawdown", "max_drawdown"),
        ("excess_return", "excess_return"),
        ("alpha", "alpha"),
        ("beta", "beta"),
        ("tracking_error", "tracking_error"),
        ("information_ratio", "information_ratio"),
    )
    rows = []
    for metric, strategy_name in names:
        s = _metric_value(strategy, strategy_name)
        b = _metric_value(benchmark, metric) if metric not in {"excess_return", "alpha", "beta", "tracking_error", "information_ratio"} else None
        rows.append({"metric": metric, "strategy": s, "benchmark": b, "difference": s - b if s is not None and b is not None else s, "status": "AVAILABLE" if s is not None else "UNAVAILABLE"})
    return pd.DataFrame(rows)


def _charts(output: Path, strategy_metrics: Any, benchmark_metrics: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    strategy = strategy_metrics.series["cumulative_returns"]
    bench = benchmark_metrics.series["cumulative_returns"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy.index, 1 + strategy, label="Strategy")
    ax.plot(bench.index, 1 + bench, label="Benchmark")
    ax.set_title("Strategy vs Canonical Benchmark")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "strategy_vs_benchmark.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    beta = strategy_metrics.series.get("rolling_beta")
    if beta is not None:
        ax.plot(beta.index, beta, label=f"Rolling beta ({DEFAULT_ROLLING_WINDOW})")
    ax.axhline(1.0, color="black", lw=0.8); ax.set_title("Rolling Beta"); ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "rolling_beta.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy_metrics.series["drawdown"].index, strategy_metrics.series["drawdown"], label="Strategy")
    ax.plot(benchmark_metrics.series["drawdown"].index, benchmark_metrics.series["drawdown"], label="Benchmark")
    ax.set_title("Strategy vs Benchmark Drawdown"); ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "benchmark_drawdown_comparison.png", dpi=140); plt.close(fig)


def generate_canonical_benchmark(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    identity = load_frozen_identity(root)
    if identity.strategy_id != "fundamental_g2g3_top5_reb60_score_weighted_v1" or identity.strategy_fingerprint != "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f":
        raise ValueError("FROZEN_STRATEGY_CHANGED")
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    calendar = _calendar(root)
    source_policy = {
        "policy_version": POLICY_VERSION,
        "priority": ["repository_approved_broad_market", "authoritative_broad_taiwan_index"],
        "selected_source": {"provider": "yfinance", "source_series_id": SOURCE_SERIES},
        "broad_market_requirement": "Taiwan broad equity market index",
        "etf_proxy_policy": "PROHIBITED",
        "synthetic_survivor_benchmark_policy": "PROHIBITED",
    }
    source_policy["policy_hash"] = _sha(source_policy)
    _write_json(
        output / "benchmark_inventory.json",
        {
            "existing_benchmark_sources": [],
            "existing_index_data": [],
            "available_date_ranges": {"yfinance:^TWII": ["2021-01-04", "2025-12-31"]},
            "return_convention": "index_price_return",
            "strategy_return_convention": "adjusted_price_return",
            "calendar": "canonical_strategy_trading_calendar",
            "candidate_sources": [
                {"provider": "yfinance", "source_series_id": "^TWII", "type": "broad_market_index", "canonical": True},
                {"provider": "ETF", "source_series_id": "0050/006208", "type": "proxy", "canonical": False},
            ],
            "selected_source": {"provider": "yfinance", "source_series_id": SOURCE_SERIES},
            "warnings": ["TAIEX is a price index; dividend definition differs from adjusted-price strategy returns.", "SURVIVORSHIP_BIAS_NOT_FULLY_RESOLVED"],
        },
    )
    _write_json(output / "canonical_benchmark_policy.json", source_policy)
    frame = _download()
    coverage = _validate(frame, calendar)
    _write_json(output / "benchmark_coverage_report.json", coverage)
    if coverage["status"] != "PASS":
        raise ValueError("BENCHMARK_DATA_INTEGRITY_FAILURE")
    frame = _returns(frame, calendar)
    source_records = frame[["trade_date", "benchmark_level"]].to_dict("records")
    source_hash = _sha(source_records)
    config = {"benchmark_id": BENCHMARK_ID, "start": START.date().isoformat(), "end": END.date().isoformat(), "calendar_sessions": len(calendar), "source_series_id": SOURCE_SERIES, "return_convention": "index_price_return"}
    config_hash = _sha(config)
    fingerprint = _sha({"source_hash": source_hash, "config_hash": config_hash, "benchmark_id": BENCHMARK_ID})
    spec = {"benchmark_id": BENCHMARK_ID, "benchmark_name": "TAIEX (TWII) Broad Taiwan Equity Price Index", "provider": "yfinance", "source": "Yahoo Finance", "source_series_id": SOURCE_SERIES, "market": "TW", "currency": "TWD", "return_convention": "index_price_return", "start_date": START.date().isoformat(), "end_date": END.date().isoformat(), "calendar": "canonical_strategy_trading_calendar", "adjustment_policy": "Close; no dividend treatment; no forward fill", "source_hash": source_hash, "config_hash": config_hash, "benchmark_fingerprint": fingerprint}
    _write_json(output / "canonical_benchmark_spec.json", spec)
    frame.insert(0, "date", frame["trade_date"])
    frame.to_parquet(output / "canonical_benchmark_returns.parquet", index=False)
    audit = {"strategy_return_type": "adjusted_price_return", "benchmark_return_type": "index_price_return", "dividend_treatment": "strategy adjusted prices may reflect distributions; TAIEX index series does not", "corporate_action_treatment": "provider index methodology; no local synthetic adjustment", "currency": "TWD", "frequency": "daily trading sessions", "compatibility_status": "DEFINITION_DIFFERENCE"}
    _write_json(output / "benchmark_return_convention_audit.json", audit)
    manifest = {**spec, "source": "yfinance.download(^TWII, auto_adjust=False)", "source_snapshot_hash": source_hash, "observations": len(frame), "missing_observations": 0, "generated_at": datetime.now(UTC).isoformat(), "code_commit": "working-tree"}
    _write_json(output / "canonical_benchmark_manifest.json", manifest)
    metrics_definition = {"return_frequency": "daily", "risk_free_rate": 0.0, "regression_method": "population covariance/variance on aligned daily returns", "annualization_factor": 252, "alignment_policy": "inner join on canonical strategy trading dates; no fill", "minimum_observations": 2, "rolling_beta_window": DEFAULT_ROLLING_WINDOW}
    _write_json(output / "benchmark_metric_definition.json", metrics_definition)

    periods = load_repository_performance_data(root, identity)
    strategy_data = periods["BACKTEST"]
    benchmark_series = pd.Series(frame["benchmark_return"].to_numpy(), index=pd.to_datetime(frame["date"]))
    combined = PerformanceData.from_frames("BACKTEST", strategy_data.returns, benchmark_returns=benchmark_series, positions=strategy_data.positions, transactions=strategy_data.transactions, benchmark_id=BENCHMARK_ID, benchmark_source="yfinance:^TWII", source=strategy_data.source)
    strategy_metrics = calculate_metrics(combined)
    benchmark_metrics = calculate_metrics(PerformanceData.from_frames("BACKTEST", benchmark_series))
    comparison = _comparison(strategy_metrics, benchmark_metrics)
    comparison.to_csv(output / "benchmark_comparison.csv", index=False)
    _charts(output / "charts", strategy_metrics, benchmark_metrics)
    validation = {"status": "PASS", "benchmark_status": "AVAILABLE", "benchmark_id": BENCHMARK_ID, "benchmark_fingerprint": fingerprint, "selection_policy": POLICY_VERSION, "selection_policy_hash": source_policy["policy_hash"], "coverage": coverage, "return_convention": audit, "strategy_id": identity.strategy_id, "strategy_fingerprint": identity.strategy_fingerprint, "rolling_beta_window": DEFAULT_ROLLING_WINDOW, "governance": {"historical_evidence": "HISTORICALLY_SUPPORTIVE", "fresh_oos": "INSUFFICIENT_DATA", "production_gate": "BLOCKED", "production_ready": False, "risk_overlay_route": "CLOSED_NOT_ADOPTED", "survivorship": "SURVIVORSHIP_BIAS_NOT_FULLY_RESOLVED"}}
    _write_json(output / "benchmark_validation.json", validation)
    report = f"""# Canonical Benchmark Validation Report

## 1. Benchmark Selection

Benchmark Name = {spec['benchmark_name']}
Benchmark ID = {BENCHMARK_ID}
Provider = yfinance
Source = Yahoo Finance `^TWII`

## 2. Selection Policy

The frozen policy selects an authoritative broad Taiwan equity index before any performance comparison. ETF products and synthetic current-listed-stock benchmarks are prohibited.

## 3. Return Convention

Strategy Return Convention = adjusted_price_return
Benchmark Return Convention = index_price_return
Compatibility = DEFINITION_DIFFERENCE (TAIEX series does not include dividends)

## 4. Coverage

Start = {coverage['start_date']}
End = {coverage['end_date']}
Sessions = {coverage['available_sessions']}
Missing Sessions = {len(coverage['missing_sessions'])}

## 5. Data Integrity

Duplicates = {coverage['duplicate_sessions']}
Missing = {len(coverage['missing_sessions'])}
Invalid Values = {coverage['invalid_values'] + coverage['non_positive_levels']}
Status = PASS

## 6. Strategy vs Benchmark

| Metric | Strategy | Benchmark | Difference |
|---|---:|---:|---:|
""" + "\n".join(f"| {r.metric} | {r.strategy} | {r.benchmark} | {r.difference} |" for r in comparison.itertuples()) + f"""

## 7. Relative Performance

Excess Return = {_metric_value(strategy_metrics, 'excess_return')}
Excess CAGR = {_metric_value(strategy_metrics, 'cagr') - _metric_value(benchmark_metrics, 'cagr')}
Alpha = {_metric_value(strategy_metrics, 'alpha')}
Beta = {_metric_value(strategy_metrics, 'beta')}
Tracking Error = {_metric_value(strategy_metrics, 'tracking_error')}
Information Ratio = {_metric_value(strategy_metrics, 'information_ratio')}

## 8. Drawdown

Strategy MDD = {_metric_value(strategy_metrics, 'max_drawdown')}
Benchmark MDD = {_metric_value(benchmark_metrics, 'max_drawdown')}

## 9. Limitations

SURVIVORSHIP_BIAS_NOT_FULLY_RESOLVED remains for the strategy universe. The benchmark is a price-index series, so dividend definition differences are disclosed rather than hidden.

## 10. Governance

Base Fingerprint = PASS
Base Modified = NO
Fresh OOS = INSUFFICIENT_DATA
Production Gate = BLOCKED

## 11. Conclusion

BENCHMARK_AVAILABLE
"""
    (output / "benchmark_validation_report.md").write_text(report, encoding="utf-8")

    manifest_path = root / NAMESPACE / "performance_source_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backtest = source_manifest["periods"]["BACKTEST"]
    backtest.update({"benchmark_id": BENCHMARK_ID, "benchmark_source": "yfinance:^TWII", "benchmark_status": "AVAILABLE", "benchmark_returns": (OUTPUT / "canonical_benchmark_returns.parquet").as_posix(), "benchmark_reason": None})
    manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


__all__ = ["generate_canonical_benchmark", "BENCHMARK_ID", "DEFAULT_ROLLING_WINDOW"]

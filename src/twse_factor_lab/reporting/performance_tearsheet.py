"""Comprehensive, evidence-safe performance charts and reports."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .performance_adapter import (
    PERIODS,
    FrozenIdentity,
    FrozenStrategyChanged,
    PerformanceData,
    load_frozen_identity,
    load_repository_performance_data,
)
from .performance_metrics import (
    DEFAULT_ROLLING_WINDOW,
    METRIC_NAMES,
    PerformanceMetrics,
    calculate_metrics,
)

OUTPUT_NAMESPACE = "data/research/fundamental-production-final-v1/performance"
EVIDENCE_RULES = {
    "minimum_fresh_oos_observations": 180,
    "supportive": {
        "total_return_min": 0.0,
        "sharpe_min": 0.0,
        "max_drawdown_min": -0.35,
        "excess_return_min_when_available": 0.0,
    },
    "adverse": {
        "total_return_max": 0.0,
        "sharpe_max": 0.0,
        "max_drawdown_max": -0.50,
    },
}

CHARTS = (
    ("01_cumulative_returns", "Cumulative Returns"),
    ("02_strategy_vs_benchmark", "Strategy vs Benchmark"),
    ("03_log_cumulative_returns", "Log Cumulative Returns"),
    ("04_drawdown_underwater", "Drawdown Underwater"),
    ("05_top_drawdowns", "Top Drawdowns"),
    ("06_rolling_returns", "Rolling Returns"),
    ("07_rolling_sharpe", "Rolling Sharpe Ratio"),
    ("08_rolling_volatility", "Rolling Volatility"),
    ("09_rolling_beta", "Rolling Beta"),
    ("10_monthly_returns_heatmap", "Monthly Returns Heatmap"),
    ("11_annual_returns", "Annual Returns"),
    ("12_return_distribution", "Monthly Return Distribution"),
    ("13_daily_return_distribution", "Daily Return Distribution"),
    ("14_qq_plot", "Daily Return QQ Plot"),
    ("15_gross_exposure", "Gross Exposure"),
    ("16_net_exposure", "Net Exposure"),
    ("17_position_concentration", "Position Concentration"),
    ("18_turnover", "Turnover"),
    ("19_transaction_cost", "Transaction Cost"),
    ("20_round_trip_returns", "Round-trip Returns"),
    ("21_round_trip_duration", "Round-trip Duration"),
    ("22_backtest_vs_oos", "Backtest vs Historical OOS"),
    (
        "23_historical_oos_vs_fresh_oos",
        "Historical OOS vs Fresh OOS",
    ),
    ("24_all_period_comparison", "All-period Comparison"),
)

CHART_HELP = {
    "cumulative": (
        "追蹤一元資金隨時間累積的報酬。",
        "曲線越高代表累積報酬越高；回落區段代表資產價值下降。",
    ),
    "benchmark": (
        "比較策略與 canonical Benchmark 的累積結果。",
        "兩條曲線的差距是相同重疊日期上的相對表現。",
    ),
    "drawdown": (
        "衡量資產淨值低於先前高點的幅度與持續時間。",
        "越往下風險越大；回到零表示已回本。",
    ),
    "rolling": (
        "用固定視窗觀察績效是否隨時間改變。",
        "重點看最新值、典型值、極端值及低於零的比例。",
    ),
    "calendar": (
        "把報酬依月份或年度整理，檢查時間集中與長期失效。",
        "正負月份的分布比單一總報酬更能顯示穩定度。",
    ),
    "distribution": (
        "顯示報酬分布與尾部風險。",
        "注意偏態、厚尾以及極端正負報酬，短樣本不可強推論。",
    ),
    "portfolio": (
        "顯示實際持倉曝險、集中度、換手或交易成本。",
        "數值越高通常代表槓桿、單一持倉或交易摩擦風險越大。",
    ),
    "round_trip": (
        "以已完成的買賣配對輔助觀察單筆交易。",
        "這是 portfolio return 的補充，不可取代組合層級績效。",
    ),
    "period": (
        "分開比較 Backtest、Historical OOS 與 Fresh OOS。",
        "不得把期間接起來掩蓋樣本差異，缺少的期間保持不可用。",
    ),
}


@dataclass(frozen=True)
class ChartResult:
    chart_id: str
    title: str
    status: str
    path: Path | None
    reason: str | None
    interpretation: dict[str, Any]


@dataclass
class PerformanceCharts:
    results: list[ChartResult] = field(default_factory=list)

    @property
    def generated(self) -> list[ChartResult]:
        return [result for result in self.results if result.status == "GENERATED"]

    def interpretations(self) -> list[dict[str, Any]]:
        return [result.interpretation for result in self.results]

    @classmethod
    def build(
        cls,
        data: dict[str, PerformanceData],
        metrics: dict[str, PerformanceMetrics],
        output_dir: Path,
        selected_periods: tuple[str, ...] = PERIODS,
    ) -> PerformanceCharts:
        output_dir.mkdir(parents=True, exist_ok=True)
        primary = next(
            (
                period
                for period in selected_periods
                if metrics[period].status == "AVAILABLE"
            ),
            None,
        )
        results = []
        for chart_id, title in CHARTS:
            path = output_dir / f"{chart_id}.png"
            draw, reason, key_numbers = _chart_drawer(
                chart_id, primary, data, metrics, selected_periods
            )
            if draw is None:
                path.unlink(missing_ok=True)
                status = "UNAVAILABLE"
                chart_path = None
            else:
                status, reason = _save_chart(path, title, draw)
                chart_path = path if status == "GENERATED" else None
            interpretation = _interpretation(
                chart_id,
                title,
                status,
                reason,
                key_numbers,
                primary,
                metrics,
            )
            results.append(
                ChartResult(
                    chart_id,
                    title,
                    status,
                    chart_path,
                    reason,
                    interpretation,
                )
            )
        return cls(results)


@dataclass
class PerformanceReport:
    identity: FrozenIdentity
    data: dict[str, PerformanceData]
    metrics: dict[str, PerformanceMetrics]
    charts: PerformanceCharts
    summary: dict[str, Any]
    output_dir: Path
    files: dict[str, Path]


def _save_chart(
    path: Path, title: str, draw: Callable[[Any, Any], None]
) -> tuple[str, str | None]:
    figure = None
    try:
        figure, axis = plt.subplots(figsize=(10, 5))
        draw(figure, axis)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(path, dpi=130, bbox_inches="tight")
        return "GENERATED", None
    except Exception as exc:
        path.unlink(missing_ok=True)
        return "UNAVAILABLE", f"{type(exc).__name__}: {exc}"
    finally:
        if figure is not None:
            plt.close(figure)


def _metric(metrics: PerformanceMetrics, name: str) -> Any:
    item = metrics.metrics.get(name)
    return item.value if item is not None and item.status == "AVAILABLE" else None


def _chart_drawer(
    chart_id: str,
    primary: str | None,
    data: dict[str, PerformanceData],
    metrics: dict[str, PerformanceMetrics],
    selected: tuple[str, ...],
) -> tuple[Callable[[Any, Any], None] | None, str | None, dict[str, Any]]:
    key_numbers: dict[str, Any] = {}
    if chart_id.startswith(("22_", "23_", "24_")):
        return _period_chart(chart_id, metrics, selected)
    if primary is None:
        return None, "NO_SELECTED_PERIOD_WITH_CANONICAL_RETURNS", key_numbers
    current = metrics[primary]
    source = current.series
    key_numbers["period"] = primary
    if chart_id == "01_cumulative_returns":
        series = source["cumulative_returns"]
        key_numbers["total_return"] = _metric(current, "total_return")
        return _line(series, "Cumulative return"), None, key_numbers
    if chart_id == "02_strategy_vs_benchmark":
        benchmark = source.get("benchmark")
        if not isinstance(benchmark, pd.Series):
            return None, "BENCHMARK_UNAVAILABLE", key_numbers
        strategy = (1.0 + data[primary].returns.reindex(benchmark.index)).cumprod() - 1
        benchmark_curve = (1.0 + benchmark).cumprod() - 1
        key_numbers.update(
            {
                "strategy_return": _metric(current, "total_return"),
                "benchmark_return": _metric(current, "benchmark_return"),
                "excess_return": _metric(current, "excess_return"),
            }
        )
        return _multi_line(
            {"Strategy": strategy, "Benchmark": benchmark_curve},
            "Cumulative return",
        ), None, key_numbers
    if chart_id == "03_log_cumulative_returns":
        equity = source["cumulative_returns"] + 1.0
        if bool((equity <= 0).any()):
            return None, "NON_POSITIVE_EQUITY_CANNOT_USE_LOG_SCALE", key_numbers
        return _line(equity, "Growth of 1", log=True), None, key_numbers
    if chart_id == "04_drawdown_underwater":
        key_numbers["max_drawdown"] = _metric(current, "max_drawdown")
        return _filled_line(source["drawdown"], "Drawdown"), None, key_numbers
    if chart_id == "05_top_drawdowns":
        if not current.drawdowns:
            return None, "NO_DRAWDOWN_EPISODES", key_numbers
        key_numbers["top_drawdowns"] = current.drawdowns
        return _drawdown_bars(current.drawdowns), None, key_numbers
    rolling = {
        "06_rolling_returns": ("rolling_return", "Rolling return"),
        "07_rolling_sharpe": ("rolling_sharpe", "Rolling Sharpe"),
        "08_rolling_volatility": ("rolling_volatility", "Annualized volatility"),
        "09_rolling_beta": ("rolling_beta", "Rolling beta"),
    }
    if chart_id in rolling:
        name, label = rolling[chart_id]
        series = source.get(name)
        if not isinstance(series, pd.Series) or series.dropna().empty:
            reason = (
                "BENCHMARK_OR_WINDOW_UNAVAILABLE"
                if chart_id == "09_rolling_beta"
                else f"REQUIRES_{DEFAULT_ROLLING_WINDOW}_OBSERVATIONS"
            )
            return None, reason, key_numbers
        summary_name = name.removeprefix("rolling_")
        key_numbers.update(current.rolling[summary_name])
        return _line(series.dropna(), label), None, key_numbers
    if chart_id == "10_monthly_returns_heatmap":
        monthly = source["monthly_returns"]
        key_numbers["positive_month_ratio"] = _metric(
            current, "positive_month_ratio"
        )
        return _monthly_heatmap(monthly), None, key_numbers
    if chart_id == "11_annual_returns":
        annual = source["annual_returns"]
        return _bar(annual, "Annual return"), None, key_numbers
    if chart_id == "12_return_distribution":
        monthly = source["monthly_returns"]
        return _histogram(monthly, "Monthly return"), None, key_numbers
    if chart_id == "13_daily_return_distribution":
        returns = source["returns"]
        key_numbers.update(
            {
                "best_day": _metric(current, "best_day"),
                "worst_day": _metric(current, "worst_day"),
                "skew": _metric(current, "skew"),
                "kurtosis": _metric(current, "kurtosis"),
            }
        )
        return _histogram(returns, "Daily return"), None, key_numbers
    if chart_id == "14_qq_plot":
        returns = source["returns"]
        if len(returns) < 8:
            return None, "QQ_PLOT_REQUIRES_8_OBSERVATIONS", key_numbers
        return _qq_plot(returns), None, key_numbers
    simple_series = {
        "15_gross_exposure": ("gross_exposure", "Gross exposure"),
        "16_net_exposure": ("net_exposure", "Net exposure"),
        "17_position_concentration": (
            "position_concentration",
            "HHI = sum(weight²)",
        ),
        "18_turnover": ("turnover", "Turnover"),
        "19_transaction_cost": ("transaction_cost", "Trading cost"),
    }
    if chart_id in simple_series:
        name, label = simple_series[chart_id]
        series = source.get(name)
        if not isinstance(series, pd.Series) or series.dropna().empty:
            reason = (
                "POSITIONS_UNAVAILABLE"
                if chart_id.startswith(("15_", "16_", "17_"))
                else "TRANSACTIONS_UNAVAILABLE"
            )
            return None, reason, key_numbers
        key_numbers[name] = _metric(current, name)
        return _line(series.dropna(), label), None, key_numbers
    if chart_id in ("20_round_trip_returns", "21_round_trip_duration"):
        trips = source.get("round_trips")
        if not isinstance(trips, pd.DataFrame) or trips.empty:
            return None, "CLOSED_ROUND_TRIPS_UNAVAILABLE", key_numbers
        column = "return" if chart_id.startswith("20_") else "duration"
        label = "Round-trip return" if column == "return" else "Holding days"
        return _histogram(trips[column], label), None, key_numbers
    return None, "CHART_IMPLEMENTATION_UNAVAILABLE", key_numbers


def _period_chart(
    chart_id: str,
    metrics: dict[str, PerformanceMetrics],
    selected: tuple[str, ...],
) -> tuple[Callable[[Any, Any], None] | None, str | None, dict[str, Any]]:
    required = {
        "22_backtest_vs_oos": ("BACKTEST", "HISTORICAL_OOS"),
        "23_historical_oos_vs_fresh_oos": ("HISTORICAL_OOS", "FRESH_OOS"),
        "24_all_period_comparison": PERIODS,
    }[chart_id]
    available = [
        period
        for period in required
        if period in selected and metrics[period].status == "AVAILABLE"
    ]
    if len(available) != len(required):
        missing = [period for period in required if period not in available]
        return None, f"PERIOD_DATA_UNAVAILABLE: {', '.join(missing)}", {}
    values = {
        period: float(_metric(metrics[period], "total_return"))
        for period in required
    }
    return _comparison_bar(values), None, {"total_return": values}


def _line(series: pd.Series, ylabel: str, log: bool = False) -> Callable:
    def draw(_figure: Any, axis: Any) -> None:
        axis.plot(series.index, series.values)
        axis.set_ylabel(ylabel)
        if log:
            axis.set_yscale("log")

    return draw


def _filled_line(series: pd.Series, ylabel: str) -> Callable:
    def draw(_figure: Any, axis: Any) -> None:
        axis.plot(series.index, series.values, color="#b22222")
        axis.fill_between(series.index, series.values, 0, color="#f4a6a6")
        axis.set_ylabel(ylabel)

    return draw


def _multi_line(series: dict[str, pd.Series], ylabel: str) -> Callable:
    def draw(_figure: Any, axis: Any) -> None:
        for name, values in series.items():
            axis.plot(values.index, values.values, label=name)
        axis.set_ylabel(ylabel)
        axis.legend(loc="best")

    return draw


def _bar(series: pd.Series, ylabel: str) -> Callable:
    def draw(_figure: Any, axis: Any) -> None:
        labels = [str(item.year) for item in series.index]
        axis.bar(labels, series.values)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel(ylabel)

    return draw


def _comparison_bar(values: dict[str, float]) -> Callable:
    def draw(_figure: Any, axis: Any) -> None:
        axis.bar(list(values), list(values.values()))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Total return")

    return draw


def _drawdown_bars(rows: list[dict[str, Any]]) -> Callable:
    def draw(_figure: Any, axis: Any) -> None:
        axis.bar([str(row["rank"]) for row in rows], [row["drawdown"] for row in rows])
        axis.set_xlabel("Rank")
        axis.set_ylabel("Drawdown")

    return draw


def _histogram(series: pd.Series, xlabel: str) -> Callable:
    def draw(_figure: Any, axis: Any) -> None:
        values = pd.Series(series).dropna()
        bins = min(30, max(5, int(np.sqrt(len(values)))))
        axis.hist(values, bins=bins, edgecolor="white")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")

    return draw


def _qq_plot(series: pd.Series) -> Callable:
    def draw(_figure: Any, axis: Any) -> None:
        stats.probplot(series.dropna().to_numpy(), dist="norm", plot=axis)
        axis.set_ylabel("Observed quantiles")

    return draw


def _monthly_heatmap(series: pd.Series) -> Callable:
    def draw(figure: Any, axis: Any) -> None:
        frame = series.to_frame("return")
        frame["year"] = frame.index.year
        frame["month"] = frame.index.month
        pivot = frame.pivot(index="year", columns="month", values="return")
        image = axis.imshow(pivot, aspect="auto", cmap="RdYlGn")
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        axis.set_xticks(range(12), range(1, 13))
        axis.set_xlabel("Month")
        figure.colorbar(image, ax=axis, label="Return")

    return draw


def _chart_group(chart_id: str) -> str:
    number = int(chart_id[:2])
    if number in (1, 3):
        return "cumulative"
    if number == 2:
        return "benchmark"
    if number in (4, 5):
        return "drawdown"
    if 6 <= number <= 9:
        return "rolling"
    if number in (10, 11):
        return "calendar"
    if 12 <= number <= 14:
        return "distribution"
    if 15 <= number <= 19:
        return "portfolio"
    if number in (20, 21):
        return "round_trip"
    return "period"


def _regime_observations(metrics: PerformanceMetrics) -> list[str]:
    returns = metrics.series.get("returns")
    if not isinstance(returns, pd.Series) or len(returns) < 21:
        return []
    rolling = (1.0 + returns).rolling(21).apply(np.prod, raw=True).sub(1.0)
    observations = []
    rules = (
        (rolling.abs() <= 0.02, "21-session flat period", rolling.abs().idxmin()),
        (rolling >= 0.10, "21-session rapid gain", rolling.idxmax()),
        (rolling <= -0.10, "21-session large loss", rolling.idxmin()),
    )
    for mask, label, date in rules:
        if bool(mask.any()) and pd.notna(date):
            value = float(rolling.loc[date])
            observations.append(
                f"{label}: ending {date.date().isoformat()}, return={value:.2%}"
            )
    if metrics.drawdowns:
        recovered = next(
            (
                row
                for row in metrics.drawdowns
                if row["recovery"] != "NOT_RECOVERED"
            ),
            None,
        )
        if recovered:
            observations.append(
                "recovery period: "
                f"{recovered['trough']} to {recovered['recovery']}"
            )
    return observations


def _interpretation(
    chart_id: str,
    title: str,
    status: str,
    reason: str | None,
    key_numbers: dict[str, Any],
    primary: str | None,
    metrics: dict[str, PerformanceMetrics],
) -> dict[str, Any]:
    group = _chart_group(chart_id)
    what, how = CHART_HELP[group]
    observations = []
    if status == "GENERATED" and primary is not None and group == "cumulative":
        observations = _regime_observations(metrics[primary])
    if status == "GENERATED" and not observations:
        observations = ["圖表已由可追溯的 canonical 資料生成。"]
    warning = None
    if primary is not None and "TAIL_RISK_SAMPLE_WARNING" in metrics[primary].warnings:
        if group == "distribution":
            warning = "TAIL_RISK_SAMPLE_WARNING"
    interpretation = (
        f"此圖使用 {primary} canonical evidence；結論限於該期間。"
        if status == "GENERATED"
        else f"無法解讀：{reason}。未建立空白或推測圖表。"
    )
    return {
        "chart_id": chart_id,
        "title": title,
        "status": status,
        "reason": reason,
        "what_it_is": what,
        "how_to_read": how,
        "observations": observations,
        "key_numbers": key_numbers,
        "interpretation": interpretation,
        "warning": warning,
        "sections": {
            "1_這張圖在看什麼": what,
            "2_怎麼看": how,
            "3_目前數值是多少": key_numbers,
            "4_圖上發生了什麼": observations,
            "5_對策略代表什麼": interpretation,
        },
    }


def classify_performance_evidence(
    data: dict[str, PerformanceData],
    metrics: dict[str, PerformanceMetrics],
) -> str:
    fresh = data["FRESH_OOS"]
    if (
        not fresh.available
        or fresh.returns is None
        or len(fresh.returns) < EVIDENCE_RULES["minimum_fresh_oos_observations"]
    ):
        return "INSUFFICIENT_DATA"
    values = metrics["FRESH_OOS"]
    total = _metric(values, "total_return")
    sharpe = _metric(values, "sharpe")
    drawdown = _metric(values, "max_drawdown")
    excess = _metric(values, "excess_return")
    if total is None or sharpe is None or drawdown is None:
        return "INSUFFICIENT_DATA"
    supportive = EVIDENCE_RULES["supportive"]
    if (
        total > supportive["total_return_min"]
        and sharpe > supportive["sharpe_min"]
        and drawdown >= supportive["max_drawdown_min"]
        and (excess is None or excess >= supportive["excess_return_min_when_available"])
    ):
        return "SUPPORTIVE"
    adverse = EVIDENCE_RULES["adverse"]
    if (
        (total <= adverse["total_return_max"] and sharpe <= adverse["sharpe_max"])
        or drawdown <= adverse["max_drawdown_max"]
    ):
        return "ADVERSE"
    return "MIXED"


def _coverage(data: PerformanceData) -> dict[str, Any]:
    returns = data.returns
    return {
        "status": data.status,
        "reason": data.reason,
        "start": (
            returns.index.min().date().isoformat() if returns is not None else None
        ),
        "end": returns.index.max().date().isoformat() if returns is not None else None,
        "observations": len(returns) if returns is not None else 0,
        "benchmark": data.benchmark,
        "positions": "AVAILABLE" if data.positions is not None else "UNAVAILABLE",
        "transactions": data.transaction_status,
        "timezone_policy": data.timezone_policy,
    }


def _summary(
    identity: FrozenIdentity,
    data: dict[str, PerformanceData],
    metrics: dict[str, PerformanceMetrics],
    charts: PerformanceCharts,
    gate: dict[str, Any],
) -> dict[str, Any]:
    evidence = classify_performance_evidence(data, metrics)
    available = next(
        (period for period in PERIODS if metrics[period].status == "AVAILABLE"),
        None,
    )
    primary = metrics[available].metrics if available else {}
    return {
        "report_status": "PASS",
        "strategy_id": identity.strategy_id,
        "strategy_fingerprint": identity.strategy_fingerprint,
        "generated_at": datetime.now(UTC).isoformat(),
        "metric_source": "backtest.robustness.compute_metrics + diagnostics",
        "pyfolio_role": "diagnostic_visualization_only",
        "pyfolio_version": _dependency_version("pyfolio-reloaded"),
        "classification_policy": EVIDENCE_RULES,
        "periods": {
            period.lower(): {
                "coverage": _coverage(data[period]),
                **metrics[period].to_dict(),
            }
            for period in PERIODS
        },
        "benchmark": {
            period.lower(): data[period].benchmark for period in PERIODS
        },
        "risk": {
            name: primary[name].to_dict()
            for name in (
                "annualized_volatility",
                "sharpe",
                "sortino",
                "var_95",
                "cvar_95",
            )
            if name in primary
        },
        "drawdown": metrics[available].drawdowns if available else [],
        "turnover": (
            primary.get("turnover", _missing_summary()).to_dict()
            if available
            else _missing_summary().to_dict()
        ),
        "costs": {
            name: primary[name].to_dict()
            for name in ("transaction_cost", "cost_drag", "gross_return", "net_return")
            if name in primary
        },
        "concentration": {
            "formula": "HHI = sum(abs(position_weight)^2)",
            **{
                name: primary[name].to_dict()
                for name in (
                    "largest_position_weight",
                    "average_largest_position_weight",
                    "top_3_weight",
                    "top_5_weight",
                    "hhi",
                )
                if name in primary
            },
        },
        "charts": {
            "generated": [item.chart_id for item in charts.generated],
            "unavailable": [
                {"chart": item.chart_id, "reason": item.reason}
                for item in charts.results
                if item.status != "GENERATED"
            ],
        },
        "fresh_oos_status": data["FRESH_OOS"].status,
        "fresh_oos_reason": data["FRESH_OOS"].reason,
        "performance_evidence": evidence,
        "promotion_gate": {
            "status": gate.get("production_eligibility", "UNKNOWN"),
            "production_ready": bool(gate.get("production_ready", False)),
            "reason": gate.get("production_block_reason"),
        },
        "reports": {},
    }


def _missing_summary() -> Any:
    from .performance_metrics import MetricValue

    return MetricValue(None, "UNAVAILABLE", "NO_AVAILABLE_PERIOD")


def _dependency_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_metrics_csv(
    path: Path, metrics: dict[str, PerformanceMetrics]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["period", "metric", "value", "status"])
        for period in PERIODS:
            for name in METRIC_NAMES:
                item = metrics[period].metrics[name]
                writer.writerow([period, name, item.value, item.status])


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _dashboard(summary: dict[str, Any]) -> str:
    period = next(
        (
            payload
            for payload in summary["periods"].values()
            if payload["status"] == "AVAILABLE"
        ),
        None,
    )
    metrics = period["metrics"] if period else {}
    rows = [
        ("Strategy", summary["strategy_id"]),
        (
            "Period",
            "N/A"
            if period is None
            else f"{period['coverage']['start']} → {period['coverage']['end']}",
        ),
    ]
    for name in (
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "annualized_volatility",
        "benchmark_return",
        "benchmark_cagr",
        "excess_return",
        "alpha",
        "beta",
    ):
        rows.append((name, metrics.get(name, {}).get("value")))
    rows.extend(
        [
            ("Historical Evidence", summary.get("historical_evidence")),
            ("Fresh OOS", summary["fresh_oos_status"]),
            ("Promotion Gate", summary["promotion_gate"]["status"]),
        ]
    )
    return "".join(
        "<div class='card'><span>"
        f"{html.escape(label)}</span><strong>{html.escape(_format_value(value))}</strong>"
        "</div>"
        for label, value in rows
    )


def _period_table(summary: dict[str, Any]) -> str:
    names = (
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "annualized_volatility",
        "turnover",
    )
    rows = []
    for name in names:
        cells = [f"<th>{html.escape(name)}</th>"]
        for period in PERIODS:
            payload = summary["periods"][period.lower()]["metrics"][name]
            cells.append(f"<td>{html.escape(_format_value(payload['value']))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<table><thead><tr><th>Metric</th>"
        + "".join(f"<th>{period}</th>" for period in PERIODS)
        + f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _annual_table(summary: dict[str, Any]) -> str:
    available = next(
        (
            (name, payload)
            for name, payload in summary["periods"].items()
            if payload["status"] == "AVAILABLE"
        ),
        None,
    )
    if available is None:
        return "<p class='unavailable'>UNAVAILABLE — annual returns unavailable.</p>"
    name, payload = available
    rows = "".join(
        f"<tr><td>{item['year']}</td><td>{item['value']:.6f}</td></tr>"
        for item in payload["calendar"]["annual_returns"]
    )
    return (
        f"<p>Period: {html.escape(name.upper())}</p>"
        f"<table><thead><tr><th>Year</th><th>Return</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _chart_html(charts: PerformanceCharts) -> str:
    blocks = []
    for result in charts.results:
        item = result.interpretation
        image = (
            f"<img src='charts/{result.path.name}' alt='{html.escape(result.title)}'>"
            if result.path is not None
            else f"<p class='unavailable'>UNAVAILABLE — {html.escape(str(result.reason))}</p>"
        )
        blocks.append(
            f"<article><h3>{html.escape(result.title)}</h3>{image}"
            f"<p><b>這張圖在看什麼？</b> {html.escape(item['what_it_is'])}</p>"
            f"<p><b>怎麼看？</b> {html.escape(item['how_to_read'])}</p>"
            f"<p><b>對策略代表什麼？</b> "
            f"{html.escape(item['interpretation'])}</p></article>"
        )
    return "".join(blocks)


def _render_html(summary: dict[str, Any], charts: PerformanceCharts) -> str:
    fresh_reason = summary["fresh_oos_reason"] or "N/A"
    limitations = [
        f"{period.upper()}: {payload['coverage']['reason']}"
        for period, payload in summary["periods"].items()
        if payload["coverage"]["status"] != "AVAILABLE"
    ]
    sections = [
        ("01 Executive Summary", "<div class='dashboard'>" + _dashboard(summary) + "</div>"),
        ("02 Strategy Definition", f"<p>{html.escape(summary['strategy_id'])}<br>Fingerprint: {html.escape(summary['strategy_fingerprint'])}</p>"),
        ("03 Data Coverage", _period_table(summary)),
        ("04 Core Performance", _period_table(summary)),
        ("05 Benchmark Comparison", "<p>Canonical benchmark only; unavailable sources are not substituted.</p>"),
        ("06 Cumulative Return", ""),
        ("07 Drawdown", "<p>Top drawdowns include start, trough, recovery, depth and session duration when available.</p>"),
        ("08 Rolling Performance", f"<p>Central window: {DEFAULT_ROLLING_WINDOW} sessions.</p>"),
        ("09 Monthly / Annual Performance", _annual_table(summary)),
        ("10 Return Distribution", ""),
        ("11 Tail Risk", "<p>VaR/CVaR use the 5% empirical tail; short samples are explicitly warned.</p>"),
        ("12 Portfolio Exposure", ""),
        ("13 Concentration", "<p>HHI = sum(abs(position_weight)²).</p>"),
        ("14 Turnover", ""),
        ("15 Trading Cost", "<p>Only canonical transaction costs are accepted.</p>"),
        ("16 Transactions", "<p>No position-difference transaction fabrication.</p>"),
        ("17 Round Trips", "<p>Auxiliary only; portfolio-level return remains authoritative.</p>"),
        ("18 Backtest vs OOS", _period_table(summary)),
        ("19 Fresh OOS", f"<p>Status: {html.escape(summary['fresh_oos_status'])}<br>Reason: {html.escape(fresh_reason)}</p>"),
        ("20 Limitations", "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in limitations) + "</ul>"),
        ("21 Performance Conclusion", f"<p>Performance Evidence Classification: <strong>{html.escape(summary['performance_evidence'])}</strong></p><p>Production Ready: {'YES' if summary['promotion_gate']['production_ready'] else 'NO'}</p>"),
    ]
    rendered = []
    chart_html = _chart_html(charts)
    for title, body in sections:
        if title.startswith("06 "):
            body += chart_html
        rendered.append(f"<section><h2>{title}</h2>{body}</section>")
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>Performance Tear Sheet</title>
<style>
body{{font-family:"Noto Sans CJK TC","Microsoft JhengHei",sans-serif;max-width:1200px;margin:auto;padding:24px;color:#222}}
.dashboard{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;display:flex;flex-direction:column}}
.card span{{color:#666;font-size:.85rem}} .card strong{{overflow-wrap:anywhere}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{border:1px solid #ddd;padding:6px;text-align:right}} th:first-child{{text-align:left}}
img{{max-width:100%;height:auto}} article{{break-inside:avoid;border-top:1px solid #ddd;margin-top:18px}} .unavailable{{color:#8b0000}}
@media print{{section,article{{break-inside:avoid}} h2{{break-after:avoid}} body{{padding:0}}}}
</style></head><body><h1>Comprehensive Performance Tear Sheet</h1>
{''.join(rendered)}</body></html>"""


def _write_final_validation(
    path: Path, summary: dict[str, Any]
) -> None:
    periods = summary["periods"]
    analysis = {
        "Backtest Metrics": periods["backtest"]["status"],
        "Historical OOS": periods["historical_oos"]["status"],
        "Fresh OOS": periods["fresh_oos"]["status"],
        "Benchmark Comparison": (
            "AVAILABLE"
            if any(
                value.get("benchmark_id")
                for value in summary["benchmark"].values()
            )
            else "UNAVAILABLE"
        ),
        "Drawdown Analysis": (
            "AVAILABLE" if summary["drawdown"] else "UNAVAILABLE"
        ),
        "Cost Analysis": (
            summary["costs"].get("transaction_cost", {}).get(
                "status", "UNAVAILABLE"
            )
        ),
    }
    benchmark = periods["backtest"]["metrics"]
    benchmark_meta = summary["benchmark"].get("backtest", {})
    lines = [
        "# Final Strategy Validation Report",
        "",
        f"STRATEGY_ID = {summary['strategy_id']}",
        f"STRATEGY_FINGERPRINT = {summary['strategy_fingerprint']}",
        "",
        "## Performance Analysis",
        "",
        "Performance Tear Sheet = PASS",
        *[f"{name} = {status}" for name, status in analysis.items()],
        "",
        "## Benchmark Comparison",
        "",
        f"Benchmark ID = {benchmark_meta.get('benchmark_id') or 'N/A'}",
        f"Benchmark Period = {benchmark_meta.get('benchmark_start') or 'N/A'} -> {benchmark_meta.get('benchmark_end') or 'N/A'}",
        f"Benchmark Return = {benchmark.get('benchmark_return', {}).get('value')}",
        f"Benchmark CAGR = {benchmark.get('benchmark_cagr', {}).get('value')}",
        f"Strategy Excess Return = {benchmark.get('excess_return', {}).get('value')}",
        f"Alpha = {benchmark.get('alpha', {}).get('value')}",
        f"Beta = {benchmark.get('beta', {}).get('value')}",
        f"Tracking Error = {benchmark.get('tracking_error', {}).get('value')}",
        f"Information Ratio = {benchmark.get('information_ratio', {}).get('value')}",
        "",
        "## Risk Overlay Research Status",
        "",
        "REB60 MA60 Breadth = ADVERSE",
        "Daily/T+1 MA60 Breadth = ADVERSE",
        "Risk Overlay Route = CLOSED_NOT_ADOPTED",
        "Base Strategy = UNCHANGED",
        "Further Historical Tuning = CLOSED",
        "",
        f"Performance Evidence = {summary['performance_evidence']}",
        "Historical Evidence = "
        + str(summary.get("historical_evidence", "INSUFFICIENT_DATA")),
        "Historical OOS = "
        + str(periods["historical_oos"]["status"]),
        f"Fresh OOS Reason = {summary['fresh_oos_reason']}",
        "",
        "## Production Gate",
        "",
        f"Status = {summary['promotion_gate']['status']}",
        f"Reason = {summary['promotion_gate']['reason']}",
        "Production Ready = "
        + ("YES" if summary["promotion_gate"]["production_ready"] else "NO"),
        "",
        "Performance reporting does not override Fresh OOS eligibility or promotion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pdf(html_path: Path, pdf_path: Path) -> dict[str, Any]:
    errors = []
    try:
        from weasyprint import HTML

        HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(pdf_path)
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise RuntimeError("PDF output is empty")
        return {"status": "GENERATED", "reason": None, "backend": "weasyprint"}
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        errors.append(f"weasyprint: {type(exc).__name__}: {exc}")
    candidates = (
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    browser = next(
        (Path(value) for value in candidates if value and Path(value).is_file()),
        None,
    )
    if browser is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="performance-pdf-") as profile:
                subprocess.run(
                    [
                        str(browser),
                        "--headless=new",
                        "--disable-gpu",
                        "--no-pdf-header-footer",
                        f"--user-data-dir={profile}",
                        f"--print-to-pdf={pdf_path}",
                        html_path.resolve().as_uri(),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                raise RuntimeError("PDF output is empty")
            return {
                "status": "GENERATED",
                "reason": None,
                "backend": browser.name,
            }
        except Exception as exc:
            if _valid_pdf(pdf_path):
                return {
                    "status": "GENERATED",
                    "reason": (
                        f"browser emitted a valid PDF before {type(exc).__name__}"
                    ),
                    "backend": browser.name,
                }
            pdf_path.unlink(missing_ok=True)
            errors.append(f"browser: {type(exc).__name__}: {exc}")
    return {"status": "UNAVAILABLE", "reason": "; ".join(errors)}


def _valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        header = handle.read(5)
        handle.seek(max(0, path.stat().st_size - 2048))
        trailer = handle.read()
    return header == b"%PDF-" and b"%%EOF" in trailer


def _gate(root: Path) -> dict[str, Any]:
    path = (
        root
        / "data/research/fundamental-production-final-v1"
        / "production_final_gate.json"
    )
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _publish(staging: Path, output: Path, force: bool) -> dict[str, Path]:
    if (output / "performance_summary.json").exists() and not force:
        raise FileExistsError("performance report exists; use --force to regenerate")
    output.mkdir(parents=True, exist_ok=True)
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    if force:
        for stale in charts.glob("*.png"):
            stale.unlink()
        pdf = output / "performance_tearsheet_report.pdf"
        if not (staging / pdf.name).exists():
            pdf.unlink(missing_ok=True)
    files: dict[str, Path] = {}
    for source in staging.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(staging)
        if relative.as_posix() == "final_strategy_validation_report.md":
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative.as_posix()] = target
    return files


def _manifest(staging: Path, identity: FrozenIdentity) -> dict[str, Any]:
    return {
        "strategy_id": identity.strategy_id,
        "strategy_fingerprint": identity.strategy_fingerprint,
        "artifact_sha256": {
            path.relative_to(staging).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(staging.rglob("*"))
            if path.is_file()
            and path.name
            not in {
                "performance_run_manifest.json",
                "final_strategy_validation_report.md",
            }
        },
        "manifest_self_hash_excluded": True,
    }


def generate_performance_report(
    root: str | Path,
    *,
    period: str = "all",
    output_format: str = "all",
    force: bool = False,
    data_override: dict[str, PerformanceData] | None = None,
) -> PerformanceReport:
    """Generate formal outputs without altering Strategy Runtime or its inputs."""

    root = Path(root).resolve()
    requested = period.lower()
    period_map = {
        "all": PERIODS,
        "backtest": ("BACKTEST",),
        "historical-oos": ("HISTORICAL_OOS",),
        "fresh-oos": ("FRESH_OOS",),
    }
    if requested not in period_map:
        raise ValueError(f"unsupported period: {period}")
    if output_format not in ("html", "pdf", "all"):
        raise ValueError(f"unsupported format: {output_format}")
    identity_before = load_frozen_identity(root)
    data = data_override or load_repository_performance_data(root, identity_before)
    if set(data) != set(PERIODS):
        raise ValueError(f"performance data must contain exactly {PERIODS}")
    metrics = {name: calculate_metrics(data[name]) for name in PERIODS}
    final_dir = root / "data/research/fundamental-production-final-v1"
    output = root / OUTPUT_NAMESPACE
    final_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="performance-", dir=final_dir) as temp:
        staging = Path(temp)
        charts = PerformanceCharts.build(
            data, metrics, staging / "charts", period_map[requested]
        )
        summary = _summary(identity_before, data, metrics, charts, _gate(root))
        historical_path = (
            final_dir
            / "historical-evidence"
            / "historical_strategy_validation.json"
        )
        if historical_path.exists():
            historical = json.loads(historical_path.read_text(encoding="utf-8"))
            summary["historical_evidence"] = historical.get(
                "historical_evidence", {}
            ).get("classification", "INSUFFICIENT_DATA")
        else:
            summary["historical_evidence"] = "INSUFFICIENT_DATA"
        _write_json(
            staging / "chart_interpretations.json", charts.interpretations()
        )
        _write_metrics_csv(staging / "performance_metrics.csv", metrics)
        html_path = staging / "performance_tearsheet_report.html"
        html_path.write_text(_render_html(summary, charts), encoding="utf-8")
        summary["reports"]["html"] = {
            "status": "GENERATED",
            "canonical": True,
        }
        pdf_path = staging / "performance_tearsheet_report.pdf"
        summary["reports"]["pdf"] = (
            _pdf(html_path, pdf_path)
            if output_format in ("pdf", "all")
            else {"status": "NOT_REQUESTED", "reason": None}
        )
        _write_json(staging / "performance_summary.json", summary)
        _write_final_validation(
            staging / "final_strategy_validation_report.md", summary
        )
        _write_json(
            staging / "performance_run_manifest.json",
            _manifest(staging, identity_before),
        )
        identity_after = load_frozen_identity(root)
        if identity_after != identity_before:
            blocked = final_dir / "performance_report_blocked.json"
            _write_json(
                blocked,
                {"report_status": "BLOCKED", "reason": "FROZEN_STRATEGY_CHANGED"},
            )
            raise FrozenStrategyChanged("FROZEN_STRATEGY_CHANGED")
        files = _publish(staging, output, force)
        final_validation = final_dir / "final_strategy_validation_report.md"
        shutil.copy2(
            staging / "final_strategy_validation_report.md", final_validation
        )
        files["../final_strategy_validation_report.md"] = final_validation
        charts.results = [
            ChartResult(
                item.chart_id,
                item.title,
                item.status,
                output / "charts" / item.path.name if item.path else None,
                item.reason,
                item.interpretation,
            )
            for item in charts.results
        ]
    return PerformanceReport(
        identity_before,
        data,
        metrics,
        charts,
        summary,
        output,
        files,
    )


__all__ = [
    "CHARTS",
    "EVIDENCE_RULES",
    "ChartResult",
    "PerformanceCharts",
    "PerformanceReport",
    "classify_performance_evidence",
    "generate_performance_report",
]

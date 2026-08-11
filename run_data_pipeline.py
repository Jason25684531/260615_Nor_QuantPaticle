from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from twse_factor_lab.data.manifest import append_manifest_entries, build_manifest_entry
from twse_factor_lab.data.normalizer import (
    normalize_ohlcv,
    normalize_universe,
    normalize_valuation,
)
from twse_factor_lab.data.parquet_store import ParquetStore
from twse_factor_lab.data.twse_client import TWSEClient
from twse_factor_lab.data.universe import (
    build_research_universe,
    build_universe_coverage,
)
from twse_factor_lab.data.yfinance_client import OhlcvDownloadResult, YFinanceClient
from twse_factor_lab.validation.ohlcv_integrity import sort_ohlcv, validate_ohlcv
from twse_factor_lab.validation.research_period import validate_research_periods


@dataclass(frozen=True)
class OhlcvSettings:
    ticker_limit: int | None
    batch_size: int
    retry: int
    sleep_seconds: float
    fail_fast: bool


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_root_for_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    if path.parent.name == "config":
        return path.parent.parent
    return path.parent


def resolve_path(config_path: str | Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root_for_config(config_path) / path


def missing_ratio(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    return {
        column: round(float(value), 4) for column, value in frame.isna().mean().items()
    }


def ohlcv_settings_from_config(data_config: dict[str, Any]) -> OhlcvSettings:
    ohlcv_config = data_config.get("ohlcv", {}) or {}
    configured_limit = ohlcv_config.get(
        "ticker_limit", data_config.get("ohlcv_ticker_limit", 100)
    )
    ticker_limit = None if configured_limit in (None, "all") else int(configured_limit)
    return OhlcvSettings(
        ticker_limit=ticker_limit,
        batch_size=max(1, int(ohlcv_config.get("batch_size", 20))),
        retry=max(0, int(ohlcv_config.get("retry", 3))),
        sleep_seconds=float(ohlcv_config.get("sleep_seconds", 1)),
        fail_fast=bool(ohlcv_config.get("fail_fast", False)),
    )


def universe_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Read deterministic universe defaults while rejecting unsupported measures."""

    universe = config.get("universe", {}) or {}
    liquidity = universe.get("liquidity", {}) or {}
    settings = {
        "membership": str(universe.get("membership", "current_listed_only")),
        "liquidity": {
            "enabled": bool(liquidity.get("enabled", True)),
            "window": int(liquidity.get("window", 20)),
            "measure": str(liquidity.get("measure", "median")),
            "minimum_traded_value": float(
                liquidity.get("minimum_traded_value", 50_000_000)
            ),
        },
    }
    if settings["membership"] != "current_listed_only":
        raise ValueError("Only current_listed_only membership is currently supported")
    if settings["liquidity"]["window"] < 1:
        raise ValueError("universe.liquidity.window must be positive")
    if settings["liquidity"]["measure"] != "median":
        raise ValueError("universe.liquidity.measure must be median")
    return settings


def select_ohlcv_tickers(
    universe: pd.DataFrame, *, ticker_limit: int | None
) -> list[str]:
    tickers = universe["ticker"].dropna().astype(str).drop_duplicates().tolist()
    if ticker_limit is None:
        return tickers
    return tickers[:ticker_limit]


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def download_ohlcv_with_retries(
    *,
    client: YFinanceClient,
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int,
    retry: int,
    sleep_seconds: float,
    fail_fast: bool,
) -> OhlcvDownloadResult:
    frames: list[pd.DataFrame] = []
    failed_final: list[str] = []

    for batch in _chunks(tickers, batch_size):
        remaining = list(batch)
        for attempt in range(retry + 1):
            result = client.download_ohlcv(
                tickers=remaining,
                start=start,
                end=end,
            )
            if not result.data.empty:
                frames.append(result.data)

            failed = [str(ticker) for ticker in result.failed_tickers]
            if not failed:
                remaining = []
                break
            remaining = failed
            if attempt < retry and sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if remaining:
            failed_final.extend(remaining)
            if fail_fast:
                raise RuntimeError(
                    "OHLCV download failed for tickers: "
                    + ", ".join(sorted(set(remaining)))
                )

    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return OhlcvDownloadResult(data=data, failed_tickers=sorted(set(failed_final)))


def build_quality_report(
    universe: pd.DataFrame,
    valuation: pd.DataFrame,
    ohlcv: pd.DataFrame,
    failed_tickers: list[str],
    *,
    configured_start_date: str = "N/A",
    configured_end_date: str = "N/A",
    ohlcv_ticker_subset_size: int = 0,
    ohlcv_source: str = "N/A",
    valuation_source: str = "N/A",
    configured_ticker_limit: int | None = None,
    ohlcv_requested_tickers: int | None = None,
    universe_coverage: pd.DataFrame | None = None,
    membership: str = "current_listed_only",
    minimum_universe_coverage: float | None = None,
    universe_scope: str = "FULL",
    research_config: dict[str, Any] | None = None,
    liquidity_config: dict[str, Any] | None = None,
    fundamental_coverage_report: str | Path | None = None,
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    ticker_count = int(universe["ticker"].nunique()) if "ticker" in universe else 0
    actual_ohlcv_ticker_count = (
        int(ohlcv["ticker"].nunique()) if not ohlcv.empty and "ticker" in ohlcv else 0
    )
    requested_tickers = (
        int(ohlcv_requested_tickers)
        if ohlcv_requested_tickers is not None
        else int(ohlcv_ticker_subset_size)
    )
    failed_count = len(set(failed_tickers))
    coverage_ratio = (
        float(actual_ohlcv_ticker_count / ticker_count) if ticker_count else 0.0
    )
    if not ohlcv.empty and "date" in ohlcv:
        date_min = str(ohlcv["date"].min().date())
        date_max = str(ohlcv["date"].max().date())
    else:
        date_min = "N/A"
        date_max = "N/A"
    coverage = universe_coverage if universe_coverage is not None else pd.DataFrame()
    listed_dates = universe.get("listed_date", pd.Series(dtype="datetime64[ns]"))
    missing_listed_date_count = int(listed_dates.isna().sum())
    liquidity = liquidity_config or {}
    liquidity_rule = (
        "disabled"
        if not liquidity.get("enabled", True)
        else (
            f"trailing {liquidity.get('window', 20)}-observation median "
            "adjusted close x reported volume > "
            f"{liquidity.get('minimum_traded_value', 50_000_000):g}"
        )
    )
    live_pipeline_status = (
        "PARTIAL" if failed_tickers or universe_scope == "PARTIAL" else "SUCCESS"
    )
    fundamental_report_status = "not generated"
    if fundamental_coverage_report and Path(fundamental_coverage_report).exists():
        fundamental_report_status = str(fundamental_coverage_report)
    if coverage.empty:
        coverage_lines = ["- No coverage rows"]
    else:
        latest = coverage.sort_values("date").iloc[-1]
        coverage_lines = [
            f"- coverage_date_rows: {len(coverage)}",
            f"- latest_total_listing_eligible: {int(latest['listing_eligible_count'])}",
            f"- latest_ohlcv_available: {int(latest['ohlcv_available_count'])}",
            f"- latest_liquidity_pass: {int(latest['liquidity_pass_count'])}",
            f"- latest_analysis_ready: {int(latest['analysis_ready_count'])}",
            f"- latest_analysis_ready_ratio: {latest['coverage_ratio']:.4f}",
            f"- mean_analysis_ready_ratio: {coverage['coverage_ratio'].mean():.4f}",
        ]

    lines = [
        "# Data Quality Summary",
        "",
        f"Generated at: {generated_at}",
        "",
        f"Configured date range: {configured_start_date} to {configured_end_date}",
        f"Actual OHLCV date range: {date_min} to {date_max}",
        f"OHLCV ticker subset size: {ohlcv_ticker_subset_size}",
        f"universe_total_tickers: {ticker_count}",
        f"ohlcv_requested_tickers: {requested_tickers}",
        f"ohlcv_successful_tickers: {actual_ohlcv_ticker_count}",
        f"ohlcv_failed_tickers: {failed_count}",
        f"ohlcv_coverage_ratio: {coverage_ratio:.4f}",
        f"configured_ticker_limit: {configured_ticker_limit}",
        f"actual_ohlcv_ticker_count: {actual_ohlcv_ticker_count}",
        "failed_yfinance_tickers: "
        + (", ".join(failed_tickers) if failed_tickers else "None"),
        f"Universe rows: {len(universe)}",
        f"Universe total count: {len(universe)}",
        f"Valuation rows: {len(valuation)}",
        f"OHLCV rows: {len(ohlcv)}",
        f"Ticker count: {ticker_count}",
        f"OHLCV date range: {date_min} to {date_max}",
        "",
        "## Sources And Limitations",
        "",
        f"- live_pipeline_status: {live_pipeline_status}",
        f"- OHLCV source: {ohlcv_source}",
        "- price_adjustment: auto_adjusted",
        "- volume_basis: reported_shares",
        f"- Valuation source: {valuation_source}",
        (
            "- market source: TWSE listed-company universe; if the source field is "
            "missing, normalization defaults market to TWSE."
        ),
        (
            "- valuation.date is empty because the current TWSE valuation snapshot "
            "endpoint does not return historical valuation dates."
        ),
        (
            "- data source limitations: OHLCV is a bounded yfinance fallback subset, "
            "while valuation data is latest snapshot data rather than "
            "point-in-time history."
        ),
        (
            "- survivorship bias warning: the current universe is a present-day listed "
            "universe and can bias historical research if used without a "
            "dated membership source."
        ),
        "",
        "## Research Universe Quality",
        "",
        f"- membership: {membership}",
        f"- membership_missing_listed_date_count: {missing_listed_date_count}",
        "- survivorship_disclosure: current_listed_only is not point-in-time.",
        f"- universe_scope: {universe_scope}",
        f"- liquidity_rule: {liquidity_rule}",
        "- liquidity_measure_source: proxy_close_times_volume",
        f"- minimum_universe_coverage: {minimum_universe_coverage}",
        f"- fundamental_coverage_report: {fundamental_report_status}",
        *coverage_lines,
        (
            f"- in_sample: {research_config.get('in_sample')}"
            if research_config
            else "- in_sample: not configured"
        ),
        (
            f"- out_of_sample: {research_config.get('out_of_sample')}"
            if research_config
            else "- out_of_sample: not configured"
        ),
        "",
        "## Missing Ratios",
        "",
    ]
    for name, frame in {
        "universe": universe,
        "valuation": valuation,
        "ohlcv": ohlcv,
    }.items():
        lines.append(f"### {name}")
        ratios = missing_ratio(frame)
        if not ratios:
            lines.append("- No rows")
        else:
            for column, value in ratios.items():
                lines.append(f"- {column}: {value:.4f}")
        lines.append("")

    lines.extend(
        [
            "## Failed yfinance tickers",
            "",
            ", ".join(failed_tickers) if failed_tickers else "None",
            "",
        ]
    )
    return "\n".join(lines)


def run_pipeline(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    data_config = config["data"]
    paths = config["paths"]
    twse_config = config["twse"]
    store = ParquetStore()

    twse_client = TWSEClient(
        base_url=twse_config["base_url"],
        timeout=int(twse_config.get("timeout_seconds", 30)),
    )
    yfinance_client = YFinanceClient()

    universe = normalize_universe(twse_client.fetch_dataframe("listed_companies"))
    valuation = normalize_valuation(twse_client.fetch_dataframe("valuation"))

    ohlcv_settings = ohlcv_settings_from_config(data_config)
    tickers = select_ohlcv_tickers(
        universe,
        ticker_limit=ohlcv_settings.ticker_limit,
    )
    download = download_ohlcv_with_retries(
        client=yfinance_client,
        tickers=tickers,
        start=data_config["start_date"],
        end=data_config["end_date"],
        batch_size=ohlcv_settings.batch_size,
        retry=ohlcv_settings.retry,
        sleep_seconds=ohlcv_settings.sleep_seconds,
        fail_fast=ohlcv_settings.fail_fast,
    )
    if download.data.empty:
        raise RuntimeError(
            "OHLCV download produced no usable rows; existing artifacts were not "
            "overwritten"
        )
    ohlcv = normalize_ohlcv(download.data)
    validate_ohlcv(ohlcv)
    ohlcv = sort_ohlcv(ohlcv)
    universe_settings = universe_settings_from_config(config)
    research_config = config.get("research")
    if research_config:
        validate_research_periods(
            research_config, ohlcv["date"].min(), ohlcv["date"].max()
        )
    research_universe = build_research_universe(
        universe, ohlcv, liquidity=universe_settings["liquidity"]
    )
    universe_coverage = build_universe_coverage(research_universe)
    minimum_coverage = float(
        (config.get("research_quality", {}) or {}).get("minimum_universe_coverage", 0.0)
    )
    universe_scope = "FULL"
    if (universe_coverage["coverage_ratio"] < minimum_coverage).any():
        universe_scope = "PARTIAL"
        warnings.warn(
            "Research-universe coverage is below minimum_universe_coverage; "
            "continuing with PARTIAL scope.",
            stacklevel=2,
        )

    output_paths = {
        "universe": resolve_path(config_path, paths["universe"]),
        "valuation": resolve_path(config_path, paths["valuation"]),
        "ohlcv": resolve_path(config_path, paths["ohlcv"]),
        "research_universe": resolve_path(config_path, paths["research_universe"]),
        "universe_coverage": resolve_path(config_path, paths["universe_coverage"]),
        "data_quality_report": resolve_path(config_path, paths["data_quality_report"]),
        "manifest": resolve_path(config_path, paths["manifest"]),
    }

    store.save(universe, output_paths["universe"])
    store.save(valuation, output_paths["valuation"])
    store.save(ohlcv, output_paths["ohlcv"])
    store.save(research_universe, output_paths["research_universe"])
    store.save(universe_coverage, output_paths["universe_coverage"])

    report = build_quality_report(
        universe=universe,
        valuation=valuation,
        ohlcv=ohlcv,
        failed_tickers=download.failed_tickers,
        configured_start_date=str(data_config["start_date"]),
        configured_end_date=str(data_config["end_date"]),
        ohlcv_ticker_subset_size=len(tickers),
        ohlcv_source="yfinance fallback",
        valuation_source="TWSE latest snapshot valuation endpoint",
        configured_ticker_limit=ohlcv_settings.ticker_limit,
        ohlcv_requested_tickers=len(tickers),
        universe_coverage=universe_coverage,
        membership=universe_settings["membership"],
        minimum_universe_coverage=minimum_coverage,
        universe_scope=universe_scope,
        research_config=research_config,
        liquidity_config=universe_settings["liquidity"],
        fundamental_coverage_report=resolve_path(
            config_path, paths.get("fundamental_coverage_report", "")
        )
        if paths.get("fundamental_coverage_report")
        else None,
    )
    output_paths["data_quality_report"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["data_quality_report"].write_text(report, encoding="utf-8")

    created_at = datetime.now(UTC)
    append_manifest_entries(
        [
            build_manifest_entry(
                artifact_name="ohlcv",
                path=str(output_paths["ohlcv"]),
                frame=ohlcv,
                source_inputs=["yfinance"],
                schema_version="ohlcv-v2",
                created_at=created_at,
                notes="canonical adjusted OHLCV",
                provenance={
                    "source": "yfinance",
                    "price_adjustment": "auto_adjusted",
                    "volume_basis": "reported_shares",
                },
            ),
            build_manifest_entry(
                artifact_name="research_universe",
                path=str(output_paths["research_universe"]),
                frame=research_universe,
                source_inputs=[
                    str(output_paths["universe"]),
                    str(output_paths["ohlcv"]),
                ],
                schema_version="research-universe-v1",
                created_at=created_at,
                notes="date-aware research eligibility",
                provenance={
                    "membership": universe_settings["membership"],
                    "liquidity_measure_source": "proxy_close_times_volume",
                },
            ),
            build_manifest_entry(
                artifact_name="universe_coverage",
                path=str(output_paths["universe_coverage"]),
                frame=universe_coverage,
                source_inputs=[str(output_paths["research_universe"])],
                schema_version="research-universe-coverage-v1",
                created_at=created_at,
                notes="date-level research-universe coverage",
            ),
        ],
        output_paths["manifest"],
    )

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Week 1 data pipeline.")
    parser.add_argument("--config", required=True, help="Path to strategy YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

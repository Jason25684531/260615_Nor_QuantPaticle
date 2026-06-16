from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from twse_factor_lab.data.normalizer import (
    normalize_ohlcv,
    normalize_universe,
    normalize_valuation,
)
from twse_factor_lab.data.parquet_store import ParquetStore
from twse_factor_lab.data.twse_client import TWSEClient
from twse_factor_lab.data.yfinance_client import OhlcvDownloadResult, YFinanceClient


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
        f"- OHLCV source: {ohlcv_source}",
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
    ohlcv = normalize_ohlcv(download.data)

    output_paths = {
        "universe": resolve_path(config_path, paths["universe"]),
        "valuation": resolve_path(config_path, paths["valuation"]),
        "ohlcv": resolve_path(config_path, paths["ohlcv"]),
        "data_quality_report": resolve_path(config_path, paths["data_quality_report"]),
    }

    store.save(universe, output_paths["universe"])
    store.save(valuation, output_paths["valuation"])
    store.save(ohlcv, output_paths["ohlcv"])

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
    )
    output_paths["data_quality_report"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["data_quality_report"].write_text(report, encoding="utf-8")

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

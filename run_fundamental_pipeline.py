"""Build point-in-time fundamental research artifacts from official sources."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from run_data_pipeline import load_config, resolve_path
from twse_factor_lab.data.fundamental import (
    FundamentalClient,
    FundamentalDataError,
    RawFundamentalCache,
    build_financial_pit,
    build_fundamental_coverage,
    build_fundamental_coverage_report,
    build_fundamental_matrix,
    build_monthly_revenue_pit,
    build_valuation_pit,
    derive_metrics,
    validate_pit_records,
)
from twse_factor_lab.data.manifest import append_manifest_entries, build_manifest_entry
from twse_factor_lab.data.parquet_store import ParquetStore


def roc_years(start_roc_year: int, ohlcv: pd.DataFrame) -> range:
    """Return only source years represented by the configured OHLCV window."""

    end_roc_year = int(pd.to_datetime(ohlcv["date"]).max().year) - 1911
    return range(int(start_roc_year), end_roc_year + 1)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_fundamental_data(
    client: FundamentalClient,
    *,
    ohlcv: pd.DataFrame,
    research_universe: pd.DataFrame,
    start_roc_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Fetch source rows, retaining successful source partitions only."""

    errors: list[str] = []
    statements: list[pd.DataFrame] = []
    publications: list[pd.DataFrame] = []
    revenue: list[pd.DataFrame] = []
    valuation: list[pd.DataFrame] = []
    years = list(roc_years(start_roc_year, ohlcv))
    for year in years:
        for season in range(1, 5):
            for statement in ("income", "balance"):
                try:
                    statements.append(client.mops_statement(year, season, statement))
                except FundamentalDataError as exc:
                    errors.append(f"mops {statement} {year}Q{season}: {exc}")
        for month in range(1, 13):
            try:
                revenue.append(client.monthly_revenue(year, month))
            except FundamentalDataError as exc:
                errors.append(f"monthly_revenue {year}-{month:02d}: {exc}")

    tickers = (
        research_universe.loc[research_universe["is_eligible"].astype(bool), "ticker"]
        .astype(str)
        .drop_duplicates()
    )
    for ticker in tickers:
        for year in years:
            try:
                publication_rows = client.publication_dates(ticker, year)
                if not publication_rows.empty:
                    publications.append(publication_rows)
            except FundamentalDataError as exc:
                errors.append(f"publication {ticker} {year}: {exc}")

    for date in pd.to_datetime(ohlcv["date"]).drop_duplicates().sort_values():
        try:
            valuation.append(client.valuation_daily(date))
        except FundamentalDataError as exc:
            errors.append(f"valuation {date:%Y-%m-%d}: {exc}")

    return (
        _concat(statements),
        _concat(publications),
        _concat(revenue),
        _concat(valuation),
        errors,
    )


def run_pipeline(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    paths = config["paths"]
    fundamental = config["fundamental"]
    if fundamental.get("availability_policy", "next_trading_day") != "next_trading_day":
        raise ValueError(
            "Only fundamental.availability_policy=next_trading_day is supported"
        )
    store = ParquetStore()
    ohlcv = store.load(resolve_path(config_path, paths["ohlcv"]))
    research_universe = store.load(
        resolve_path(config_path, paths["research_universe"])
    )
    cache = RawFundamentalCache(resolve_path(config_path, fundamental["raw_cache_dir"]))
    client = FundamentalClient(
        cache=cache,
        twse_base_url=config["twse"]["base_url"],
        timeout=int(config["twse"].get("timeout_seconds", 30)),
        throttle_seconds=float(fundamental.get("throttle_seconds", 0.5)),
        retry=int(fundamental.get("retry", 3)),
    )
    statements, publications, revenue, valuation, errors = collect_fundamental_data(
        client,
        ohlcv=ohlcv,
        research_universe=research_universe,
        start_roc_year=int(fundamental["start_roc_year"]),
    )
    if statements.empty or publications.empty or revenue.empty or valuation.empty:
        raise RuntimeError(
            "Fundamental pipeline produced incomplete source data; existing artifacts "
            "were not overwritten. " + "; ".join(errors[:10])
        )

    calendar = pd.DatetimeIndex(pd.to_datetime(ohlcv["date"]).drop_duplicates())
    financial = build_financial_pit(
        statements,
        publications,
        trading_days=calendar,
        missing_publication_date=str(
            fundamental.get("missing_publication_date", "exclude")
        ),
    )
    monthly = build_monthly_revenue_pit(revenue, trading_days=calendar)
    daily_valuation = build_valuation_pit(valuation)
    pit = derive_metrics(
        pd.concat([financial, monthly, daily_valuation], ignore_index=True)
    )
    validate_pit_records(pit)
    matrix = build_fundamental_matrix(pit, research_universe)
    coverage = build_fundamental_coverage(pit, research_universe)
    outputs = {
        name: resolve_path(config_path, paths[name])
        for name in (
            "fundamental_pit",
            "fundamental_matrix",
            "valuation_daily",
            "fundamental_coverage",
            "fundamental_coverage_report",
            "manifest",
        )
    }
    store.save(pit, outputs["fundamental_pit"])
    store.save(matrix, outputs["fundamental_matrix"])
    store.save(valuation, outputs["valuation_daily"])
    store.save(coverage, outputs["fundamental_coverage"])
    outputs["fundamental_coverage_report"].parent.mkdir(parents=True, exist_ok=True)
    report = build_fundamental_coverage_report(coverage, pit)
    if errors:
        report += "\n\n## Source Gaps\n\n" + "\n".join(f"- {item}" for item in errors)
    outputs["fundamental_coverage_report"].write_text(report, encoding="utf-8")
    quality_report = paths.get("data_quality_report")
    if quality_report:
        quality_path = resolve_path(config_path, quality_report)
        if quality_path.exists():
            marker = "## Fundamental Coverage"
            summary = quality_path.read_text(encoding="utf-8")
            if marker not in summary:
                quality_path.write_text(
                    summary.rstrip()
                    + f"\n\n{marker}\n\n"
                    + f"- report: {outputs['fundamental_coverage_report']}\n",
                    encoding="utf-8",
                )
    created_at = datetime.now(UTC)
    append_manifest_entries(
        [
            build_manifest_entry(
                artifact_name="fundamental_pit",
                path=str(outputs["fundamental_pit"]),
                frame=pit,
                source_inputs=["mopsov", "doc.twse", "TWSE BWIBBU_d"],
                schema_version="fundamental-pit-v1",
                created_at=created_at,
                provenance={
                    "source": "mopsov+doc.twse+TWSE",
                    "endpoint": "ajax_t163sb04,ajax_t163sb05,t57sb01,t21sc03,BWIBBU_d",
                    "publication_date_source": "doc.twse t57sb01",
                    "pit_status": "FULL_PIT,PUBLICATION_DATE_AWARE,PERIOD_ONLY",
                },
            ),
            build_manifest_entry(
                artifact_name="fundamental_matrix",
                path=str(outputs["fundamental_matrix"]),
                frame=matrix,
                source_inputs=[
                    str(outputs["fundamental_pit"]),
                    paths["research_universe"],
                ],
                schema_version="fundamental-matrix-v1",
                created_at=created_at,
                provenance={"pit_status": "historical snapshot excluded"},
            ),
            build_manifest_entry(
                artifact_name="fundamental_coverage",
                path=str(outputs["fundamental_coverage"]),
                frame=coverage,
                source_inputs=[
                    str(outputs["fundamental_pit"]),
                    paths["research_universe"],
                ],
                schema_version="fundamental-coverage-v1",
                created_at=created_at,
                provenance={"generated_at": created_at.isoformat()},
            ),
        ],
        outputs["manifest"],
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()

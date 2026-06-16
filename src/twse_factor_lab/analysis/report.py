"""Markdown report builder for historical factor analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def _pivot_quantile_summary(
    quantile_returns: pd.DataFrame, quantiles: int
) -> pd.DataFrame:
    if quantile_returns.empty:
        columns = ["factor", "horizon", *[f"Q{i}" for i in range(1, quantiles + 1)]]
        return pd.DataFrame(columns=columns + ["top_bottom_spread"])

    pivot = (
        quantile_returns.pivot_table(
            index=["factor", "horizon"],
            columns="quantile",
            values="mean_return",
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    renamed_columns: list[str] = []
    for column in pivot.columns:
        if column == "factor":
            renamed_columns.append("factor")
        elif column == "horizon":
            renamed_columns.append("horizon")
        else:
            renamed_columns.append(f"Q{column}")
    pivot.columns = renamed_columns
    pivot["top_bottom_spread"] = pivot[f"Q{quantiles}"] - pivot["Q1"]
    return pivot


def build_factor_analysis_report(
    *,
    config_path: str | Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    factors_analyzed: list[str],
    excluded_snapshot: dict[str, str],
    horizons: list[int],
    quantiles: int,
    close_matrix: pd.DataFrame,
    ic_summary: pd.DataFrame,
    quantile_returns: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    monotonicity: pd.DataFrame,
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    ticker_count = len(close_matrix.columns)
    date_range = "N/A"
    if not close_matrix.empty:
        date_range = (
            f"{close_matrix.index.min().date()} to {close_matrix.index.max().date()}"
        )

    quantile_summary = _pivot_quantile_summary(quantile_returns, quantiles)
    lines = [
        "# Factor Analysis Report",
        "",
        "## Run Metadata",
        "",
        f"- generated_at: {generated_at}",
        f"- config_path: {Path(config_path)}",
        "- pipeline_name: run_factor_analysis.py",
        "",
        "## Scope",
        "",
        f"- historical factors analyzed: {', '.join(factors_analyzed)}",
        "- snapshot factors excluded:",
    ]
    for factor_name, status in excluded_snapshot.items():
        lines.append(f"  - {factor_name}: {status}")
    lines.extend(
        [
            (
                "- reason for exclusion: valuation snapshot factors are not "
                "point-in-time historical series."
            ),
            "",
            "## Input Artifacts",
            "",
            f"- close_matrix: {input_paths['close_matrix']}",
            f"- factors_price_volume: {input_paths['factors_price_volume']}",
            f"- factors_composite: {input_paths['factors_composite']}",
            f"- close_matrix_shape: {close_matrix.shape}",
            f"- close_matrix_date_range: {date_range}",
            f"- ohlcv_subset_ticker_count: {ticker_count}",
            "",
            "## Forward Return Setup",
            "",
            f"- horizons: {', '.join(f'{h}D' for h in horizons)}",
            (
                "- target return definition: forward return from date T close "
                "to date T+h close."
            ),
            (
                "- no-lookahead note: factor values are evaluated at date T "
                "without shifting the factor forward."
            ),
            "",
            "## IC / IR Summary",
            "",
        ]
    )
    if ic_summary.empty:
        lines.append("No IC summary rows generated.")
    else:
        for row in ic_summary.itertuples(index=False):
            lines.append(
                "- "
                f"{row.factor} | {row.horizon}D | "
                f"IC mean={row.ic_mean:.4f}, IC std={row.ic_std:.4f}, "
                f"IR={'NaN' if pd.isna(row.ir) else f'{row.ir:.4f}'}, "
                f"valid dates={int(row.valid_date_count)}, "
                f"avg assets={row.average_asset_count:.2f}"
            )

    lines.extend(["", "## Quantile Returns", ""])
    if quantile_summary.empty:
        lines.append("No quantile-return rows generated.")
    else:
        for row in quantile_summary.itertuples(index=False):
            quantile_parts = [
                f"Q{i}={getattr(row, f'Q{i}'):.4f}" for i in range(1, quantiles + 1)
            ]
            lines.append(
                "- "
                f"{row.factor} | {row.horizon}D | "
                f"{', '.join(quantile_parts)} | "
                f"top-bottom={row.top_bottom_spread:.4f}"
            )

    lines.extend(["", "## Factor Turnover", ""])
    if turnover_summary.empty:
        lines.append("No turnover rows generated.")
    else:
        for row in turnover_summary.itertuples(index=False):
            lines.append(
                "- "
                f"{row.factor} | avg top-quantile turnover="
                f"{row.average_best_bucket_turnover:.4f} | dates={int(row.date_count)}"
            )

    lines.extend(["", "## Monotonicity Check", ""])
    if monotonicity.empty:
        lines.append("No monotonicity rows generated.")
    else:
        for row in monotonicity.itertuples(index=False):
            score = (
                "NaN"
                if pd.isna(row.adjacent_agreement_ratio)
                else f"{row.adjacent_agreement_ratio:.4f}"
            )
            lines.append(
                "- "
                f"{row.factor} | {row.horizon}D | "
                f"monotonicity_pass={str(bool(row.monotonicity_pass)).lower()} | "
                f"score={score} | "
                f"notes={row.notes}"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Factors with consistently positive IC, positive top-bottom spreads, "
                "and monotonic quantile structure appear more promising."
            ),
            (
                "- Weak or unstable signals should be treated as research findings, "
                "not pipeline failures."
            ),
            (
                f"- Current OHLCV coverage is limited to {ticker_count} tickers, so "
                "results are subset-level rather than full-market conclusions."
            ),
            "",
            "## Limitations",
            "",
            f"- only {ticker_count} tickers are covered by the current OHLCV subset",
            "- yfinance fallback data may differ from official adjusted TWSE data",
            "- valuation factors are excluded because they remain snapshot-only",
            (
                "- no transaction cost model, portfolio construction, or "
                "backtest is included"
            ),
            "- not investment advice",
            "",
            "## Generated Artifacts",
            "",
        ]
    )
    for name, path in output_paths.items():
        if name == "manifest":
            continue
        lines.append(f"- {name}: {path}")

    return "\n".join(lines)

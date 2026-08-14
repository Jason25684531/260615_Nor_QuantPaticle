"""Build Week 3 factor scoreboard artifacts."""

from __future__ import annotations

import pandas as pd

ELIGIBLE_HISTORICAL_FACTORS = {
    "momentum_60d",
    "low_volatility_20d",
    "volume_ratio_5d_60d",
    "historical_price_volume",
}
SNAPSHOT_FACTORS = {
    "pb_inverse",
    "pe_inverse",
    "dividend_yield",
    "latest_snapshot_mixed",
}
SCOREBOARD_COLUMNS = [
    "factor",
    "best_horizon",
    "ic_mean",
    "ic_std",
    "ir",
    "top_bottom_spread",
    "avg_turnover",
    "monotonicity_score",
    "monotonicity_pass",
    "is_historical_ready",
    "is_backtest_candidate",
    "notes",
    "factor_family",
    "coverage",
    "correlation_warning",
    "pit_ready",
    "status",
]


def _best_ic_rows(ic_summary: pd.DataFrame) -> pd.DataFrame:
    if ic_summary.empty:
        return pd.DataFrame(columns=["factor", "horizon", "ic_mean", "ic_std", "ir"])
    ranked = ic_summary.copy()
    ranked["_rank_ir"] = ranked["ir"].fillna(float("-inf"))
    ranked = ranked.sort_values(["factor", "_rank_ir"], ascending=[True, False])
    return ranked.drop_duplicates("factor").drop(columns="_rank_ir")


def _top_bottom_spreads(quantile_returns: pd.DataFrame) -> pd.DataFrame:
    if quantile_returns.empty:
        return pd.DataFrame(columns=["factor", "horizon", "top_bottom_spread"])
    grouped = (
        quantile_returns.groupby(["factor", "horizon"])["mean_return"]
        .agg(lambda values: float(values.max() - values.min()))
        .reset_index(name="top_bottom_spread")
    )
    return grouped


def build_factor_scoreboard(
    *,
    ic_summary: pd.DataFrame,
    quantile_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    monotonicity: pd.DataFrame,
    ohlcv_ticker_count: int | None = None,
    metadata: dict[str, dict[str, str]] | None = None,
    thresholds: dict[str, float] | None = None,
    correlation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metadata, thresholds = metadata or {}, thresholds or {}
    best = _best_ic_rows(ic_summary)
    spreads = _top_bottom_spreads(quantile_returns)
    factors = (
        set(best.get("factor", pd.Series(dtype=str)).dropna().astype(str))
        | set(spreads.get("factor", pd.Series(dtype=str)).dropna().astype(str))
        | set(turnover.get("factor", pd.Series(dtype=str)).dropna().astype(str))
        | set(monotonicity.get("factor", pd.Series(dtype=str)).dropna().astype(str))
        | SNAPSHOT_FACTORS
    )
    rows: list[dict[str, object]] = []
    for factor in sorted(factors):
        best_row = best[best["factor"] == factor]
        best_horizon = int(best_row.iloc[0]["horizon"]) if not best_row.empty else pd.NA
        spread_row = spreads[
            (spreads["factor"] == factor) & (spreads["horizon"] == best_horizon)
        ]
        turnover_row = turnover[turnover["factor"] == factor]
        monotonicity_row = monotonicity[
            (monotonicity["factor"] == factor)
            & (monotonicity["horizon"] == best_horizon)
        ]
        is_historical = factor in ELIGIBLE_HISTORICAL_FACTORS and not best_row.empty
        notes: list[str] = []
        if factor in SNAPSHOT_FACTORS:
            notes.append("excluded: snapshot-only valuation semantics")
        if not monotonicity_row.empty and not bool(
            monotonicity_row.iloc[0]["monotonicity_pass"]
        ):
            notes.append("weak monotonicity warning")
        if ohlcv_ticker_count is not None:
            notes.append(f"limited OHLCV coverage: {ohlcv_ticker_count} tickers")

        high_correlation = False
        if correlation is not None and not correlation.empty:
            pairs = correlation[
                (correlation["factor_a"] == factor)
                | (correlation["factor_b"] == factor)
            ]
            high_correlation = bool(
                pairs["correlation"]
                .abs()
                .ge(float(thresholds.get("correlation_warning", 0.8)))
                .any()
            )
        insufficient = best_row.empty or (
            "valid_date_count" in best_row
            and int(best_row.iloc[0]["valid_date_count"])
            < int(thresholds.get("min_ic_observations", 1))
        )
        ir = (
            float(best_row.iloc[0]["ir"])
            if not best_row.empty and pd.notna(best_row.iloc[0]["ir"])
            else 0.0
        )
        spread = (
            float(spread_row.iloc[0]["top_bottom_spread"])
            if not spread_row.empty
            else 0.0
        )
        status = (
            "INSUFFICIENT_DATA"
            if insufficient
            else "REJECT"
            if ir <= 0 or spread <= 0
            else "CANDIDATE"
            if abs(ir) >= float(thresholds.get("min_abs_ir", 0))
            and not monotonicity_row.empty
            and bool(monotonicity_row.iloc[0]["monotonicity_pass"])
            else "WEAK"
        )
        rows.append(
            {
                "factor": factor,
                "best_horizon": best_horizon,
                "ic_mean": (
                    float(best_row.iloc[0]["ic_mean"]) if not best_row.empty else pd.NA
                ),
                "ic_std": (
                    float(best_row.iloc[0]["ic_std"]) if not best_row.empty else pd.NA
                ),
                "ir": float(best_row.iloc[0]["ir"]) if not best_row.empty else pd.NA,
                "top_bottom_spread": (
                    float(spread_row.iloc[0]["top_bottom_spread"])
                    if not spread_row.empty
                    else pd.NA
                ),
                "avg_turnover": (
                    float(turnover_row.iloc[0]["average_best_bucket_turnover"])
                    if not turnover_row.empty
                    else pd.NA
                ),
                "monotonicity_score": (
                    float(monotonicity_row.iloc[0]["adjacent_agreement_ratio"])
                    if not monotonicity_row.empty
                    else pd.NA
                ),
                "monotonicity_pass": (
                    bool(monotonicity_row.iloc[0]["monotonicity_pass"])
                    if not monotonicity_row.empty
                    else False
                ),
                "is_historical_ready": is_historical,
                "is_backtest_candidate": is_historical
                and factor not in SNAPSHOT_FACTORS,
                "notes": "; ".join(notes),
                "factor_family": metadata.get(factor, {}).get("family", "TECHNICAL"),
                "coverage": int(best_row.iloc[0]["valid_date_count"])
                if not best_row.empty and "valid_date_count" in best_row
                else 0,
                "correlation_warning": high_correlation,
                "pit_ready": metadata.get(factor, {}).get("family") != "FUNDAMENTAL"
                or not best_row.empty,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=SCOREBOARD_COLUMNS)


def select_backtest_factor(
    scoreboard: pd.DataFrame,
    requested_factor: str | None = None,
) -> str:
    factor_name = requested_factor or "historical_price_volume"
    match = scoreboard[
        (scoreboard["factor"] == factor_name) & (scoreboard["is_backtest_candidate"])
    ]
    if match.empty:
        raise ValueError(f"Requested factor is not backtest eligible: {factor_name}")
    return factor_name

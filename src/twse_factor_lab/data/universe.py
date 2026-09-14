"""Date-aware, function-only research-universe helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def build_listing_eligibility(
    universe: pd.DataFrame, dates: Iterable[object]
) -> pd.DataFrame:
    """Expand the current listing table across dates using listed_date eligibility."""

    members = universe[["ticker", "listed_date"]].drop_duplicates("ticker").copy()
    members["ticker"] = members["ticker"].astype(str)
    members["listed_date"] = pd.to_datetime(members["listed_date"], errors="coerce")
    calendar = pd.DataFrame({"date": pd.to_datetime(list(dates))}).drop_duplicates()
    result = calendar.merge(members, how="cross")
    result["listing_eligible"] = result["listed_date"].isna() | (
        result["date"] >= result["listed_date"]
    )
    result.attrs["missing_listed_date_count"] = int(result["listed_date"].isna().sum())
    return result[["date", "ticker", "listing_eligible"]]


def build_liquidity_eligibility(
    ohlcv: pd.DataFrame,
    *,
    window: int = 20,
    measure: str = "median",
    minimum_traded_value: float = 50_000_000,
) -> pd.DataFrame:
    """Calculate T eligibility strictly from the preceding trading observations."""

    if window < 1 or measure != "median":
        raise ValueError("Liquidity requires a positive median window")
    required = {"date", "ticker", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise KeyError(f"Missing liquidity columns: {sorted(missing)}")
    frame = ohlcv[["date", "ticker", "close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(["ticker", "date"], kind="stable")
    frame["traded_value"] = frame["close"] * frame["volume"]
    frame["liquidity_value"] = frame.groupby("ticker", sort=False)[
        "traded_value"
    ].transform(
        lambda values: values.rolling(window, min_periods=window).median().shift(1)
    )
    frame["liquidity_eligible"] = frame["liquidity_value"] > minimum_traded_value
    return frame[["date", "ticker", "liquidity_eligible"]]


def build_research_universe(
    universe: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    dates: Iterable[object] | None = None,
    liquidity: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Build eligibility flags without treating download limits as membership."""

    date_values = (
        pd.to_datetime(list(dates))
        if dates is not None
        else pd.to_datetime(ohlcv["date"])
    )
    listing = build_listing_eligibility(universe, date_values)
    available = ohlcv[["date", "ticker"]].drop_duplicates().copy()
    available["date"] = pd.to_datetime(available["date"], errors="coerce")
    available["ticker"] = available["ticker"].astype(str)
    result = listing.merge(
        available.assign(has_ohlcv=True), how="left", on=["date", "ticker"]
    )
    result["has_ohlcv"] = result["has_ohlcv"].fillna(False).astype(bool)

    options = dict(liquidity or {})
    if bool(options.get("enabled", True)):
        eligible = build_liquidity_eligibility(
            ohlcv,
            window=int(options.get("window", 20)),
            measure=str(options.get("measure", "median")),
            minimum_traded_value=float(options.get("minimum_traded_value", 50_000_000)),
        )
        result = result.merge(eligible, how="left", on=["date", "ticker"])
        result["liquidity_eligible"] = (
            result["liquidity_eligible"].fillna(False).astype(bool)
        )
    else:
        result["liquidity_eligible"] = result["has_ohlcv"]
    result["is_eligible"] = (
        result["listing_eligible"] & result["has_ohlcv"] & result["liquidity_eligible"]
    )
    return result.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


LIQUIDITY_MEASURE_SOURCE = "proxy_close_times_volume"


def _metric_available(
    fundamental_matrix: pd.DataFrame, metric: str
) -> pd.DataFrame:
    """PIT (date, ticker) pairs where `metric` has a known, already-available value."""

    required = {"date", "ticker", "metric", "value"}
    missing = required - set(fundamental_matrix.columns)
    if missing:
        raise KeyError(f"Missing fundamental columns: {sorted(missing)}")
    frame = fundamental_matrix[fundamental_matrix["metric"] == metric].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str)
    frame = frame[frame["value"].notna()]
    if "available_date" in frame.columns:
        available = pd.to_datetime(frame["available_date"], errors="coerce")
        frame = frame[available.isna() | (available <= frame["date"])]
    pairs = frame[["date", "ticker"]].drop_duplicates()
    pairs[f"{metric}_available"] = True
    return pairs


def build_liquid_pit_universe(
    universe: pd.DataFrame,
    ohlcv: pd.DataFrame,
    fundamental_matrix: pd.DataFrame,
    *,
    dates: Iterable[object] | None = None,
    window: int = 20,
    threshold: float = 50_000_000,
    fundamental_metrics: Iterable[str] = ("eps", "roe"),
) -> pd.DataFrame:
    """Per-date PIT universe: listed -> tradable -> liquid -> fundamental_available.

    Traded value is the canonical ``close * volume`` proxy (see
    LIQUIDITY_MEASURE_SOURCE); it is not official TWSE turnover. Fundamental
    availability requires every declared metric to carry a known, already-
    available PIT value on the date.
    """

    metrics = tuple(fundamental_metrics)
    base = build_research_universe(
        universe,
        ohlcv,
        dates=dates,
        liquidity={
            "enabled": True,
            "window": window,
            "measure": "median",
            "minimum_traded_value": threshold,
        },
    )

    frame = ohlcv[["date", "ticker", "close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str)
    frame = frame.sort_values(["ticker", "date"], kind="stable")
    frame["traded_value"] = frame["close"] * frame["volume"]
    frame["median_traded_value_20d"] = frame.groupby("ticker", sort=False)[
        "traded_value"
    ].transform(
        lambda values: values.rolling(window, min_periods=window).median().shift(1)
    )
    base = base.merge(
        frame[["date", "ticker", "median_traded_value_20d"]],
        how="left",
        on=["date", "ticker"],
    )

    available_flags: pd.DataFrame | None = None
    for metric in metrics:
        pairs = _metric_available(fundamental_matrix, metric)
        base = base.merge(pairs, how="left", on=["date", "ticker"])
        base[f"{metric}_available"] = base[f"{metric}_available"].fillna(False)
        column = base[["date", "ticker", f"{metric}_available"]]
        available_flags = (
            column
            if available_flags is None
            else available_flags.merge(column, on=["date", "ticker"])
        )
    metric_columns = [f"{metric}_available" for metric in metrics]
    base["fundamental_available"] = base[metric_columns].all(axis=1)

    base["listed_status"] = base["listing_eligible"].map(
        {True: "listed", False: "not_yet_listed"}
    )
    base["tradable"] = base["has_ohlcv"].astype(bool)
    base["liquidity_pass"] = base["liquidity_eligible"].astype(bool)
    base["universe_included"] = base["is_eligible"] & base["fundamental_available"]

    columns = [
        "date",
        "ticker",
        "listed_status",
        "tradable",
        "median_traded_value_20d",
        "liquidity_pass",
        "fundamental_available",
        "universe_included",
        *metric_columns,
    ]
    result = base[columns].sort_values(["date", "ticker"], kind="stable")
    result.attrs["liquidity_measure_source"] = LIQUIDITY_MEASURE_SOURCE
    result.attrs["fundamental_metrics"] = list(metrics)
    return result.reset_index(drop=True)


def build_liquid_pit_coverage(
    v2_universe: pd.DataFrame, *, fundamental_metrics: Iterable[str] = ("eps", "roe")
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Per-date counts and a mean/median/min/max summary for the v2 universe."""

    metrics = tuple(fundamental_metrics)
    frame = v2_universe.copy()
    aggregations: dict[str, tuple[str, str]] = {
        "universe_count": ("universe_included", "sum"),
        "liquidity_eligible_count": ("liquidity_pass", "sum"),
        "fundamental_joint_count": ("fundamental_available", "sum"),
    }
    for metric in metrics:
        aggregations[f"{metric}_valid_count"] = (f"{metric}_available", "sum")
    per_date = frame.groupby("date", as_index=False).agg(**aggregations)
    per_date["coverage_ratio"] = (
        per_date["universe_count"]
        .div(per_date["liquidity_eligible_count"].replace(0, np.nan))
        .fillna(0.0)
    )
    count_columns = [name for name in per_date.columns if name != "date"]
    summary = {
        column: {
            "mean": float(per_date[column].mean()),
            "median": float(per_date[column].median()),
            "min": float(per_date[column].min()),
            "max": float(per_date[column].max()),
        }
        for column in count_columns
    }
    summary["liquidity_measure_source"] = LIQUIDITY_MEASURE_SOURCE
    summary["survivorship"] = (
        "current_listed_only; no delisted history available, so the universe "
        "cannot include historically delisted securities"
    )
    return per_date, summary


def build_universe_coverage(research_universe: pd.DataFrame) -> pd.DataFrame:
    """Summarize date-level availability against listing-eligible members."""

    required = {
        "date",
        "listing_eligible",
        "has_ohlcv",
        "liquidity_eligible",
        "is_eligible",
    }
    missing = required - set(research_universe.columns)
    if missing:
        raise KeyError(f"Missing research-universe columns: {sorted(missing)}")
    frame = research_universe.copy()
    listing = frame["listing_eligible"].astype(bool)
    frame["_listing"] = listing
    frame["_ohlcv"] = listing & frame["has_ohlcv"].astype(bool)
    frame["_liquidity"] = (
        listing
        & frame["has_ohlcv"].astype(bool)
        & frame["liquidity_eligible"].astype(bool)
    )
    frame["_ready"] = frame["is_eligible"].astype(bool)
    coverage = frame.groupby("date", as_index=False)[
        ["_listing", "_ohlcv", "_liquidity", "_ready"]
    ].sum()
    coverage = coverage.rename(
        columns={
            "_listing": "listing_eligible_count",
            "_ohlcv": "ohlcv_available_count",
            "_liquidity": "liquidity_pass_count",
            "_ready": "analysis_ready_count",
        }
    )
    denominator = coverage["listing_eligible_count"]
    coverage["coverage_ratio"] = (
        coverage["analysis_ready_count"].div(denominator).fillna(0.0)
    )
    invalid = (
        (coverage["analysis_ready_count"] > coverage["liquidity_pass_count"])
        | (coverage["liquidity_pass_count"] > coverage["ohlcv_available_count"])
        | (coverage["ohlcv_available_count"] > denominator)
        | ~coverage["coverage_ratio"].between(0, 1)
    )
    if invalid.any():
        raise ValueError("Universe coverage has an inconsistent denominator or ratio")
    return coverage

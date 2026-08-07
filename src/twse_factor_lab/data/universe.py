"""Date-aware, function-only research-universe helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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

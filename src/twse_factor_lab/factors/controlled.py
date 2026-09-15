"""Fixed, PIT-safe factor constructors for controlled factor discovery."""
# ruff: noqa: E501

from __future__ import annotations

import pandas as pd


def build_controlled_price_factors(
    close_matrix: pd.DataFrame, volume_matrix: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Return the eight predeclared OHLCV candidates without future bars."""
    returns = close_matrix.pct_change()
    dollar_volume = volume_matrix * close_matrix
    amihud = returns.abs().div(dollar_volume.where(dollar_volume.gt(0)))
    return {
        "M0_MOMENTUM_20D": close_matrix.div(close_matrix.shift(20)).sub(1),
        "M1_MOMENTUM_60D": close_matrix.div(close_matrix.shift(60)).sub(1),
        "M2_NEAR_HIGH_252D": close_matrix.div(close_matrix.rolling(252, min_periods=252).max()),
        "R1_REVERSAL_5D": -close_matrix.div(close_matrix.shift(5)).sub(1),
        "L1_LOW_VOL_20D": -returns.rolling(20, min_periods=20).std(),
        "L3_DOWNSIDE_VOL_20D": -returns.clip(upper=0).rolling(20, min_periods=20).std(),
        "L4_DOLLAR_VOLUME_20D": dollar_volume.rolling(20, min_periods=20).mean(),
        "L2_AMIHUD_20D": -amihud.rolling(20, min_periods=20).mean(),
    }


def build_eps_yoy_change_matrix(
    records: pd.DataFrame, index: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    """As-of EPS(P) minus EPS(P-1Y same fiscal quarter), never shift(4)."""
    eps = records.loc[records["metric"].eq("eps"), ["ticker", "period_end", "available_date", "value"]].copy()
    eps["ticker"] = eps["ticker"].astype(str)
    eps["period_end"] = pd.to_datetime(eps["period_end"])
    eps["available_date"] = pd.to_datetime(eps["available_date"])
    current = eps.rename(columns={"available_date": "current_available", "value": "current_eps"})
    prior = eps.rename(columns={"period_end": "prior_period_end", "available_date": "prior_available", "value": "prior_eps"})
    current["prior_period_end"] = current["period_end"] - pd.DateOffset(years=1)
    pairs = current.merge(prior, on=["ticker", "prior_period_end"], how="inner")
    pairs["available_date"] = pairs[["current_available", "prior_available"]].max(axis=1)
    pairs["value"] = pairs["current_eps"] - pairs["prior_eps"]
    pairs = pairs.sort_values(["available_date", "period_end"]).drop_duplicates(
        ["ticker", "available_date"], keep="last"
    )
    events = pairs.pivot(index="available_date", columns="ticker", values="value")
    return events.reindex(events.index.union(index)).ffill().reindex(index=index, columns=columns)

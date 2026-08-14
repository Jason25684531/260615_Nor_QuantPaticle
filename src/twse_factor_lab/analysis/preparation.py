"""Prepare eligible historical factors for Week 2.5 analysis."""

from __future__ import annotations

import pandas as pd

FACTOR_DIRECTIONS = {
    "momentum_60d": "higher_is_better",
    "low_volatility_20d": "lower_is_better",
    "volume_ratio_5d_60d": "higher_is_better",
    "historical_price_volume": "higher_is_better",
}

SNAPSHOT_EXCLUSION_STATUS = "snapshot_only_not_historical_ready"


def _long_to_matrix(
    frame: pd.DataFrame, value_column: str, *, date_column: str = "date"
) -> pd.DataFrame:
    return frame.pivot(
        index=date_column, columns="ticker", values=value_column
    ).sort_index()


def select_historical_factor_matrices(
    *,
    price_volume_factors: pd.DataFrame,
    composite_factors: pd.DataFrame,
    historical_factors: list[str],
    optional_composites: list[str],
    excluded_snapshot: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    matrices: dict[str, pd.DataFrame] = {}
    for factor_name in historical_factors:
        if factor_name not in price_volume_factors.columns:
            continue
        matrices[factor_name] = _long_to_matrix(price_volume_factors, factor_name)

    if "historical_price_volume" in optional_composites:
        historical_composite = composite_factors[
            composite_factors["composite_type"] == "historical_price_volume"
        ]
        if not historical_composite.empty:
            matrices["historical_price_volume"] = _long_to_matrix(
                historical_composite, "composite_score"
            )

    excluded = {
        factor_name: SNAPSHOT_EXCLUSION_STATUS for factor_name in excluded_snapshot
    }
    return matrices, excluded


def normalized_factor_matrices(
    frames: list[pd.DataFrame], metadata: dict[str, dict[str, str]]
) -> dict[str, pd.DataFrame]:
    """Direction-aware percentile exposures; NaNs remain excluded."""
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["date", "ticker"], how="outer")
    result: dict[str, pd.DataFrame] = {}
    for factor, info in metadata.items():
        if factor not in merged:
            continue
        direction = info.get("direction")
        if direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(f"Ranking factor {factor} has no declared direction")
        values = merged[["date", "ticker", factor]].copy()
        values[factor] = values[factor].replace([float("inf"), float("-inf")], pd.NA)
        values["score"] = values.groupby("date")[factor].rank(pct=True)
        if direction == "lower_is_better":
            values["score"] = 1 - values["score"]
        result[factor] = _long_to_matrix(
            values[["date", "ticker", "score"]].rename(columns={"score": factor}),
            factor,
        )
    return result

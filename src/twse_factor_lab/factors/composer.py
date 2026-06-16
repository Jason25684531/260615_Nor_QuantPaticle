"""Composite factor generation utilities."""

from __future__ import annotations

import pandas as pd

from twse_factor_lab.factors.ranking import rank_factor

PRICE_VOLUME_WEIGHTS = {
    "momentum_60d": 0.20,
    "low_volatility_20d": 0.15,
    "volume_ratio_5d_60d": 0.15,
}

VALUATION_WEIGHTS = {
    "pb_inverse": 0.20,
    "pe_inverse": 0.15,
    "dividend_yield": 0.15,
}


def _long_to_matrix(
    frame: pd.DataFrame, value_column: str, date_column: str = "date"
) -> pd.DataFrame:
    return frame.pivot(
        index=date_column, columns="ticker", values=value_column
    ).sort_index()


def _weighted_average(
    matrices: dict[str, pd.DataFrame],
    directions: dict[str, str],
    weights: dict[str, float],
) -> pd.DataFrame:
    ranked: dict[str, pd.DataFrame] = {
        name: rank_factor(matrix, direction=directions[name])
        for name, matrix in matrices.items()
    }

    weighted_sum: pd.DataFrame | None = None
    weight_sum: pd.DataFrame | None = None
    for name, ranked_matrix in ranked.items():
        weight = weights[name]
        valid = ranked_matrix.notna().astype(float) * weight
        contribution = ranked_matrix * weight
        weighted_sum = (
            contribution
            if weighted_sum is None
            else weighted_sum.add(contribution, fill_value=0.0)
        )
        weight_sum = (
            valid if weight_sum is None else weight_sum.add(valid, fill_value=0.0)
        )

    if weighted_sum is None or weight_sum is None:
        raise ValueError("No ranked factors available for composite computation")
    return weighted_sum / weight_sum.where(weight_sum > 0)


def build_composite_factor_frame(
    price_volume_factors: pd.DataFrame,
    valuation_factors: pd.DataFrame,
) -> pd.DataFrame:
    price_volume_matrices = {
        name: _long_to_matrix(price_volume_factors, name)
        for name in PRICE_VOLUME_WEIGHTS
    }
    directions = {
        "momentum_60d": "higher_is_better",
        "low_volatility_20d": "lower_is_better",
        "volume_ratio_5d_60d": "higher_is_better",
    }
    historical = _weighted_average(
        price_volume_matrices, directions, PRICE_VOLUME_WEIGHTS
    )
    historical_long = (
        historical.stack(future_stack=True).rename("composite_score").reset_index()
    )
    historical_long.columns = ["date", "ticker", "composite_score"]
    historical_long["composite_type"] = "historical_price_volume"
    historical_long["is_snapshot_component_used"] = False

    snapshot_date = pd.Timestamp(valuation_factors["as_of_date"].iloc[0])
    latest_price_volume = price_volume_factors[
        price_volume_factors["date"] == price_volume_factors["date"].max()
    ].copy()
    latest_price_volume["date"] = snapshot_date
    merged = latest_price_volume.merge(
        valuation_factors[
            ["as_of_date", "ticker", "pb_inverse", "pe_inverse", "dividend_yield"]
        ],
        left_on=["date", "ticker"],
        right_on=["as_of_date", "ticker"],
        how="left",
    ).drop(columns=["as_of_date"])

    mixed_matrices = {
        "momentum_60d": _long_to_matrix(merged, "momentum_60d"),
        "low_volatility_20d": _long_to_matrix(merged, "low_volatility_20d"),
        "volume_ratio_5d_60d": _long_to_matrix(merged, "volume_ratio_5d_60d"),
        "pb_inverse": _long_to_matrix(merged, "pb_inverse"),
        "pe_inverse": _long_to_matrix(merged, "pe_inverse"),
        "dividend_yield": _long_to_matrix(merged, "dividend_yield"),
    }
    mixed_directions = directions | {
        "pb_inverse": "higher_is_better",
        "pe_inverse": "higher_is_better",
        "dividend_yield": "higher_is_better",
    }
    mixed_weights = PRICE_VOLUME_WEIGHTS | VALUATION_WEIGHTS
    mixed = _weighted_average(mixed_matrices, mixed_directions, mixed_weights)
    mixed_long = mixed.stack(future_stack=True).rename("composite_score").reset_index()
    mixed_long.columns = ["date", "ticker", "composite_score"]
    mixed_long["composite_type"] = "latest_snapshot_mixed"
    mixed_long["is_snapshot_component_used"] = True

    return (
        pd.concat([historical_long, mixed_long], ignore_index=True)
        .sort_values(["date", "ticker", "composite_type"])
        .reset_index(drop=True)
    )

"""Reusable composite replay primitives shared by strategy and audit commands."""

from __future__ import annotations

import pandas as pd

from twse_factor_lab.analysis.composite_strategy_lab import (
    BUFFER_HOLD_RANK,
    COMPONENTS,
    TOP_N,
)
from twse_factor_lab.factors.controlled import build_controlled_price_factors
from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar
from twse_factor_lab.portfolio.selection import build_topn_positions
from twse_factor_lab.portfolio.weights import build_equal_weight_portfolio


def build_composite(
    close: pd.DataFrame, volume: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build the locked two-factor composite used by replay workflows."""

    factors = build_controlled_price_factors(close, volume)
    matrices = {name: factors[name].reindex_like(close) for name in COMPONENTS}
    ranked = {
        name: matrix.rank(axis=1, pct=True) for name, matrix in matrices.items()
    }
    complete = ranked[COMPONENTS[0]].notna() & ranked[COMPONENTS[1]].notna()
    score = (ranked[COMPONENTS[0]] * 0.5 + ranked[COMPONENTS[1]] * 0.5).where(
        complete
    )
    return score, matrices


def build_targets(
    score: pd.DataFrame, *, rebalance: str, buffer_on: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic T+1 targets from a composite score matrix."""

    frame = (
        score.stack(future_stack=True)
        .rename("composite_score")
        .rename_axis(["date", "ticker"])
        .reset_index()
    )
    frame["composite_type"] = "composite_l2_l4_5050"
    frame["is_snapshot_component_used"] = False
    calendar = build_rebalance_calendar(
        score.index, frequency=rebalance, execution_lag_days=1
    )
    positions = build_topn_positions(
        frame,
        top_n=TOP_N,
        factor_name="composite_l2_l4_5050",
        rebalance_dates=pd.DatetimeIndex(calendar["signal_date"]),
        hold_until_drop=buffer_on,
        drop_rank_buffer=BUFFER_HOLD_RANK if buffer_on else 0,
        rebalance_frequency=rebalance,
    )
    return (
        build_equal_weight_portfolio(positions, rebalance_calendar=calendar),
        calendar,
    )


__all__ = ["build_composite", "build_targets"]

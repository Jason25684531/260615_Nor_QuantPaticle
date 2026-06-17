"""Top N selection from historical composite factor scores."""

from __future__ import annotations

import pandas as pd


def build_topn_positions(
    factors_composite: pd.DataFrame,
    *,
    top_n: int = 20,
    factor_name: str = "historical_price_volume",
    rebalance_dates: pd.DatetimeIndex | None = None,
    hold_until_drop: bool = False,
    drop_rank_buffer: int = 30,
    rebalance_frequency: str = "daily",
) -> pd.DataFrame:
    historical = factors_composite[
        (factors_composite["composite_type"] == factor_name)
        & (~factors_composite["is_snapshot_component_used"].astype(bool))
    ].copy()
    historical = historical.dropna(subset=["composite_score"])
    historical = historical.rename(columns={"composite_score": "factor_score"})
    historical["rank"] = historical.groupby("date")["factor_score"].rank(
        ascending=False,
        method="first",
    )
    historical["rank"] = historical["rank"].astype(int)
    historical["top_n"] = int(top_n)

    if rebalance_dates is not None:
        rebalance_set = set(pd.to_datetime(rebalance_dates))
        historical["date"] = pd.to_datetime(historical["date"])
        historical = historical[historical["date"].isin(rebalance_set)].copy()

    if hold_until_drop:
        result = _apply_buffer_rule(
            historical, top_n=top_n, drop_rank_buffer=drop_rank_buffer
        )
    else:
        result = historical.copy()
        result["selected"] = result["rank"] <= int(top_n)

    result["selection_rule"] = "buffer" if hold_until_drop else "top_n"
    result["rebalance_frequency"] = rebalance_frequency
    result["buffer_enabled"] = hold_until_drop
    result["drop_rank_buffer"] = int(drop_rank_buffer)

    return (
        result[
            [
                "date",
                "ticker",
                "factor_score",
                "rank",
                "selected",
                "top_n",
                "selection_rule",
                "rebalance_frequency",
                "buffer_enabled",
                "drop_rank_buffer",
            ]
        ]
        .sort_values(["date", "rank", "ticker"])
        .reset_index(drop=True)
    )


def buffered_top_n_selection(
    factor_scores_day: pd.Series,
    *,
    top_n: int,
    drop_rank_buffer: int,
    current_holdings: set[str],
) -> set[str]:
    """Select stocks for one date using hold-until-drop buffer rule.

    factor_scores_day: Series indexed by ticker, higher = better rank.
    Returns set of selected tickers.
    """
    ranked = factor_scores_day.dropna().rank(ascending=False, method="first")
    ranked = ranked.astype(int)

    # Retain holdings still within buffer zone
    retained = {
        t
        for t in current_holdings
        if t in ranked.index and ranked[t] <= drop_rank_buffer
    }

    # Add new names from top_n to fill up to top_n
    top_candidates = ranked[ranked <= top_n].index.tolist()
    top_candidates_sorted = sorted(top_candidates, key=lambda t: ranked[t])
    selected = set(retained)
    for ticker in top_candidates_sorted:
        if len(selected) >= top_n:
            break
        selected.add(ticker)

    # If still under top_n, fill from next-best candidates
    if len(selected) < top_n:
        remaining = [t for t in ranked.sort_values().index if t not in selected]
        for ticker in remaining:
            if len(selected) >= top_n:
                break
            selected.add(ticker)

    return selected


def _apply_buffer_rule(
    historical: pd.DataFrame,
    *,
    top_n: int,
    drop_rank_buffer: int,
) -> pd.DataFrame:
    rows = []
    current_holdings: set[str] = set()

    for _date, group in historical.groupby("date", sort=True):
        scores = group.set_index("ticker")["factor_score"]
        selected_tickers = buffered_top_n_selection(
            scores,
            top_n=top_n,
            drop_rank_buffer=drop_rank_buffer,
            current_holdings=current_holdings,
        )
        current_holdings = selected_tickers
        group = group.copy()
        group["selected"] = group["ticker"].isin(selected_tickers)
        rows.append(group)

    if not rows:
        historical["selected"] = False
        return historical

    return pd.concat(rows, ignore_index=True)

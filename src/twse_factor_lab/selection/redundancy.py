"""Deterministic D3.5 composite factor selection: family cap + correlation cap.

Selection is driven only by D3 archived scoreboard/correlation evidence, never
by backtest performance (frozen contract from multi-factor-research-v1).
"""

from __future__ import annotations

import pandas as pd

_STATUS_PRIORITY = {"CANDIDATE": 0, "WEAK": 1}
_ELIGIBLE_STATUS = set(_STATUS_PRIORITY)

WEIGHTS_COLUMNS = [
    "factor",
    "family",
    "ic_ir",
    "scoreboard_status",
    "selection_reason",
    "selected",
    "weight",
]


def validate_canonical_provenance(
    *,
    scoreboard: pd.DataFrame,
    correlation: pd.DataFrame | None,
    canonical_pool: list[str],
    family_map: dict[str, list[str]],
) -> None:
    """Fail fast unless the pool is fully backed by D3 archived evidence."""
    pool_set = set(canonical_pool)
    for family, members in family_map.items():
        outside = [factor for factor in members if factor not in pool_set]
        if outside:
            raise ValueError(
                f"Family '{family}' declares factors outside the canonical D3 "
                f"pool: {outside}"
            )
    if scoreboard is None or scoreboard.empty:
        raise ValueError(
            "factor_scoreboard.parquet is required for D3.5 composite provenance"
        )
    known_factors = set(scoreboard["factor"])
    missing = [factor for factor in canonical_pool if factor not in known_factors]
    if missing:
        raise ValueError(
            f"factor_scoreboard.parquet does not cover canonical pool "
            f"factors: {missing}"
        )
    if correlation is None or correlation.empty:
        raise ValueError(
            "factor_correlation.parquet is required for D3.5 composite provenance"
        )


def _family_representative(
    board: pd.DataFrame, family: str, members: list[str]
) -> dict[str, object]:
    candidates = board[
        board["factor"].isin(members) & board["status"].isin(_ELIGIBLE_STATUS)
    ].copy()
    if candidates.empty:
        return {
            "family": family,
            "factor": None,
            "ic_ir": float("nan"),
            "scoreboard_status": None,
            "selection_reason": "family_excluded:all_reject_or_insufficient_data",
        }
    candidates["_status_rank"] = candidates["status"].map(_STATUS_PRIORITY)
    candidates = candidates.sort_values(
        ["_status_rank", "abs_ir"], ascending=[True, False]
    )
    winner = candidates.iloc[0]
    return {
        "family": family,
        "factor": winner["factor"],
        "ic_ir": float(winner["ir"]),
        "scoreboard_status": winner["status"],
        "selection_reason": (
            f"family_cap:{winner['status'].lower()}_highest_abs_ir"
        ),
    }


def select_composite_factors(
    *,
    scoreboard: pd.DataFrame,
    correlation: pd.DataFrame,
    canonical_pool: list[str],
    family_map: dict[str, list[str]],
    correlation_cap: float,
) -> pd.DataFrame:
    """One representative per family, then drop the weaker of any over-cap pair."""
    board = scoreboard[scoreboard["factor"].isin(set(canonical_pool))].copy()
    board["abs_ir"] = board["ir"].abs()

    representatives = [
        _family_representative(board, family, members)
        for family, members in family_map.items()
    ]
    selected = [row for row in representatives if row["factor"]]
    excluded = [row for row in representatives if not row["factor"]]

    corr_lookup: dict[tuple[str, str], tuple[float, str]] = {}
    for row in correlation.itertuples(index=False):
        corr_lookup[(row.factor_a, row.factor_b)] = (row.correlation, row.status)
        corr_lookup[(row.factor_b, row.factor_a)] = (row.correlation, row.status)

    removed: set[str] = set()
    names = [row["factor"] for row in selected]
    for i, factor_a in enumerate(names):
        for factor_b in names[i + 1 :]:
            pair = corr_lookup.get((factor_a, factor_b))
            if pair is None:
                continue
            correlation_value, status = pair
            # UNKNOWN correlation is not evidence of redundancy: never removes.
            if status != "KNOWN" or pd.isna(correlation_value):
                continue
            if abs(correlation_value) <= correlation_cap:
                continue
            ir_a = next(r["ic_ir"] for r in selected if r["factor"] == factor_a)
            ir_b = next(r["ic_ir"] for r in selected if r["factor"] == factor_b)
            removed.add(factor_a if abs(ir_a) < abs(ir_b) else factor_b)

    rows: list[dict[str, object]] = []
    for row in selected:
        is_removed = row["factor"] in removed
        reason = row["selection_reason"]
        if is_removed:
            reason = f"{reason};removed:correlation_cap_exceeded"
        rows.append(
            {
                "factor": row["factor"],
                "family": row["family"],
                "ic_ir": row["ic_ir"],
                "scoreboard_status": row["scoreboard_status"],
                "selection_reason": reason,
                "selected": not is_removed,
                "weight": 0.0,
            }
        )
    for row in excluded:
        rows.append(
            {
                "factor": None,
                "family": row["family"],
                "ic_ir": float("nan"),
                "scoreboard_status": None,
                "selection_reason": row["selection_reason"],
                "selected": False,
                "weight": 0.0,
            }
        )

    weights = pd.DataFrame(rows, columns=WEIGHTS_COLUMNS)
    selected_count = int(weights["selected"].sum())
    if selected_count > 0:
        weights.loc[weights["selected"], "weight"] = 1.0 / selected_count
    return weights

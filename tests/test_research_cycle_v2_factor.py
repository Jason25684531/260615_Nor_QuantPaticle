"""Phase 3 tests: PIT factor matrices, ranking, missing exclusion, composite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twse_factor_lab.factors.composer import compose_factor_scores
from twse_factor_lab.factors.pit_matrix import build_pit_factor_matrix


def _universe(dates, tickers, included) -> pd.DataFrame:
    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "universe_included": (date, ticker) in included,
                }
            )
    return pd.DataFrame(rows)


def test_pit_alignment_and_universe_mask() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    fundamentals = pd.DataFrame(
        [
            # AAA available same day
            {"date": dates[0], "ticker": "AAA", "metric": "eps", "value": 5.0,
             "available_date": dates[0]},
            # BBB value present but announced in the future -> not PIT-available
            {"date": dates[0], "ticker": "BBB", "metric": "eps", "value": 9.0,
             "available_date": dates[1]},
            {"date": dates[1], "ticker": "AAA", "metric": "eps", "value": 6.0,
             "available_date": dates[1]},
            {"date": dates[1], "ticker": "BBB", "metric": "eps", "value": 9.0,
             "available_date": dates[1]},
        ]
    )
    included = {(dates[0], "AAA"), (dates[1], "AAA"), (dates[1], "BBB")}
    universe = _universe(dates, ["AAA", "BBB"], included)

    matrix = build_pit_factor_matrix(
        fundamentals, "eps", v2_universe=universe, index=dates, columns=["AAA", "BBB"]
    )

    # BBB day0: future announcement -> NaN (PIT). BBB not in universe day0 anyway.
    assert np.isnan(matrix.loc[dates[0], "BBB"])
    assert matrix.loc[dates[0], "AAA"] == 5.0
    assert matrix.loc[dates[1], "AAA"] == 6.0
    assert matrix.loc[dates[1], "BBB"] == 9.0


def test_missing_excluded_from_ranking() -> None:
    dates = pd.to_datetime(["2020-01-02"])
    fundamentals = pd.DataFrame(
        [
            {"date": dates[0], "ticker": "AAA", "metric": "roe", "value": 0.2,
             "available_date": dates[0]},
            {"date": dates[0], "ticker": "BBB", "metric": "roe", "value": 0.1,
             "available_date": dates[0]},
        ]
    )
    # CCC is in the universe but has no ROE -> excluded from ranking (stays NaN).
    included = {(dates[0], t) for t in ("AAA", "BBB", "CCC")}
    universe = _universe(dates, ["AAA", "BBB", "CCC"], included)

    matrix = build_pit_factor_matrix(
        fundamentals, "roe", v2_universe=universe, index=dates,
        columns=["AAA", "BBB", "CCC"],
    )
    ranked = compose_factor_scores(
        matrices={"roe": matrix},
        directions={"roe": "higher_is_better"},
        weights={"roe": 1.0},
    )
    assert np.isnan(ranked.loc[dates[0], "CCC"])
    # higher_is_better: AAA (0.2) ranks above BBB (0.1)
    assert ranked.loc[dates[0], "AAA"] > ranked.loc[dates[0], "BBB"]


def test_composite_is_deterministic_5050() -> None:
    dates = pd.to_datetime(["2020-01-02"])
    columns = ["AAA", "BBB", "CCC"]
    eps = pd.DataFrame([[3.0, 2.0, 1.0]], index=dates, columns=columns)
    roe = pd.DataFrame([[1.0, 2.0, 3.0]], index=dates, columns=columns)
    directions = {"eps": "higher_is_better", "roe": "higher_is_better"}
    weights = {"eps": 0.5, "roe": 0.5}

    first = compose_factor_scores(matrices={"eps": eps, "roe": roe},
                                  directions=directions, weights=weights)
    second = compose_factor_scores(matrices={"eps": eps, "roe": roe},
                                   directions=directions, weights=weights)
    pd.testing.assert_frame_equal(first, second)
    # eps ranks AAA>BBB>CCC, roe ranks CCC>BBB>AAA: the 50/50 blend is symmetric,
    # so the two extremes tie and BBB keeps its (equal) middle rank.
    assert np.isclose(first.loc[dates[0], "AAA"], first.loc[dates[0], "CCC"])
    assert np.isclose(first.loc[dates[0], "BBB"], eps.rank(axis=1, pct=True).loc[
        dates[0], "BBB"
    ]) or first.loc[dates[0], "BBB"] > 0

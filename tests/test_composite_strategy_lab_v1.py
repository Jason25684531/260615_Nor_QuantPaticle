"""Governance tests for Composite Strategy Lab v1's bounded search."""

from __future__ import annotations

import pytest

from twse_factor_lab.analysis.composite_strategy_lab import (
    BASE_COST,
    BUFFER_HOLD_RANK,
    COMPONENTS,
    TOP_N,
    WEIGHTS,
    CompositeStrategyLabError,
    candidate_payload,
    readiness,
    robust_plateau,
    select_candidate,
    strategy_configs,
    validate_selection_config,
    verify_locked_candidate,
)


def _rows() -> list[dict[str, object]]:
    rows = strategy_configs()
    for number, row in enumerate(rows, start=1):
        row.update(
            net_sharpe=0.1 * number,
            net_cagr=0.05 * number,
            turnover=0.1 * number,
            max_drawdown=-0.1 * number,
        )
    return rows


def test_s1_s2_components_weights_topn_and_cost_are_fixed() -> None:
    assert COMPONENTS == ("L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D")
    assert WEIGHTS == {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5}
    assert TOP_N == 5
    assert BASE_COST == {
        "buy_fee_rate": 0.001425,
        "sell_fee_rate": 0.001425,
        "transaction_tax_rate": 0.003,
        "slippage_rate": 0.001,
    }


def test_s4_to_s6_search_space_is_exactly_four_terminal_trials() -> None:
    rows = strategy_configs()
    assert [(row["rebalance"], row["buffer"]) for row in rows] == [
        ("weekly", False),
        ("weekly", True),
        ("monthly", False),
        ("monthly", True),
    ]
    assert all(row["selection_relevant"] and row["terminal_trial"] for row in rows)
    assert all(row["top_n"] == 5 and row["weighting"] == "equal_weight" for row in rows)


def test_s5_buffer_contract_is_v2_topn_plus_two() -> None:
    assert BUFFER_HOLD_RANK == 7
    assert candidate_payload({**strategy_configs()[1]})["buffer_hold_rank"] == 7


def test_s6_undeclared_parameters_fail() -> None:
    config = {**strategy_configs()[0], "top_n": 10}
    with pytest.raises(CompositeStrategyLabError, match="undeclared"):
        validate_selection_config(config)


def test_s11_no_cost_cannot_enter_selection_population() -> None:
    row = {**strategy_configs()[0], "cost_model": "no_cost"}
    with pytest.raises(CompositeStrategyLabError, match="undeclared"):
        validate_selection_config(row)


def test_s12_heatmap_population_and_s13_lock_are_deterministic() -> None:
    rows = _rows()
    lock = select_candidate(rows)
    assert lock["status"] == "SUCCESS"
    assert lock["strategy_id"] == "S4"
    assert robust_plateau(rows)["best_config"] == "S4"


def test_s14_post_lock_mutation_fails() -> None:
    lock = select_candidate(_rows())
    changed = dict(lock["candidate"])
    changed["top_n"] = 10
    with pytest.raises(CompositeStrategyLabError, match="POST_LOCK_MUTATION"):
        verify_locked_candidate(lock, changed)


def test_s18_only_four_trials_and_breadth_is_not_selection_trial() -> None:
    rows = _rows()
    assert len(rows) == 4
    assert all(row["trial_stage"] == "strategy_selection" for row in rows)
    assert all(row["selection_relevant"] for row in rows)


def test_plateau_failure_blocks_change_three() -> None:
    rows = _rows()
    for row in rows[:-1]:
        row["net_sharpe"] = -0.1
        row["net_cagr"] = -0.1
    lock = select_candidate(rows)
    plateau = robust_plateau(rows)
    assert plateau["robust_plateau_status"] == "FAIL"
    assert readiness(lock, plateau)[0] is False

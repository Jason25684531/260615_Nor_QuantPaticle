"""Fresh-state OOS re-execution tests (tasks 3.1-3.5)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from twse_factor_lab.acceptance.research_cycle import (
    ResearchCycleError,
    evaluate_oos,
    run_fresh_oos,
)
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.governance import (
    ResearchManifest,
    load_experiment_registry,
    save_research_manifest,
)
from twse_factor_lab.strategy.lab import StrategyDefinition, build_strategy_targets


def _make_research(**overrides) -> ResearchManifest:
    base = ResearchManifest(
        research_id="fresh-oos-test",
        hypothesis="fresh-state OOS fixture",
        factor_candidates=["pe"],
        universe="fixture",
        dataset_version="dataset-v1",
        is_start="2020-01-06",
        is_end="2020-01-09",
        oos_start="2020-01-10",
        oos_end="2020-01-13",
        rebalance_search_space=["daily"],
        top_n_search_space=[1],
        cost_scenarios=["none"],
        selection_relevant=True,
        status="active",
    )
    return replace(base, **overrides)


def _prepare_root(root_path) -> str:
    save_research_manifest(_make_research(), root_path)
    return str(root_path)


def _make_definition(**overrides) -> StrategyDefinition:
    base = StrategyDefinition(
        strategy_id="value-only",
        research_id="fresh-oos-test",
        factor_ids=("pe",),
        factor_weights={"pe": 1.0},
        top_n=1,
        rebalance_frequency="daily",
        buffer_enabled=False,
        drop_rank_buffer=0,
        cost_scenario="none",
        universe="fixture",
        dataset_version="dataset-v1",
    )
    return replace(base, **overrides)


def _make_price_jump_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """A always ranks best (lowest pe); A's close jumps +50% on the first OOS day."""
    dates = pd.bdate_range("2020-01-06", periods=6)
    close = pd.DataFrame(
        {"A": [100.0, 100.0, 100.0, 100.0, 150.0, 150.0], "B": [50.0] * 6},
        index=dates,
    )
    pe = pd.DataFrame({"A": [1.0] * 6, "B": [2.0] * 6}, index=dates)
    return close, {"pe": pe}


def test_fresh_oos_reexecution_reports_fresh_state_evidence(tmp_path):
    root = _prepare_root(tmp_path)
    close, factors = _make_price_jump_data()
    result = run_fresh_oos(
        root=root,
        research_id="fresh-oos-test",
        strategy_id="value-only",
        experiment_id="fresh-oos-1",
        definition=_make_definition(),
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT"},
        cost_model=CostModel(0.0, 0.0, 0.0, 0.0),
        oos_start="2020-01-10",
        oos_end="2020-01-13",
    )

    assert result["fresh_state"] is True
    assert result["evidence_type"] == "fresh_state_reexecution"
    assert result["initial_positions"] == {}
    assert result["inherited_is_state"] is False
    assert result["oos_start"] == "2020-01-10"
    assert result["oos_end"] == "2020-01-13"
    assert result["warmup_start"] == "2020-01-06"
    assert result["warmup_end"] == "2020-01-09"
    assert len(result["oos_returns"]) == 2
    assert len(result["oos_nav"]) == 2
    assert result["oos_positions"]
    assert result["oos_metrics"]["observation_count"] == 2

    [record] = load_experiment_registry(root, "fresh-oos-test")
    assert record.status == "completed"
    assert record.experiment_type == "diagnostic"
    assert record.selection_relevant is False
    assert (tmp_path / result["artifact_path"]).exists()


def test_fresh_oos_does_not_inherit_is_price_jump_but_legacy_slice_does(tmp_path):
    root = _prepare_root(tmp_path)
    close, factors = _make_price_jump_data()
    definition = _make_definition()
    initial_cash = 1_000_000.0
    cost_model = CostModel(0.0, 0.0, 0.0, 0.0)

    # Legacy path: one continuous full-period run, sliced after the fact.
    full_close, portfolio_weights, _resolved = build_strategy_targets(
        definition=definition,
        root=root,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT"},
    )
    full_results, _metrics_frame = run_weight_backtest(
        close_matrix=full_close,
        portfolio_weights=portfolio_weights,
        cost_model=cost_model,
        initial_cash=initial_cash,
        top_n=1,
        use_vectorbt=False,
        allow_fallback=True,
    )
    full_returns = pd.Series(
        full_results["returns"].to_numpy(),
        index=pd.DatetimeIndex(full_results["date"]),
    )
    legacy = evaluate_oos(
        full_returns,
        oos_start="2020-01-10",
        oos_end="2020-01-13",
        initial_capital=initial_cash,
    )
    assert legacy["fresh_state"] is False
    assert legacy["oos_returns"][0]["value"] == pytest.approx(0.5)

    # Fresh-state path: re-executed from empty positions right at OOS start.
    fresh = run_fresh_oos(
        root=root,
        research_id="fresh-oos-test",
        strategy_id="value-only",
        experiment_id="fresh-oos-hard",
        definition=definition,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT"},
        cost_model=cost_model,
        oos_start="2020-01-10",
        oos_end="2020-01-13",
    )
    assert fresh["oos_returns"][0]["value"] == pytest.approx(0.0)


def test_future_data_does_not_change_past_target_construction(tmp_path):
    root_short = _prepare_root(tmp_path / "short")
    root_long = _prepare_root(tmp_path / "long")
    close_short, factors_short = _make_price_jump_data()

    extra_dates = pd.bdate_range(
        close_short.index[-1] + pd.Timedelta(days=1), periods=100
    )
    all_dates = close_short.index.append(extra_dates)
    close_long = pd.DataFrame(
        {
            "A": list(close_short["A"]) + [150.0] * 100,
            "B": list(close_short["B"]) + [50.0] * 100,
        },
        index=all_dates,
    )
    pe_long = pd.DataFrame(
        {"A": [1.0] * len(all_dates), "B": [2.0] * len(all_dates)}, index=all_dates
    )

    definition = _make_definition()
    _close_short, weights_short, _r1 = build_strategy_targets(
        definition=definition,
        root=root_short,
        factor_matrices=factors_short,
        close_matrix=close_short,
        admission_results={"pe": "ACCEPT"},
    )
    _close_long, weights_long, _r2 = build_strategy_targets(
        definition=definition,
        root=root_long,
        factor_matrices={"pe": pe_long},
        close_matrix=close_long,
        admission_results={"pe": "ACCEPT"},
    )

    cutoff = close_short.index[-1]
    events_short = (
        weights_short[weights_short["execution_date"] <= cutoff]
        .sort_values(["execution_date", "ticker"])
        .reset_index(drop=True)
    )
    events_long = (
        weights_long[weights_long["execution_date"] <= cutoff]
        .sort_values(["execution_date", "ticker"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(events_short, events_long)


def test_fresh_oos_rejects_invalid_boundary_before_touching_registry(tmp_path):
    root = _prepare_root(tmp_path)
    close, factors = _make_price_jump_data()
    with pytest.raises(ResearchCycleError):
        run_fresh_oos(
            root=root,
            research_id="fresh-oos-test",
            strategy_id="value-only",
            experiment_id="fresh-oos-bad-boundary",
            definition=_make_definition(),
            factor_matrices=factors,
            close_matrix=close,
            admission_results={"pe": "ACCEPT"},
            cost_model=CostModel(0.0, 0.0, 0.0, 0.0),
            oos_start="2020-01-13",
            oos_end="2020-01-10",
        )
    assert load_experiment_registry(root, "fresh-oos-test") == []

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

import twse_factor_lab.strategy.lab as lab
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.factors.composer import compose_factor_scores
from twse_factor_lab.governance import (
    ResearchManifest,
    load_experiment_registry,
    save_research_manifest,
)
from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar
from twse_factor_lab.strategy import (
    StrategyDefinition,
    StrategyLabError,
    StrategyRegistry,
    compare_incremental_contribution,
    run_strategy_trial,
)


def make_research(**overrides) -> ResearchManifest:
    base = ResearchManifest(
        research_id="strategy-lab-test",
        hypothesis="factor composition fixture",
        factor_candidates=["pe", "roe", "momentum_40d"],
        universe="fixture",
        dataset_version="dataset-v1",
        is_start="2020-01-01",
        is_end="2020-01-03",
        oos_start="2020-01-04",
        oos_end="2020-01-31",
        rebalance_search_space=["daily", "weekly"],
        top_n_search_space=[1, 2],
        cost_scenarios=["none", "base_cost"],
        selection_relevant=True,
        status="active",
    )
    return replace(base, **overrides)


def make_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dates = pd.bdate_range("2020-01-06", periods=12)
    tickers = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        {
            ticker: [100.0 + day * growth for day in range(len(dates))]
            for ticker, growth in zip(tickers, [1.0, 0.8, 0.4, 0.2], strict=True)
        },
        index=dates,
    )
    pe = pd.DataFrame(
        [[4.0, 3.0, 2.0, 1.0]] * len(dates), index=dates, columns=tickers
    )
    roe = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]] * len(dates), index=dates, columns=tickers
    )
    momentum = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]] * len(dates), index=dates, columns=tickers
    )
    return close, {"pe": pe, "roe": roe, "momentum_40d": momentum}


def prepare_root(tmp_path) -> str:
    save_research_manifest(make_research(), tmp_path)
    return str(tmp_path)


def make_definition(**overrides) -> StrategyDefinition:
    base = StrategyDefinition(
        strategy_id="value-quality",
        research_id="strategy-lab-test",
        factor_ids=("pe", "roe"),
        factor_weights={"pe": 0.5, "roe": 0.5},
        top_n=2,
        rebalance_frequency="daily",
        buffer_enabled=False,
        drop_rank_buffer=0,
        cost_scenario="none",
        universe="fixture",
        dataset_version="dataset-v1",
    )
    return replace(base, **overrides)


def test_valid_single_and_multi_factor_definitions_run(tmp_path):
    root = prepare_root(tmp_path)
    close, factors = make_data()
    single = make_definition(
        strategy_id="value-only",
        factor_ids=("pe",),
        factor_weights={"pe": 1.0},
    )
    multi = make_definition(strategy_id="value-quality")

    first = run_strategy_trial(
        definition=single,
        experiment_id="trial-single",
        root=root,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT"},
        cost_model=CostModel(0.0, 0.0, 0.0, 0.0),
    )
    second = run_strategy_trial(
        definition=multi,
        experiment_id="trial-multi",
        root=root,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT", "roe": "CANDIDATE"},
        cost_model=CostModel(0.0, 0.0, 0.0, 0.0),
    )

    assert first.status == second.status == "completed"
    assert second.factor_families == {"pe": "VALUE", "roe": "QUALITY"}
    assert first.factor_ids == ("pe",)
    assert len(load_experiment_registry(root, "strategy-lab-test")) == 2


def test_lower_is_better_direction_is_used_before_composition():
    _, factors = make_data()
    scores = compose_factor_scores(
        {"pe": factors["pe"]}, {"pe": "lower_is_better"}, {"pe": 1.0}
    )
    assert scores.loc[scores.index[0], "D"] > scores.loc[scores.index[0], "A"]


def test_composition_is_deterministic_and_weights_are_valid():
    _, factors = make_data()
    first = compose_factor_scores(
        {"pe": factors["pe"], "roe": factors["roe"]},
        {"pe": "lower_is_better", "roe": "higher_is_better"},
        {"pe": 0.25, "roe": 0.75},
    )
    second = compose_factor_scores(
        {"roe": factors["roe"], "pe": factors["pe"]},
        {"pe": "lower_is_better", "roe": "higher_is_better"},
        {"roe": 0.75, "pe": 0.25},
    )
    assert first.equals(second)
    with pytest.raises(StrategyLabError, match="exactly match"):
        make_definition(factor_weights={"pe": 1.0}).validate_basic()
    with pytest.raises(StrategyLabError, match="sum to one"):
        make_definition(factor_weights={"pe": 0.4, "roe": 0.4}).validate_basic()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_n", 3, "top_n"),
        ("rebalance_frequency", "monthly", "rebalance_frequency"),
        ("cost_scenario", "high_cost", "cost_scenario"),
    ],
)
def test_manifest_search_space_boundaries_fail_before_registration(
    tmp_path, field, value, message
):
    root = prepare_root(tmp_path)
    close, factors = make_data()
    definition = make_definition(**{field: value})
    with pytest.raises(StrategyLabError, match=message):
        run_strategy_trial(
            definition=definition,
            experiment_id="trial-boundary",
            root=root,
            factor_matrices=factors,
            close_matrix=close,
            admission_results={"pe": "ACCEPT", "roe": "ACCEPT"},
        )
    assert load_experiment_registry(root, "strategy-lab-test") == []


def test_unknown_and_rejected_factors_are_not_silently_admitted(tmp_path):
    root = prepare_root(tmp_path)
    close, factors = make_data()
    with pytest.raises(StrategyLabError, match="unknown factor_id"):
        run_strategy_trial(
            definition=make_definition(
                strategy_id="unknown-factor",
                factor_ids=("unknown",),
                factor_weights={"unknown": 1.0},
            ),
            experiment_id="trial-unknown",
            root=root,
            factor_matrices={"unknown": factors["pe"]},
            close_matrix=close,
            admission_results={"unknown": "ACCEPT"},
        )
    with pytest.raises(StrategyLabError, match="REJECT factors"):
        run_strategy_trial(
            definition=make_definition(strategy_id="rejected-factor"),
            experiment_id="trial-rejected",
            root=root,
            factor_matrices=factors,
            close_matrix=close,
            admission_results={"pe": "REJECT", "roe": "ACCEPT"},
        )
    assert load_experiment_registry(root, "strategy-lab-test") == []


def test_rejected_factor_is_allowed_only_as_diagnostic(tmp_path):
    root = prepare_root(tmp_path)
    close, factors = make_data()
    result = run_strategy_trial(
        definition=make_definition(
            strategy_id="rejected-negative-control", selection_relevant=False
        ),
        experiment_id="trial-negative-control",
        root=root,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "REJECT", "roe": "ACCEPT"},
    )
    [record] = load_experiment_registry(root, "strategy-lab-test")
    assert result.status == "completed"
    assert record.selection_relevant is False
    assert record.config["admission_verdicts"]["pe"] == "REJECT"


def test_duplicate_strategy_identity_is_rejected():
    registry = StrategyRegistry()
    registry.register(make_definition())
    with pytest.raises(StrategyLabError, match="duplicate strategy_id"):
        registry.register(make_definition())


def test_preregistration_happens_before_backtest(monkeypatch, tmp_path):
    root = prepare_root(tmp_path)
    close, factors = make_data()
    original = lab.run_weight_backtest
    seen = []

    def wrapped(**kwargs):
        [record] = load_experiment_registry(root, "strategy-lab-test")
        seen.append(record.status)
        return original(**kwargs)

    monkeypatch.setattr(lab, "run_weight_backtest", wrapped)
    run_strategy_trial(
        definition=make_definition(strategy_id="preregistered"),
        experiment_id="trial-preregistered",
        root=root,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT", "roe": "ACCEPT"},
    )
    assert seen == ["running"]


def test_failed_trial_is_retained(monkeypatch, tmp_path):
    root = prepare_root(tmp_path)
    close, factors = make_data()

    def fail(**_kwargs):
        raise RuntimeError("fixture execution failure")

    monkeypatch.setattr(lab, "run_weight_backtest", fail)
    with pytest.raises(RuntimeError, match="fixture execution failure"):
        run_strategy_trial(
            definition=make_definition(strategy_id="failed-strategy"),
            experiment_id="trial-failed",
            root=root,
            factor_matrices=factors,
            close_matrix=close,
            admission_results={"pe": "ACCEPT", "roe": "ACCEPT"},
        )
    [record] = load_experiment_registry(root, "strategy-lab-test")
    assert record.status == "failed"
    assert record.result == {"error": "fixture execution failure"}


def test_handoff_has_returns_nav_positions_and_honest_transactions(tmp_path):
    root = prepare_root(tmp_path)
    close, factors = make_data()
    result = run_strategy_trial(
        definition=make_definition(strategy_id="handoff-strategy"),
        experiment_id="trial-handoff",
        root=root,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT", "roe": "ACCEPT"},
        cost_model=CostModel(0.0, 0.0, 0.0, 0.0),
    )
    handoff = tmp_path / result.handoff_dir
    assert {"returns.csv", "nav.csv", "positions.csv", "metadata.json"} == {
        path.name for path in handoff.iterdir()
    }
    metadata = json.loads((handoff / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["transactions"]["available"] is False
    assert not (handoff / "transactions.csv").exists()
    positions = pd.read_csv(handoff / "positions.csv")
    assert positions.iloc[0].drop(["date", "cash"]).eq(0.0).all()
    assert positions.iloc[1].drop("date").sum() > 0.0


def test_custom_execution_keeps_signal_to_next_trading_day():
    days = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08"])
    calendar = build_rebalance_calendar(days, frequency="daily", execution_lag_days=1)
    assert calendar.iloc[0].signal_date == days[0]
    assert calendar.iloc[0].execution_date == days[1]


def test_vectorbt_request_is_labelled_with_actual_engine(tmp_path):
    root = prepare_root(tmp_path)
    close, factors = make_data()
    result = run_strategy_trial(
        definition=make_definition(
            strategy_id="vectorbt-strategy", execution_engine="vectorbt"
        ),
        experiment_id="trial-vectorbt",
        root=root,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT", "roe": "ACCEPT"},
    )
    assert result.requested_engine == "vectorbt"
    assert result.actual_engine in {"vectorbt", "fallback_custom"}


def test_incremental_contribution_is_enriched_minus_baseline():
    baseline = {
        "strategy_id": "value",
        "total_return": 0.10,
        "sharpe": 0.5,
        "max_drawdown": -0.20,
        "turnover": 0.30,
        "exposure": 0.90,
    }
    enriched = {
        "strategy_id": "value-quality",
        "total_return": 0.14,
        "sharpe": 0.7,
        "max_drawdown": -0.15,
        "turnover": 0.25,
        "exposure": 0.95,
    }
    assert compare_incremental_contribution(baseline, enriched) == {
        "baseline_strategy_id": "value",
        "enriched_strategy_id": "value-quality",
        "total_return": pytest.approx(0.04),
        "sharpe": pytest.approx(0.2),
        "max_drawdown": pytest.approx(0.05),
        "turnover": pytest.approx(-0.05),
        "exposure": pytest.approx(0.05),
    }

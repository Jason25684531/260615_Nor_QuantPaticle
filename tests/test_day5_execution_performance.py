from __future__ import annotations

import json

import pandas as pd
import pytest

from twse_factor_lab.analysis.performance import (
    PerformanceDiagnosticError,
    evaluate_performance,
    run_performance_diagnostic,
)
from twse_factor_lab.backtest.backtrader_engine import run_backtrader_engine
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.tri_engine import (
    compare_engine_results,
    run_tri_engine_validation,
)
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.governance import (
    ResearchManifest,
    load_experiment_registry,
    save_research_manifest,
)
from twse_factor_lab.strategy import (
    StrategyDefinition,
    run_strategy_trial,
)
from twse_factor_lab.strategy.handoff import (
    HandoffValidationError,
    load_strategy_handoff,
)


def _manifest() -> ResearchManifest:
    return ResearchManifest(
        research_id="day5-research",
        hypothesis="event validation fixture",
        factor_candidates=["pe", "roe"],
        universe="fixture",
        dataset_version="dataset-v1",
        is_start="2020-01-01",
        is_end="2020-01-03",
        oos_start="2020-01-04",
        oos_end="2024-01-31",
        rebalance_search_space=["daily"],
        top_n_search_space=[1, 2],
        cost_scenarios=["base_cost", "none"],
        selection_relevant=True,
        status="active",
    )


def _market() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame(
        {"A": range(10, 20), "B": range(20, 30), "C": range(30, 40)},
        index=dates,
        dtype=float,
    )
    factors = {
        "pe": pd.DataFrame(
            [[3.0, 2.0, 1.0]] * len(dates), index=dates, columns=close.columns
        ),
        "roe": pd.DataFrame(
            [[1.0, 2.0, 3.0]] * len(dates), index=dates, columns=close.columns
        ),
    }
    return close, factors


def _make_handoff(tmp_path):
    root = tmp_path
    save_research_manifest(_manifest(), root)
    close, factors = _market()
    definition = StrategyDefinition(
        strategy_id="day5-strategy",
        research_id="day5-research",
        factor_ids=("pe", "roe"),
        factor_weights={"pe": 0.5, "roe": 0.5},
        top_n=2,
        rebalance_frequency="daily",
        buffer_enabled=False,
        drop_rank_buffer=0,
        cost_scenario="base_cost",
        universe="fixture",
        dataset_version="dataset-v1",
    )
    trial = run_strategy_trial(
        definition=definition,
        experiment_id="strategy-trial",
        root=root,
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "CANDIDATE", "roe": "CANDIDATE"},
        cost_model=CostModel(),
    )
    return root, close, trial.handoff_dir


def _direct_targets(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [dates[0], dates[0], dates[2]],
            "execution_date": [dates[1], dates[1], dates[3]],
            "ticker": ["A", "B", "A"],
            "target_weight": [0.5, 0.5, 1.0],
        }
    )


def test_valid_handoff_loads_with_explicit_targets(tmp_path):
    root, _close, handoff_dir = _make_handoff(tmp_path)
    handoff = load_strategy_handoff(root, handoff_dir)
    assert handoff.returns.index.equals(handoff.nav.index)
    assert not handoff.target_weights.empty
    assert handoff.metadata["positions_contract"] == "dollar_positions_including_cash"


def test_handoff_missing_returns_fails(tmp_path):
    root, _close, handoff_dir = _make_handoff(tmp_path)
    (root / handoff_dir / "returns.csv").unlink()
    with pytest.raises(HandoffValidationError, match="missing returns"):
        load_strategy_handoff(root, handoff_dir)


def test_handoff_metadata_and_dataset_mismatch_fail(tmp_path):
    root, _close, handoff_dir = _make_handoff(tmp_path)
    metadata_path = root / handoff_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["research_id"] = "other-research"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(HandoffValidationError, match="research_id"):
        load_strategy_handoff(root, handoff_dir)
    metadata["research_id"] = "day5-research"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(HandoffValidationError, match="dataset_version"):
        load_strategy_handoff(
            root, handoff_dir, expected_dataset_version="other-dataset"
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda frame: frame.iloc[::-1], "monotonic"),
        (lambda frame: frame.assign(nav=frame["nav"] + 1), "inconsistent"),
    ],
)
def test_handoff_date_and_nav_integrity_fail(tmp_path, mutation, message):
    root, _close, handoff_dir = _make_handoff(tmp_path)
    nav_path = root / handoff_dir / "nav.csv"
    nav = pd.read_csv(nav_path)
    nav_path.write_text(
        mutation(nav).to_csv(index=False, lineterminator="\n"), encoding="utf-8"
    )
    with pytest.raises(HandoffValidationError, match=message):
        load_strategy_handoff(root, handoff_dir)


def test_backtrader_preserves_t_plus_one_removed_ticker_and_hold():
    dates = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame(
        {"A": [10, 11, 12, 13, 14], "B": [20, 21, 22, 23, 24]},
        index=dates,
        dtype=float,
    )
    targets = _direct_targets(dates).query("ticker != 'C'")
    run = run_backtrader_engine(
        close_matrix=close,
        target_weights=targets,
        cost_model=CostModel(0, 0, 0, 0),
        initial_cash=1_000,
    )
    assert run.results.loc[0, "position:A"] == 0
    assert run.results.loc[1, "position:B"] > 0
    assert run.results.loc[2, "position:B"] > 0
    assert run.results.loc[3, "position:B"] == pytest.approx(0)
    assert run.results.loc[3, "position:A"] > 0
    assert run.fills["date"].dt.date.tolist() == [
        dates[1].date(),
        dates[1].date(),
        dates[3].date(),
        dates[3].date(),
    ]


def test_backtrader_cost_is_applied_once_and_is_deterministic():
    dates = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame({"A": [10, 11, 12, 13, 14]}, index=dates, dtype=float)
    targets = pd.DataFrame(
        {"execution_date": [dates[1]], "ticker": ["A"], "target_weight": [1.0]}
    )
    cost = CostModel()
    first = run_backtrader_engine(
        close_matrix=close,
        target_weights=targets,
        cost_model=cost,
        initial_cash=1_000,
    )
    second = run_backtrader_engine(
        close_matrix=close,
        target_weights=targets,
        cost_model=cost,
        initial_cash=1_000,
    )
    pd.testing.assert_frame_equal(first.results, second.results)
    assert first.fills["commission"].sum() > 0
    assert first.results.loc[1, "equity"] < 1_000


def test_tri_engine_passes_actual_engines_and_detects_divergence(tmp_path):
    root, close, handoff_dir = _make_handoff(tmp_path)
    result = run_tri_engine_validation(
        root=root,
        research_id="day5-research",
        strategy_id="day5-strategy",
        handoff_dir=handoff_dir,
        experiment_id="execution-diagnostic",
        close_matrix=close,
        cost_model=CostModel(),
    )
    assert result["status"] == "PASS"
    assert result["engines"] == {
        "custom": "custom",
        "vectorbt": "vectorbt",
        "backtrader": "backtrader",
    }
    custom, _ = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=load_strategy_handoff(root, handoff_dir).target_weights,
        cost_model=CostModel(),
        initial_cash=1_000_000,
        top_n=2,
        use_vectorbt=False,
    )
    divergent = custom.copy()
    divergent.loc[1, "returns"] += 0.01
    comparison = compare_engine_results(
        {"custom": custom, "vectorbt": custom, "backtrader": divergent},
        actual_engines={
            "custom": "custom",
            "vectorbt": "vectorbt",
            "backtrader": "backtrader",
        },
    )
    assert comparison["status"] == "FAIL"
    assert comparison["first_divergence"]["date"] == str(close.index[1].date())


def test_fallback_vectorbt_is_not_tri_engine_pass():
    frame = pd.DataFrame(
        {"date": pd.bdate_range("2024-01-01", periods=2), "returns": [0.0, 0.0]}
    )
    result = compare_engine_results(
        {"custom": frame, "vectorbt": frame, "backtrader": frame},
        actual_engines={
            "custom": "custom",
            "vectorbt": "fallback_custom",
            "backtrader": "backtrader",
        },
    )
    assert result["status"] == "FAIL"


def test_performance_report_and_transaction_limitation(tmp_path):
    root, _close, handoff_dir = _make_handoff(tmp_path)
    report = run_performance_diagnostic(
        root=root,
        research_id="day5-research",
        strategy_id="day5-strategy",
        handoff_dir=handoff_dir,
        experiment_id="performance-diagnostic",
    )
    assert report["status"] == "completed"
    assert report["transactions"]["status"] == "UNAVAILABLE"
    assert (root / report["artifact_paths"]["canonical_metrics"]).exists()
    assert (root / report["artifact_paths"]["pyfolio_metrics"]).exists()
    metadata = json.loads(
        (root / report["artifact_paths"]["pyfolio_metadata"]).read_text()
    )
    assert "INFRASTRUCTURE VALIDATION ONLY" in metadata["limitation"]


def test_performance_positions_optional_and_nan_is_rejected():
    returns = pd.Series(
        [0.0, 0.01, -0.005], index=pd.bdate_range("2024-01-01", periods=3)
    )
    report = evaluate_performance(returns)
    assert report["positions"]["available"] is False
    with pytest.raises(PerformanceDiagnosticError, match="finite"):
        bad_returns = returns.copy()
        bad_returns.iloc[1] = float("nan")
        evaluate_performance(bad_returns)


def test_cross_check_is_definition_aware(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    returns = pd.Series(
        rng.normal(0.0005, 0.01, 300), index=pd.bdate_range("2023-01-01", periods=300)
    )
    report = evaluate_performance(returns)
    checks = {row["metric"]: row for row in report["cross_check"]["checks"]}
    assert checks["total_return"]["status"] == "PASS"
    assert checks["cagr"]["status"] == "PASS"
    assert checks["max_drawdown"]["status"] == "PASS"
    assert checks["sharpe"]["status"] == "DEFINITION_DIFFERENCE"
    assert checks["sortino"]["status"] == "DEFINITION_DIFFERENCE"
    assert report["cross_check"]["status"] == "PASS_WITH_DEFINITION_DIFFERENCE"
    assert "reason" in checks["sharpe"]
    notes = report["cross_check"]["definition_notes"]
    assert notes["canonical"]["risk_free_rate"] == 0.0
    assert notes["pyfolio"]["annualization"]


def test_diagnostic_registry_is_not_selection_relevant(tmp_path):
    root, close, handoff_dir = _make_handoff(tmp_path)
    run_tri_engine_validation(
        root=root,
        research_id="day5-research",
        strategy_id="day5-strategy",
        handoff_dir=handoff_dir,
        experiment_id="execution-diagnostic",
        close_matrix=close,
        cost_model=CostModel(),
    )
    run_performance_diagnostic(
        root=root,
        research_id="day5-research",
        strategy_id="day5-strategy",
        handoff_dir=handoff_dir,
        experiment_id="performance-diagnostic",
    )
    records = load_experiment_registry(root, "day5-research")
    assert sum(record.selection_relevant for record in records) == 1
    assert [record.experiment_type for record in records].count("diagnostic") == 2


def test_failed_backtrader_diagnostic_is_retained(monkeypatch, tmp_path):
    root, close, handoff_dir = _make_handoff(tmp_path)
    import twse_factor_lab.backtest.tri_engine as tri

    monkeypatch.setattr(
        tri,
        "run_backtrader_engine",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("BT failed")),
    )
    with pytest.raises(RuntimeError, match="BT failed"):
        run_tri_engine_validation(
            root=root,
            research_id="day5-research",
            strategy_id="day5-strategy",
            handoff_dir=handoff_dir,
            experiment_id="failed-execution",
            close_matrix=close,
            cost_model=CostModel(),
        )
    records = load_experiment_registry(root, "day5-research")
    failed = [
        record for record in records if record.experiment_id == "failed-execution"
    ]
    assert failed[0].status == "failed"
    assert "BT failed" in failed[0].result["error"]


def test_duplicate_diagnostic_id_is_rejected(tmp_path):
    root, close, handoff_dir = _make_handoff(tmp_path)
    kwargs = dict(
        root=root,
        research_id="day5-research",
        strategy_id="day5-strategy",
        handoff_dir=handoff_dir,
        experiment_id="same-diagnostic",
        close_matrix=close,
        cost_model=CostModel(),
    )
    run_tri_engine_validation(**kwargs)
    with pytest.raises(ValueError, match="duplicate experiment_id"):
        run_tri_engine_validation(**kwargs)

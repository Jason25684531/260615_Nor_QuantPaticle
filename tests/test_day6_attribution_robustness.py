import copy

import numpy as np
import pandas as pd
import pytest

from twse_factor_lab.analysis.attribution import (
    AttributionError,
    attribute_portfolio_returns,
    compare_factor_evidence,
    compute_portfolio_exposures,
    neutralize_factor,
    run_attribution_diagnostic,
    standardize_descriptor,
    standardize_descriptors,
)
from twse_factor_lab.analysis.research_robustness import (
    RobustnessError,
    ablate_factor,
    build_predeclared_grid,
    run_factor_ablation,
    run_robustness_sweep,
)
from twse_factor_lab.governance import (
    ExperimentRecord,
    ResearchManifest,
    load_experiment_registry,
    register_experiment,
    save_research_manifest,
)

DATES = pd.date_range("2024-01-02", periods=6, freq="D")
TICKERS = pd.Index(["A", "B", "C", "D", "E", "F"])


def _manifest() -> ResearchManifest:
    return ResearchManifest(
        research_id="day6-test",
        hypothesis="diagnostic fixture",
        factor_candidates=["pe", "roe"],
        universe="fixture",
        dataset_version="dataset-v1",
        is_start="2024-01-01",
        is_end="2024-01-03",
        oos_start="2024-01-04",
        oos_end="2024-01-10",
        rebalance_search_space=["daily", "weekly"],
        top_n_search_space=[1, 2],
        cost_scenarios=["no_cost", "base_cost"],
        selection_relevant=True,
        status="active",
    )


def _prepare(tmp_path):
    save_research_manifest(_manifest(), tmp_path)
    return tmp_path


def _frame(values):
    return pd.DataFrame(values, index=DATES, columns=TICKERS, dtype=float)


def test_descriptor_standardization_is_deterministic_and_size_is_log_scaled():
    market_cap = _frame(np.arange(1, 37).reshape(6, 6))
    first = standardize_descriptor(market_cap, name="market_cap", log_market_cap=True)
    second = standardize_descriptor(market_cap, name="market_cap", log_market_cap=True)
    assert first.equals(second)
    assert first.iloc[0].mean() == pytest.approx(0.0)
    assert first.iloc[0].std(ddof=0) == pytest.approx(1.0)


def test_size_neutralization_removes_known_size_component():
    size = _frame(np.arange(1, 37).reshape(6, 6))
    size_z = standardize_descriptor(size, name="size")
    signal = _frame(np.tile([-2, -1, 1, 2, -1, 1], (6, 1)))
    raw = 4.0 * size_z + signal
    result = neutralize_factor(raw, descriptors={"size": size}, min_assets=5)
    assert result.status == "AVAILABLE"
    assert result.residual.iloc[0].corr(size_z.iloc[0]) == pytest.approx(0.0, abs=1e-10)


def test_industry_neutralization_removes_dummy_and_preserves_residual_signal():
    industry = pd.DataFrame(
        np.tile(["A", "A", "A", "B", "B", "B"], (6, 1)),
        index=DATES,
        columns=TICKERS,
    )
    residual_signal = _frame(np.tile([-2, -1, 1, -2, 0, 2], (6, 1)))
    raw = residual_signal + _frame(np.tile([10, 10, 10, 30, 30, 30], (6, 1)))
    result = neutralize_factor(raw, industry=industry, min_assets=5)
    assert result.status == "AVAILABLE"
    assert result.residual.loc[DATES[0], ["A", "B", "C"]].mean() == pytest.approx(0.0)
    assert result.residual.loc[DATES[0], ["D", "E", "F"]].mean() == pytest.approx(0.0)
    assert result.residual.loc[DATES[0], "F"] > result.residual.loc[DATES[0], "E"]


def test_missing_pit_industry_is_unavailable_not_fabricated():
    industry = pd.DataFrame(
        np.tile(["A", "A", "A", "B", "B", "B"], (6, 1)),
        index=DATES,
        columns=TICKERS,
    )
    result = neutralize_factor(
        _frame(np.arange(1, 37).reshape(6, 6)),
        industry=industry,
        pit_industry_available=False,
    )
    assert result.status == "UNAVAILABLE"
    assert "industry" in result.unavailable_descriptors


def test_insufficient_cross_section_is_explicit():
    raw = _frame(np.arange(1, 37).reshape(6, 6))
    raw.iloc[:, 2:] = np.nan
    result = neutralize_factor(raw, descriptors={"size": raw}, min_assets=5)
    assert result.status == "INSUFFICIENT"
    assert result.residual.isna().all().all()


def test_raw_and_neutralized_factor_evidence_has_both_views():
    close = _frame(100 + np.arange(36).reshape(6, 6))
    factor = _frame(np.tile([1, 2, 3, 4, 5, 6], (6, 1)))
    evidence = compare_factor_evidence(
        factor, factor, close, horizons=(1,), quantiles=3, min_assets=5
    )
    assert evidence["raw"] == evidence["neutralized"]
    assert evidence["raw"][0]["horizon"] == 1


def test_portfolio_exposure_and_active_exposure_are_weighted():
    weights = _frame(np.tile([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], (6, 1)))
    size = _frame(np.tile([1, 2, 3, 4, 5, 6], (6, 1)))
    industry = pd.DataFrame(
        np.tile(["A", "A", "A", "B", "B", "B"], (6, 1)),
        index=DATES,
        columns=TICKERS,
    )
    benchmark = _frame(np.tile([0.0, 0.0, 0.0, 0.0, 0.5, 0.5], (6, 1)))
    result = compute_portfolio_exposures(
        weights, {"size": size, "industry": industry}, benchmark_weights=benchmark
    )
    assert result.portfolio_exposures.loc[DATES[0], "size"] == pytest.approx(3.0)
    assert result.portfolio_exposures.loc[DATES[0], "industry:A"] == pytest.approx(0.6)
    assert result.active_exposures is not None
    assert result.active_exposures.loc[DATES[0], "size"] == pytest.approx(-2.5)


def test_attribution_reconciles_and_reports_error():
    exposures = pd.DataFrame({"size": [2.0, 1.0]}, index=DATES[:2])
    factor_returns = pd.DataFrame({"size": [0.01, -0.02]}, index=DATES[:2])
    actual = pd.Series([0.025, -0.015], index=DATES[:2])
    result = attribute_portfolio_returns(actual, exposures, factor_returns)
    assert result.status == "PASS"
    assert result.explained_return.iloc[0] == pytest.approx(0.02)
    assert result.residual.iloc[0] == pytest.approx(0.005)
    bad = attribute_portfolio_returns(
        actual,
        exposures,
        factor_returns,
        residual=pd.Series([0.0, 0.0], index=DATES[:2]),
    )
    assert bad.status == "FAIL"
    assert bad.reconciliation_error.iloc[0] == pytest.approx(0.005)


def test_descriptor_bundle_marks_unavailable_inputs():
    bundle = standardize_descriptors({"size": _frame(np.arange(36).reshape(6, 6))})
    assert bundle.available_descriptors == ("size",)
    assert "industry" in bundle.unavailable_descriptors
    assert bundle.pit_industry_status == "PIT_INDUSTRY_UNAVAILABLE"


def _candidate():
    return {
        "strategy_id": "candidate",
        "factor_ids": ["pe", "roe"],
        "factor_weights": {"pe": 0.5, "roe": 0.5},
        "top_n": 1,
        "rebalance_frequency": "daily",
        "cost_scenario": "no_cost",
    }


def _metrics(sharpe=1.0):
    return {
        "total_return": 0.1,
        "sharpe": sharpe,
        "max_drawdown": -0.1,
        "turnover": 0.2,
        "exposure": 1.0,
    }


def test_predeclared_grid_is_bounded_and_deterministic(tmp_path):
    _prepare(tmp_path)
    research = _manifest()
    grid = build_predeclared_grid(
        research,
        {
            "top_n": [1, 2],
            "rebalance_frequency": ["weekly"],
            "cost_scenario": ["base_cost"],
        },
    )
    assert len(grid) == 2
    with pytest.raises(RobustnessError, match="undeclared"):
        build_predeclared_grid(research, {"top_n": [3]})


def test_robustness_keeps_candidate_fixed_and_retains_failed_run(tmp_path):
    root = _prepare(tmp_path)
    candidate = _candidate()
    before = copy.deepcopy(candidate)

    def runner(config):
        if config["top_n"] == 2 and config["cost_scenario"] == "base_cost":
            raise RuntimeError("declared run failed")
        return _metrics(0.9)

    result = run_robustness_sweep(
        root=root,
        research_id="day6-test",
        strategy_id="candidate",
        experiment_id="robustness-001",
        candidate_config=candidate,
        grid={"top_n": [1, 2], "cost_scenario": ["no_cost", "base_cost"]},
        runner=runner,
        baseline_result=_metrics(),
    )
    assert candidate == before
    assert result["selection_relevant"] is False
    assert any(row["status"] == "failed" for row in result["scenarios"])
    [record] = load_experiment_registry(root, "day6-test")
    assert record.selection_relevant is False
    assert record.status == "completed"


def test_ablation_removes_exact_factor_and_reports_canonical_deltas():
    candidate = _candidate()
    ablated = ablate_factor(candidate, "pe")
    assert candidate["factor_ids"] == ["pe", "roe"]
    assert ablated["factor_ids"] == ["roe"]
    assert ablated["factor_weights"] == {"roe": 1.0}


def test_ablation_is_registered_diagnostic_only(tmp_path):
    root = _prepare(tmp_path)
    result = run_factor_ablation(
        root=root,
        research_id="day6-test",
        strategy_id="candidate",
        experiment_id="ablation-001",
        candidate_config=_candidate(),
        factor_id="pe",
        baseline_result=_metrics(),
        runner=lambda config: _metrics(0.5),
    )
    assert result["deltas"]["sharpe"] == pytest.approx(-0.5)
    assert len(load_experiment_registry(root, "day6-test")) == 1


def test_diagnostic_experiment_status_and_identity_are_governed(tmp_path):
    root = _prepare(tmp_path)
    register_experiment(
        ExperimentRecord(
            experiment_id="manual-diagnostic",
            research_id="day6-test",
            config={"kind": "attribution"},
            dataset_version="dataset-v1",
            experiment_type="diagnostic",
            status="completed",
            selection_relevant=False,
        ),
        root,
    )
    [record] = load_experiment_registry(root, "day6-test")
    assert record.selection_relevant is False


def test_attribution_artifact_is_deterministic_and_research_scoped(tmp_path):
    root = _prepare(tmp_path)
    close = _frame(100 + np.arange(36).reshape(6, 6))
    factor = _frame(np.tile([1, 2, 3, 4, 5, 6], (6, 1)))
    weights = _frame(np.tile([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], (6, 1)))
    actual = pd.Series(np.full(6, 0.01), index=DATES)
    factor_returns = pd.DataFrame({"size": np.full(6, 0.002)}, index=DATES)
    first = run_attribution_diagnostic(
        root=root,
        research_id="day6-test",
        strategy_id="candidate",
        experiment_id="attribution-001",
        raw_factor=factor,
        close_matrix=close,
        portfolio_weights=weights,
        descriptors={"size": _frame(np.tile([1, 2, 3, 4, 5, 6], (6, 1)))},
        factor_returns=factor_returns,
        actual_returns=actual,
    )
    second = run_attribution_diagnostic(
        root=root,
        research_id="day6-test",
        strategy_id="candidate",
        experiment_id="attribution-002",
        raw_factor=factor,
        close_matrix=close,
        portfolio_weights=weights,
        descriptors={"size": _frame(np.tile([1, 2, 3, 4, 5, 6], (6, 1)))},
        factor_returns=factor_returns,
        actual_returns=actual,
    )
    assert first["attribution"] == second["attribution"]
    assert first["artifact_path"].startswith("data\\research\\day6-test\\")
    records = load_experiment_registry(root, "day6-test")
    assert [record.selection_relevant for record in records] == [False, False]


def test_failed_attribution_is_retained_as_failed(tmp_path):
    root = _prepare(tmp_path)
    with pytest.raises(AttributionError):
        run_attribution_diagnostic(
            root=root,
            research_id="day6-test",
            strategy_id="candidate",
            experiment_id="attribution-failed",
            raw_factor=_frame(np.ones((6, 6))),
            close_matrix=pd.DataFrame(np.ones((6, 6))),
            portfolio_weights=_frame(np.ones((6, 6)) / 6),
        )
    [record] = load_experiment_registry(root, "day6-test")
    assert record.status == "failed"
    assert record.selection_relevant is False


def test_failed_robustness_scenario_is_retained(tmp_path):
    root = _prepare(tmp_path)
    result = run_robustness_sweep(
        root=root,
        research_id="day6-test",
        strategy_id="candidate",
        experiment_id="robustness-failed",
        candidate_config=_candidate(),
        grid={"top_n": [1]},
        runner=lambda _config: (_ for _ in ()).throw(RuntimeError("boom")),
        baseline_result=_metrics(),
    )
    assert result["scenarios"][0]["status"] == "failed"
    assert result["robustness_result"] == "INSUFFICIENT"


def test_later_date_removal_does_not_change_earlier_neutralization():
    raw = _frame(np.arange(1, 37).reshape(6, 6))
    size = _frame(np.arange(1, 37).reshape(6, 6))
    full = neutralize_factor(raw, descriptors={"size": size})
    truncated = neutralize_factor(
        raw.iloc[:4], descriptors={"size": size.iloc[:4]}
    )
    pd.testing.assert_frame_equal(full.residual.iloc[:4], truncated.residual)

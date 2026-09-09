import pandas as pd
import pytest

from twse_factor_lab.analysis.factor_gate import (
    FactorGateConfig,
    FactorGateError,
    evaluate_factor,
)
from twse_factor_lab.governance import (
    DatasetManifest,
    ResearchManifest,
    load_experiment_registry,
    save_research_manifest,
)


def make_research() -> ResearchManifest:
    return ResearchManifest(
        research_id="factor-gate-test",
        hypothesis="factor admission fixture",
        factor_candidates=["pe", "roe"],
        universe="fixture",
        dataset_version="dataset-v1",
        is_start="2020-01-01",
        is_end="2020-01-03",
        oos_start="2020-01-04",
        oos_end="2020-01-06",
        rebalance_search_space=["daily"],
        top_n_search_space=[5],
        cost_scenarios=["none"],
        selection_relevant=True,
        status="active",
    )


def make_dataset() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="fixture-data",
        source="test",
        retrieved_at="2026-09-08T00:00:00+00:00",
        schema_version="1.0.0",
        processing_version="1.0.0",
        pit_rule="next_trading_day",
        row_count=100,
        coverage=1.0,
        artifact_sha256="a" * 64,
    )


def make_data(
    *,
    direction: str = "higher_is_better",
    inverse: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    tickers = ["A", "B", "C", "D", "E"]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    factor_values = scores if direction == "higher_is_better" else scores[::-1]
    factor_matrix = pd.DataFrame(
        [factor_values] * 3,
        index=dates[:3],
        columns=tickers,
    )
    returns = [
        [0.01, 0.02, 0.03, 0.04, 0.05],
        [0.01, 0.02, 0.03, 0.05, 0.04],
        [0.02, 0.01, 0.03, 0.04, 0.05],
    ]
    if inverse:
        returns = [row[::-1] for row in returns]
    close_rows = [[100.0] * 5]
    for row in returns:
        close_rows.append(
            [
                price * (1.0 + change)
                for price, change in zip(close_rows[-1], row, strict=True)
            ]
        )
    return factor_matrix, pd.DataFrame(close_rows, index=dates, columns=tickers)


def prepare_root(tmp_path):
    save_research_manifest(make_research(), tmp_path)
    return tmp_path


def config(**overrides) -> FactorGateConfig:
    values = {
        "horizons": (1,),
        "min_assets": 5,
        "min_ic_observations": 2,
        "min_coverage": 0.9,
    }
    values.update(overrides)
    return FactorGateConfig(**values)


def test_direction_normalizes_lower_and_higher_factors(tmp_path):
    factor_matrix, close_matrix = make_data(direction="lower_is_better")
    lower = evaluate_factor(
        factor_id="pe",
        research_id="factor-gate-test",
        experiment_id="exp-pe",
        root=prepare_root(tmp_path),
        factor_matrix=factor_matrix,
        close_matrix=close_matrix,
        dataset_manifest=make_dataset(),
        config=config(min_icir=0.0),
    )

    assert lower.direction == "lower_is_better"
    assert lower.mean_ic > 0
    assert lower.top_bottom_spread > 0
    assert lower.verdict == "ACCEPT"
    assert lower.pit_required is True
    assert lower.pit_rule == "next_trading_day"


def test_inverse_factor_has_negative_ic_and_is_rejected(tmp_path):
    factor_matrix, close_matrix = make_data(inverse=True)
    result = evaluate_factor(
        factor_id="roe",
        research_id="factor-gate-test",
        experiment_id="exp-inverse",
        root=prepare_root(tmp_path),
        factor_matrix=factor_matrix,
        close_matrix=close_matrix,
        dataset_manifest=make_dataset(),
        config=config(min_icir=0.0),
    )

    assert result.mean_ic < 0
    assert result.top_bottom_spread < 0
    assert result.verdict == "REJECT"
    assert result.reasons


def test_insufficient_assets_is_explicit_and_not_zero(tmp_path):
    factor_matrix, close_matrix = make_data()
    factor_matrix = factor_matrix.drop(columns="E")
    close_matrix = close_matrix.drop(columns="E")
    result = evaluate_factor(
        factor_id="roe",
        research_id="factor-gate-test",
        experiment_id="exp-small",
        root=prepare_root(tmp_path),
        factor_matrix=factor_matrix,
        close_matrix=close_matrix,
        dataset_manifest=make_dataset(),
        config=config(min_assets=5),
    )

    assert result.sample_status == "INSUFFICIENT"
    assert result.mean_ic is None
    assert result.icir is None
    assert result.verdict == "REJECT"
    assert any("effective assets" in reason for reason in result.reasons)


def test_missing_values_reduce_coverage_without_forward_fill(tmp_path):
    factor_matrix, close_matrix = make_data()
    factor_matrix.loc[pd.Timestamp("2020-01-02"), "E"] = float("nan")
    result = evaluate_factor(
        factor_id="roe",
        research_id="factor-gate-test",
        experiment_id="exp-missing",
        root=prepare_root(tmp_path),
        factor_matrix=factor_matrix,
        close_matrix=close_matrix,
        dataset_manifest=make_dataset(),
        config=config(min_assets=4, min_icir=0.0),
    )

    assert result.coverage < 1.0
    assert result.valid_observation_count < 15
    assert result.sample_status == "SUFFICIENT"


def test_positive_but_unstable_factor_is_candidate(tmp_path):
    factor_matrix, close_matrix = make_data()
    result = evaluate_factor(
        factor_id="roe",
        research_id="factor-gate-test",
        experiment_id="exp-candidate",
        root=prepare_root(tmp_path),
        factor_matrix=factor_matrix,
        close_matrix=close_matrix,
        dataset_manifest=make_dataset(),
        config=config(min_icir=100.0),
    )

    assert result.mean_ic > 0
    assert result.top_bottom_spread > 0
    assert result.verdict == "CANDIDATE"
    assert any("IC stability" in reason for reason in result.reasons)


def test_rank_stability_reports_low_turnover_and_high_autocorrelation(tmp_path):
    factor_matrix, close_matrix = make_data()
    result = evaluate_factor(
        factor_id="roe",
        research_id="factor-gate-test",
        experiment_id="exp-stability",
        root=prepare_root(tmp_path),
        factor_matrix=factor_matrix,
        close_matrix=close_matrix,
        dataset_manifest=make_dataset(),
        config=config(min_icir=0.0),
    )

    assert result.turnover == pytest.approx(0.0)
    assert result.rank_autocorrelation == pytest.approx(1.0)


def test_one_admission_is_one_selection_relevant_experiment(tmp_path):
    factor_matrix, close_matrix = make_data(inverse=True)
    root = prepare_root(tmp_path)
    result = evaluate_factor(
        factor_id="roe",
        research_id="factor-gate-test",
        experiment_id="exp-one",
        root=root,
        factor_matrix=factor_matrix,
        close_matrix=close_matrix,
        dataset_manifest=make_dataset(),
        config=config(min_icir=0.0),
    )

    [record] = load_experiment_registry(root, "factor-gate-test")
    assert record.status == "completed"
    assert record.selection_relevant is True
    assert record.experiment_type == "factor_test"
    assert record.config["factor_id"] == "roe"
    assert record.result["verdict"] == result.verdict


def test_failed_admission_is_retained(monkeypatch, tmp_path):
    factor_matrix, close_matrix = make_data()
    root = prepare_root(tmp_path)

    def fail(**_kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr("twse_factor_lab.analysis.factor_gate._run_diagnostics", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        evaluate_factor(
            factor_id="roe",
            research_id="factor-gate-test",
            experiment_id="exp-failed",
            root=root,
            factor_matrix=factor_matrix,
            close_matrix=close_matrix,
            dataset_manifest=make_dataset(),
            config=config(),
        )

    [record] = load_experiment_registry(root, "factor-gate-test")
    assert record.status == "failed"
    assert record.result["error"] == "fixture failure"
    assert record.selection_relevant is True


def test_unknown_factor_and_missing_provenance_fail_before_registration(tmp_path):
    factor_matrix, close_matrix = make_data()
    root = prepare_root(tmp_path)
    with pytest.raises(ValueError, match="unknown factor_id"):
        evaluate_factor(
            factor_id="unknown",
            research_id="factor-gate-test",
            experiment_id="exp-unknown",
            root=root,
            factor_matrix=factor_matrix,
            close_matrix=close_matrix,
            dataset_manifest=make_dataset(),
        )
    with pytest.raises(FactorGateError, match="provenance"):
        evaluate_factor(
            factor_id="roe",
            research_id="factor-gate-test",
            experiment_id="exp-no-data",
            root=root,
            factor_matrix=factor_matrix,
            close_matrix=close_matrix,
            dataset_manifest=None,
        )
    assert load_experiment_registry(root, "factor-gate-test") == []


def test_same_inputs_produce_identical_diagnostics(tmp_path):
    factor_matrix, close_matrix = make_data()
    first_root = prepare_root(tmp_path / "first")
    second_root = prepare_root(tmp_path / "second")
    kwargs = {
        "factor_id": "roe",
        "research_id": "factor-gate-test",
        "root": first_root,
        "factor_matrix": factor_matrix,
        "close_matrix": close_matrix,
        "dataset_manifest": make_dataset(),
        "config": config(min_icir=0.0),
    }
    first = evaluate_factor(experiment_id="exp-first", **kwargs)
    second = evaluate_factor(
        experiment_id="exp-second", **{**kwargs, "root": second_root}
    )
    assert first.to_dict() == second.to_dict()

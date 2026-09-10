import json

import pandas as pd
import pytest

from twse_factor_lab.acceptance.research_cycle import (
    ResearchCycleError,
    assert_no_lookahead,
    build_acceptance_matrix,
    build_research_trial_inventory,
    compute_statistical_acceptance,
    evaluate_oos,
    freeze_research_cycle,
    run_oos_evaluation,
    verify_research_freeze,
)
from twse_factor_lab.governance import (
    ExperimentRecord,
    ResearchManifest,
    load_experiment_registry,
    register_experiment,
    save_research_manifest,
    update_experiment_status,
)


def _manifest() -> ResearchManifest:
    return ResearchManifest(
        research_id="day7-test",
        hypothesis="acceptance fixture",
        factor_candidates=["pe", "roe"],
        universe="fixture",
        dataset_version="dataset-v1",
        is_start="2024-01-01",
        is_end="2024-01-05",
        oos_start="2024-01-08",
        oos_end="2024-01-12",
        rebalance_search_space=["daily"],
        top_n_search_space=[1, 2],
        cost_scenarios=["no_cost"],
        selection_relevant=True,
        status="active",
    )


def _prepare(tmp_path):
    save_research_manifest(_manifest(), tmp_path)
    return tmp_path


def _register(
    root,
    experiment_id,
    *,
    selection_relevant,
    config,
    status="completed",
    result=None,
    experiment_type=None,
):
    if experiment_type is None:
        experiment_type = "strategy_backtest" if selection_relevant else "diagnostic"
    register_experiment(
        ExperimentRecord(
            experiment_id=experiment_id,
            research_id="day7-test",
            config=config,
            dataset_version="dataset-v1",
            experiment_type=experiment_type,
            status=status,
            selection_relevant=selection_relevant,
            result=result or {},
        ),
        root,
    )


def _returns():
    return pd.Series(
        [0.50, -0.10, 0.02, 0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.03],
        index=pd.bdate_range("2024-01-01", periods=10),
    )


def _evidence(statuses):
    return dict(zip(
        (
            "factor_evidence",
            "strategy_incremental_value",
            "execution",
            "performance",
            "attribution",
            "robustness",
            "oos",
            "statistics",
        ),
        statuses,
        strict=True,
    ))


def test_oos_uses_fresh_capital_and_warmup_has_no_pnl():
    result = evaluate_oos(
        _returns(),
        oos_start="2024-01-08",
        oos_end="2024-01-12",
        initial_capital=1_000.0,
        strategy_id="candidate",
        research_id="day7-test",
        candidate_config={"top_n": 1},
    )
    assert result["initial_capital"] == 1_000.0
    assert result["initial_positions"] == {}
    assert result["inherited_is_state"] is False
    assert result["warmup_pnl"] == 0.0
    assert result["warmup_range"] == {"start": "2024-01-01", "end": "2024-01-05"}
    assert result["oos_nav"][0]["value"] == pytest.approx(980.0)
    assert result["oos_metrics"]["observation_count"] == 5


def test_no_lookahead_rejects_future_availability():
    valid = pd.DataFrame(
        {"signal_date": ["2024-01-05"], "available_date": ["2024-01-04"]}
    )
    assert_no_lookahead(valid)
    invalid = valid.copy()
    invalid.loc[0, "available_date"] = "2024-01-06"
    with pytest.raises(ResearchCycleError, match="future data"):
        assert_no_lookahead(invalid)


def test_oos_run_is_registered_diagnostic_and_research_scoped(tmp_path):
    root = _prepare(tmp_path)
    result = run_oos_evaluation(
        root=root,
        research_id="day7-test",
        strategy_id="candidate",
        experiment_id="oos-001",
        returns=_returns(),
        candidate_config={"strategy_id": "candidate", "top_n": 1},
        initial_capital=2_000.0,
    )
    assert result["artifact_path"].startswith("data/research/day7-test/")
    [record] = load_experiment_registry(root, "day7-test")
    assert record.experiment_type == "diagnostic"
    assert record.selection_relevant is False


def test_inventory_keeps_all_attempts_but_deduplicates_effective_configs(tmp_path):
    root = _prepare(tmp_path)
    config = {"strategy_id": "candidate", "top_n": 1}
    _register(
        root,
        "trial-completed",
        selection_relevant=True,
        config=config,
        result={"sharpe": 1.0},
    )
    _register(
        root, "trial-failed", selection_relevant=True, config=config, status="failed"
    )
    _register(
        root,
        "trial-aborted",
        selection_relevant=True,
        config={"strategy_id": "candidate", "top_n": 2},
        status="aborted",
    )
    _register(
        root,
        "diagnostic-pyfolio",
        selection_relevant=False,
        config={"kind": "pyfolio"},
    )
    inventory, effective = build_research_trial_inventory(root, "day7-test")
    assert set(inventory["status"]) == {"completed", "failed", "aborted"}
    assert len(inventory) == 4
    assert effective == 2


def test_statistics_reuses_canonical_psr_dsr_and_is_deterministic(tmp_path):
    root = _prepare(tmp_path)
    _register(
        root,
        "trial-a",
        selection_relevant=True,
        config={"strategy_id": "candidate", "top_n": 1},
        result={"sharpe": 1.0},
    )
    _register(
        root,
        "trial-b",
        selection_relevant=True,
        config={"strategy_id": "candidate", "top_n": 2},
        status="failed",
        result={"sharpe": 0.5},
    )
    inventory, _effective = build_research_trial_inventory(root, "day7-test")
    first = compute_statistical_acceptance(_returns().iloc[5:], inventory)
    second = compute_statistical_acceptance(_returns().iloc[5:], inventory)
    assert first == second
    assert first["effective_trials"] == 2
    assert first["strategy_selection_trial_count"] == 2
    assert first["method"] == "acceptance.psr-dsr-v1"
    assert first["psr"] is not None and first["dsr"] is not None


def test_dsr_population_excludes_factor_and_diagnostic_trials(tmp_path):
    root = _prepare(tmp_path)
    for i in range(10):
        _register(
            root,
            f"factor-{i}",
            selection_relevant=True,
            config={"factor_id": f"factor-{i}"},
            experiment_type="factor_test",
            result={"sharpe": 5.0},
        )
    for i in range(3):
        _register(
            root,
            f"strategy-{i}",
            selection_relevant=True,
            config={"strategy_id": "candidate", "top_n": i + 1},
            result={"sharpe": 1.0 + i},
        )
    for i in range(20):
        _register(
            root,
            f"diagnostic-{i}",
            selection_relevant=False,
            config={"kind": "diagnostic", "seq": i},
            experiment_type="diagnostic",
            result={"sharpe": 9.0},
        )

    inventory, _effective = build_research_trial_inventory(root, "day7-test")
    result = compute_statistical_acceptance(_returns().iloc[5:], inventory)

    assert result["factor_selection_trial_count"] == 10
    assert result["strategy_selection_trial_count"] == 3
    assert result["effective_trials"] == 3
    assert result["diagnostic_count"] == 20
    assert result["total_selection_relevant_experiments"] == 13


def test_missing_strategy_sharpe_fails_fast_without_imputation(tmp_path):
    root = _prepare(tmp_path)
    _register(
        root,
        "strategy-ok",
        selection_relevant=True,
        config={"strategy_id": "candidate", "top_n": 1},
        result={"sharpe": 1.0},
    )
    _register(
        root,
        "strategy-missing",
        selection_relevant=True,
        config={"strategy_id": "candidate", "top_n": 2},
        result={},
    )
    inventory, _effective = build_research_trial_inventory(root, "day7-test")
    with pytest.raises(ResearchCycleError, match="strategy-missing"):
        compute_statistical_acceptance(_returns().iloc[5:], inventory)


def test_acceptance_separates_platform_and_no_candidate_status():
    result = build_acceptance_matrix(_evidence(["UNAVAILABLE"] * 8))
    assert result["research_platform_verdict"] == "PASS"
    assert result["strategy_verdict"] == "NOT_EVALUATED"
    assert result["evaluation_status"] == "INFRASTRUCTURE_VALIDATION_ONLY"
    assert any("no real Day 3" in reason for reason in result["reasons"])


def test_mixed_real_evidence_is_candidate_not_fixture_accept():
    result = build_acceptance_matrix(
        _evidence(
            ["PASS", "PASS", "PASS", "UNAVAILABLE", "PASS", "MIXED", "PASS", "PASS"]
        ),
        real_candidate=True,
    )
    assert result["research_platform_verdict"] == "PASS"
    assert result["strategy_verdict"] == "CANDIDATE"


def test_terminal_experiment_is_byte_immutable(tmp_path):
    root = _prepare(tmp_path)
    _register(
        root,
        "terminal",
        selection_relevant=True,
        config={"top_n": 1},
        status="running",
    )
    update_experiment_status(
        root, "day7-test", "terminal", "completed", {"sharpe": 1.0}
    )
    path = tmp_path / "data/research/day7-test/experiment_registry.json"
    before = path.read_bytes()
    with pytest.raises(Exception, match="terminal records are immutable"):
        update_experiment_status(
            root, "day7-test", "terminal", "failed", {"sharpe": -1.0}
        )
    assert path.read_bytes() == before


def test_freeze_hashes_are_self_consistent_and_tamper_detected(tmp_path):
    root = _prepare(tmp_path)
    _register(root, "terminal", selection_relevant=True, config={"top_n": 1})
    acceptance = build_acceptance_matrix(_evidence(["UNAVAILABLE"] * 8))
    frozen = freeze_research_cycle(
        root=root,
        research_id="day7-test",
        candidate_config=None,
        acceptance=acceptance,
    )
    assert frozen["status"] == "PASS"
    freeze = tmp_path / "data/research/day7-test/freeze"
    expected = {
        "research_freeze_manifest.json",
        "reproducibility_manifest.json",
        "acceptance_handoff.json",
        "trial_inventory.json",
        "artifact_hashes.json",
    }
    assert expected == {path.name for path in freeze.iterdir()}
    assert verify_research_freeze(root, "day7-test")["status"] == "PASS"
    (tmp_path / "data/research/day7-test/research_manifest.json").write_text(
        json.dumps({"tampered": True}), encoding="utf-8"
    )
    with pytest.raises(ResearchCycleError, match="hash mismatch"):
        verify_research_freeze(root, "day7-test")


def test_verify_freeze_detects_unexpected_and_missing_files(tmp_path):
    root = _prepare(tmp_path)
    _register(root, "terminal", selection_relevant=True, config={"top_n": 1})
    acceptance = build_acceptance_matrix(_evidence(["UNAVAILABLE"] * 8))
    freeze_research_cycle(
        root=root,
        research_id="day7-test",
        candidate_config=None,
        acceptance=acceptance,
    )
    research_dir = tmp_path / "data/research/day7-test"
    assert verify_research_freeze(root, "day7-test")["status"] == "PASS"

    (research_dir / "freeze" / "unexpected_extra.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ResearchCycleError, match="unexpected file"):
        verify_research_freeze(root, "day7-test")
    (research_dir / "freeze" / "unexpected_extra.json").unlink()
    assert verify_research_freeze(root, "day7-test")["status"] == "PASS"

    (research_dir / "dataset_manifests.json").unlink(missing_ok=True)
    (research_dir / "research_manifest.json").unlink()
    with pytest.raises(ResearchCycleError, match="frozen artifact missing"):
        verify_research_freeze(root, "day7-test")


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/research/day7-test/research_manifest.json",
        "data/research/day7-test/experiment_registry.json",
        "data/research/day7-test/freeze/acceptance_handoff.json",
    ],
)
def test_tamper_on_any_frozen_artifact_fails_verify(tmp_path, relative_path):
    root = _prepare(tmp_path)
    _register(root, "terminal", selection_relevant=True, config={"top_n": 1})
    acceptance = build_acceptance_matrix(_evidence(["UNAVAILABLE"] * 8))
    freeze_research_cycle(
        root=root,
        research_id="day7-test",
        candidate_config=None,
        acceptance=acceptance,
    )
    assert verify_research_freeze(root, "day7-test")["status"] == "PASS"
    (tmp_path / relative_path).write_text(
        json.dumps({"tampered": True}), encoding="utf-8"
    )
    with pytest.raises(ResearchCycleError, match="hash mismatch"):
        verify_research_freeze(root, "day7-test")


def test_freeze_records_reproducibility_metadata(tmp_path):
    root = _prepare(tmp_path)
    _register(root, "terminal", selection_relevant=True, config={"top_n": 1})
    acceptance = build_acceptance_matrix(_evidence(["UNAVAILABLE"] * 8))
    freeze_research_cycle(
        root=root,
        research_id="day7-test",
        candidate_config=None,
        acceptance=acceptance,
    )
    manifest = json.loads(
        (
            tmp_path / "data/research/day7-test/freeze/research_freeze_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["code_revision"]
    assert manifest["git_dirty"] in (True, False)
    assert manifest["python_version"]
    assert manifest["dependency_snapshot"]["numpy"]
    assert manifest["dependency_snapshot"]["backtrader"]


def test_freeze_requires_clean_tree_when_formal(tmp_path, monkeypatch):
    import twse_factor_lab.acceptance.research_cycle as research_cycle

    root = _prepare(tmp_path)
    _register(root, "terminal", selection_relevant=True, config={"top_n": 1})
    acceptance = build_acceptance_matrix(_evidence(["UNAVAILABLE"] * 8))
    monkeypatch.setattr(
        research_cycle,
        "_git_state",
        lambda: {"code_revision": "deadbeef", "git_dirty": True},
    )
    with pytest.raises(ResearchCycleError, match="clean git tree"):
        freeze_research_cycle(
            root=root,
            research_id="day7-test",
            candidate_config=None,
            acceptance=acceptance,
            require_clean_tree=True,
        )
    assert not (
        tmp_path / "data/research/day7-test/freeze/research_freeze_manifest.json"
    ).exists()

    monkeypatch.setattr(
        research_cycle,
        "_git_state",
        lambda: {"code_revision": "deadbeef", "git_dirty": False},
    )
    result = freeze_research_cycle(
        root=root,
        research_id="day7-test",
        candidate_config=None,
        acceptance=acceptance,
        require_clean_tree=True,
    )
    assert result["status"] == "PASS"


def test_freeze_locks_registry_and_stays_in_research_namespace(tmp_path):
    root = _prepare(tmp_path)
    _register(root, "terminal", selection_relevant=False, config={"kind": "diagnostic"})
    freeze_research_cycle(
        root=root,
        research_id="day7-test",
        candidate_config=None,
        acceptance=build_acceptance_matrix(_evidence(["UNAVAILABLE"] * 8)),
    )
    with pytest.raises(Exception, match="research cycle is frozen"):
        register_experiment(
            ExperimentRecord(
                experiment_id="late",
                research_id="day7-test",
                config={},
                dataset_version="dataset-v1",
                experiment_type="diagnostic",
                status="completed",
                selection_relevant=False,
            ),
            root,
        )

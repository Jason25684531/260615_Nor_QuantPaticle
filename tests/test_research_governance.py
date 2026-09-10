from dataclasses import replace

import pytest

from twse_factor_lab.governance import (
    DatasetManifest,
    ExperimentRecord,
    GovernanceError,
    IsolationError,
    ResearchManifest,
    add_dataset_manifest,
    assert_write_allowed,
    load_dataset_manifests,
    load_experiment_registry,
    load_research_manifest,
    register_experiment,
    research_dir,
    save_research_manifest,
    update_experiment_status,
)


def make_manifest(**overrides) -> ResearchManifest:
    base = ResearchManifest(
        research_id="mvp-v1",
        hypothesis="momentum plus quality improves OOS sharpe",
        factor_candidates=["momentum_40d", "roe"],
        universe="twse-partial",
        dataset_version="ds-2026-09",
        is_start="2018-01-02",
        is_end="2022-12-31",
        oos_start="2023-01-01",
        oos_end="2025-12-31",
        rebalance_search_space=["daily", "weekly", "monthly"],
        top_n_search_space=[10, 20, 30],
        cost_scenarios=["no_cost", "base_cost", "high_cost"],
        selection_relevant=True,
        status="draft",
    )
    return replace(base, **overrides)


def make_dataset(**overrides) -> DatasetManifest:
    base = DatasetManifest(
        dataset_id="ohlcv-2026-09",
        source="twse_openapi",
        retrieved_at="2026-09-08T00:00:00+00:00",
        schema_version="1.0.0",
        processing_version="1.0.0",
        pit_rule="next_trading_day",
        row_count=1000,
        coverage=0.93,
        artifact_sha256="a" * 64,
    )
    return replace(base, **overrides)


def make_experiment(**overrides) -> ExperimentRecord:
    base = ExperimentRecord(
        experiment_id="exp-001",
        research_id="mvp-v1",
        config={"top_n": 20, "rebalance": "daily"},
        dataset_version="ds-2026-09",
        experiment_type="factor_test",
        status="planned",
        selection_relevant=True,
    )
    return replace(base, **overrides)


def test_research_manifest_round_trip(tmp_path):
    manifest = make_manifest()
    path = save_research_manifest(manifest, tmp_path)
    assert path == research_dir(tmp_path, "mvp-v1") / "research_manifest.json"
    assert load_research_manifest(tmp_path, "mvp-v1") == manifest


def test_missing_research_id_fails_fast(tmp_path):
    with pytest.raises(GovernanceError):
        save_research_manifest(make_manifest(research_id=""), tmp_path)
    assert not (tmp_path / "data").exists()


def test_missing_dataset_version_fails_fast(tmp_path):
    with pytest.raises(GovernanceError):
        save_research_manifest(make_manifest(dataset_version=""), tmp_path)
    assert not (tmp_path / "data").exists()


def test_duplicate_research_id_rejected(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    original = load_research_manifest(tmp_path, "mvp-v1")
    with pytest.raises(GovernanceError, match="duplicate research_id"):
        save_research_manifest(make_manifest(hypothesis="other"), tmp_path)
    assert load_research_manifest(tmp_path, "mvp-v1") == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_start": "2023-01-01"},  # IS starts after it ends
        {"oos_start": "2022-06-30"},  # OOS starts inside IS
        {"oos_end": "2022-12-30"},  # OOS ends before it starts
    ],
)
def test_invalid_is_oos_ordering_fails_fast(tmp_path, overrides):
    with pytest.raises(GovernanceError, match="IS/OOS"):
        save_research_manifest(make_manifest(**overrides), tmp_path)
    assert not (tmp_path / "data").exists()


def test_unknown_research_status_fails_fast(tmp_path):
    with pytest.raises(GovernanceError, match="status"):
        save_research_manifest(make_manifest(status="mystery"), tmp_path)


def test_frozen_locations_refuse_writes(tmp_path):
    for frozen in [
        "strategy_freeze_manifest.json",
        "final_acceptance_handoff.json",
        "research_trial_inventory.parquet",
        "data/processed/composite_scores.parquet",
        "reports/final/final_research_report.md",
        "openspec/changes/archive/anything.md",
    ]:
        with pytest.raises(IsolationError):
            assert_write_allowed(frozen, tmp_path)


def test_research_namespace_write_allowed(tmp_path):
    target = assert_write_allowed("data/research/mvp-v1/x.json", tmp_path)
    assert target == (tmp_path / "data/research/mvp-v1/x.json").resolve()


def test_write_outside_root_refused(tmp_path):
    with pytest.raises(IsolationError):
        assert_write_allowed(tmp_path.parent / "escape.json", tmp_path)


def test_frozen_rc1_research_identity_rejected(tmp_path):
    with pytest.raises(IsolationError, match="frozen RC1"):
        save_research_manifest(
            make_manifest(research_id="multi-factor-research-v1"), tmp_path
        )


def test_frozen_cycle_blocks_dataset_manifest_writes(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    freeze_dir = research_dir(tmp_path, "mvp-v1") / "freeze"
    freeze_dir.mkdir(parents=True)
    (freeze_dir / "research_freeze_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IsolationError, match="frozen"):
        add_dataset_manifest(make_dataset(), tmp_path, "mvp-v1")


def test_frozen_cycle_blocks_any_write_under_its_namespace(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    freeze_dir = research_dir(tmp_path, "mvp-v1") / "freeze"
    freeze_dir.mkdir(parents=True)
    (freeze_dir / "research_freeze_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IsolationError, match="frozen"):
        assert_write_allowed("data/research/mvp-v1/anything.json", tmp_path)


def test_frozen_cycle_registry_is_immutable(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    register_experiment(make_experiment(), tmp_path)
    freeze_dir = research_dir(tmp_path, "mvp-v1") / "freeze"
    freeze_dir.mkdir(parents=True)
    (freeze_dir / "research_freeze_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IsolationError, match="frozen"):
        register_experiment(make_experiment(experiment_id="exp-002"), tmp_path)
    with pytest.raises(IsolationError, match="frozen"):
        update_experiment_status(tmp_path, "mvp-v1", "exp-001", "running")
    [record] = load_experiment_registry(tmp_path, "mvp-v1")
    assert record == make_experiment()


def test_dataset_manifest_round_trip(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    dataset = make_dataset()
    add_dataset_manifest(dataset, tmp_path, "mvp-v1")
    assert load_dataset_manifests(tmp_path, "mvp-v1") == [dataset]


def test_missing_dataset_id_fails_fast(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    with pytest.raises(GovernanceError):
        add_dataset_manifest(make_dataset(dataset_id=""), tmp_path, "mvp-v1")
    assert load_dataset_manifests(tmp_path, "mvp-v1") == []


def test_missing_artifact_hash_fails_fast(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    with pytest.raises(GovernanceError, match="artifact_sha256"):
        add_dataset_manifest(make_dataset(artifact_sha256=""), tmp_path, "mvp-v1")


def test_duplicate_dataset_id_rejected(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    add_dataset_manifest(make_dataset(), tmp_path, "mvp-v1")
    with pytest.raises(GovernanceError, match="duplicate dataset_id"):
        add_dataset_manifest(make_dataset(source="other"), tmp_path, "mvp-v1")
    assert load_dataset_manifests(tmp_path, "mvp-v1") == [make_dataset()]


def test_experiment_registered_and_linked(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    record = make_experiment()
    register_experiment(record, tmp_path)
    assert load_experiment_registry(tmp_path, "mvp-v1") == [record]


def test_duplicate_experiment_id_rejected(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    register_experiment(make_experiment(), tmp_path)
    with pytest.raises(GovernanceError, match="duplicate experiment_id"):
        register_experiment(make_experiment(status="running"), tmp_path)
    assert load_experiment_registry(tmp_path, "mvp-v1") == [make_experiment()]


def test_failed_and_aborted_experiments_recordable(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    register_experiment(
        make_experiment(experiment_id="exp-fail", status="failed"), tmp_path
    )
    register_experiment(
        make_experiment(experiment_id="exp-abort", status="aborted"), tmp_path
    )
    statuses = {
        record.status for record in load_experiment_registry(tmp_path, "mvp-v1")
    }
    assert statuses == {"failed", "aborted"}


def test_selection_relevant_vs_diagnostic_distinguishable(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    register_experiment(make_experiment(experiment_id="exp-sel"), tmp_path)
    register_experiment(
        make_experiment(
            experiment_id="exp-diag",
            experiment_type="diagnostic",
            selection_relevant=False,
        ),
        tmp_path,
    )
    registry = load_experiment_registry(tmp_path, "mvp-v1")
    selection = [r.experiment_id for r in registry if r.selection_relevant]
    diagnostic = [r.experiment_id for r in registry if not r.selection_relevant]
    assert selection == ["exp-sel"]
    assert diagnostic == ["exp-diag"]


def test_unknown_experiment_status_fails_fast(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    with pytest.raises(GovernanceError, match="experiment status"):
        register_experiment(make_experiment(status="mystery"), tmp_path)
    assert load_experiment_registry(tmp_path, "mvp-v1") == []


def test_unknown_experiment_type_fails_fast(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    with pytest.raises(GovernanceError, match="experiment type"):
        register_experiment(make_experiment(experiment_type="mystery"), tmp_path)


def test_missing_research_linkage_fails_fast(tmp_path):
    with pytest.raises(GovernanceError, match="research manifest not found"):
        register_experiment(make_experiment(), tmp_path)


def test_update_experiment_status(tmp_path):
    save_research_manifest(make_manifest(), tmp_path)
    register_experiment(make_experiment(), tmp_path)
    update_experiment_status(tmp_path, "mvp-v1", "exp-001", "completed")
    [record] = load_experiment_registry(tmp_path, "mvp-v1")
    assert record.status == "completed"
    with pytest.raises(GovernanceError, match="unknown experiment status"):
        update_experiment_status(tmp_path, "mvp-v1", "exp-001", "mystery")
    with pytest.raises(GovernanceError, match="unknown experiment_id"):
        update_experiment_status(tmp_path, "mvp-v1", "exp-404", "completed")
    with pytest.raises(GovernanceError, match="terminal records are immutable"):
        update_experiment_status(
            tmp_path, "mvp-v1", "exp-001", "failed", {"result": "rewrite"}
        )

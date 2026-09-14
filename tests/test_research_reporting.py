"""Focused contracts for the controlled research report layer."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from twse_factor_lab.acceptance.research_cycle import verify_research_freeze
from twse_factor_lab.reporting import provenance
from twse_factor_lab.reporting.figures import build_figures
from twse_factor_lab.reporting.loader import (
    ReportLoaderError,
    load_research_report_model,
)
from twse_factor_lab.reporting.provenance import (
    REPORT_SCHEMA_VERSION,
    freeze_manifest_sha256,
)
from twse_factor_lab.reporting.render_html import render_html
from twse_factor_lab.reporting.render_md import render_markdown
from twse_factor_lab.reporting.runner import generate_research_report


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path: Path, frozen: bool = False) -> tuple[Path, Path]:
    root = tmp_path
    cycle = root / "data" / "research" / "fixture-cycle"
    _write(
        cycle / "research_manifest.json",
        {
            "research_id": "fixture-cycle",
            "dataset_version": "fixture-v1",
            "is_start": "2020-01-01",
            "is_end": "2020-12-31",
            "oos_start": "2021-01-01",
            "oos_end": "2021-12-31",
            "factor_candidates": ["bad_factor", "good_factor"],
            "universe": "fixture-universe",
            "hypothesis": "fixture hypothesis",
        },
    )
    registry = [
        {
            "experiment_id": "factor-gate-bad_factor",
            "experiment_type": "factor_test",
            "selection_relevant": True,
            "config": {"factor_id": "bad_factor"},
            "result": {
                "factor_id": "bad_factor",
                "direction": "higher_is_better",
                "primary_horizon": 20,
                "horizon_metrics": [
                    {
                        "horizon": 20,
                        "mean_ic": -0.1,
                        "icir": -0.2,
                        "positive_ic_ratio": 0.2,
                        "top_bottom_spread": -0.1,
                        "coverage": 0.5,
                        "turnover": 0.2,
                        "rank_autocorrelation": 0.9,
                    }
                ],
                "verdict": "REJECT",
                "reasons": ["weak signal"],
            },
        },
        {
            "experiment_id": "factor-gate-good_factor",
            "experiment_type": "factor_test",
            "selection_relevant": True,
            "config": {"factor_id": "good_factor"},
            "result": {
                "factor_id": "good_factor",
                "direction": "higher_is_better",
                "primary_horizon": 20,
                "horizon_metrics": [
                    {
                        "horizon": 20,
                        "mean_ic": 0.1,
                        "icir": 0.4,
                        "positive_ic_ratio": 0.8,
                        "top_bottom_spread": 0.1,
                        "coverage": 0.9,
                        "turnover": 0.1,
                        "rank_autocorrelation": 0.8,
                    }
                ],
                "verdict": "ACCEPT",
            },
        },
        {
            "experiment_id": "strategy-is-trial",
            "experiment_type": "strategy_backtest",
            "selection_relevant": True,
            "config": {
                "strategy_id": "is-trial",
                "factor_ids": ["good_factor"],
                "factor_weights": {"good_factor": 1.0},
                "top_n": 1,
                "rebalance_frequency": "daily",
                "cost_scenario": "no_cost",
            },
            "result": {
                "strategy_id": "is-trial",
                "factor_ids": ["good_factor"],
                "factor_weights": {"good_factor": 1.0},
                "top_n": 1,
                "rebalance_frequency": "daily",
                "cost_scenario": "no_cost",
                "total_return": 0.2,
                "sharpe": 0.5,
                "max_drawdown": -0.1,
                "status": "completed",
            },
        },
        {
            "experiment_id": "diagnostic-only",
            "experiment_type": "diagnostic",
            "selection_relevant": False,
            "result": {"status": "completed"},
        },
    ]
    _write(cycle / "experiment_registry.json", registry)
    _write(
        cycle / "acceptance_matrix.json",
        {
            "research_platform_verdict": "PASS",
            "strategy_verdict": "REJECT",
            "strategy_verdict_available": True,
            "reasons": ["fixture rejection"],
            "candidate_config": {"strategy_id": "is-trial"},
            "sections": {
                "factor_evidence": {"status": "PASS"},
                "statistics": {
                    "status": "REJECT",
                    "evidence": {
                        "factor_selection_trial_count": 9,
                        "strategy_selection_trial_count": 8,
                        "diagnostic_count": 7,
                        "effective_trials": 8,
                        "total_selection_relevant_experiments": 17,
                        "psr": 0.2,
                        "dsr": 0.1,
                    },
                },
            },
        },
    )
    handoff = cycle / "strategy_handoffs" / "is-trial__strategy-is-trial"
    handoff.mkdir(parents=True)
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    pd.DataFrame({"date": dates, "returns": [0.01] * 8}).to_csv(
        handoff / "returns.csv", index=False
    )
    pd.DataFrame(
        {"date": dates, "nav": [100, 101, 102, 103, 104, 105, 106, 107]}
    ).to_csv(handoff / "nav.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "AAA": [0, 50, 50, 50, 50, 50, 50, 50],
            "cash": [100, 51, 52, 53, 54, 55, 56, 57],
        }
    ).to_csv(handoff / "positions.csv", index=False)
    perf = cycle / "performance" / "is-trial"
    _write(
        perf / "canonical_metrics.json",
        {"total_return": 0.07, "cagr": 0.05, "sharpe": 0.5, "max_drawdown": -0.1},
    )
    _write(
        perf / "pyfolio_metadata.json",
        {
            "annualization_sessions": 252,
            "cross_check": {
                "checks": [
                    {
                        "metric": "sharpe",
                        "canonical": 0.5,
                        "pyfolio": 0.6,
                        "difference": 0.1,
                        "status": "DEFINITION_DIFFERENCE",
                    }
                ],
                "status": "PASS_WITH_DEFINITION_DIFFERENCE",
            },
            "transactions": {"available": False, "status": "UNAVAILABLE"},
        },
    )
    _write(perf / "pyfolio_metrics.json", {"sharpe": 0.6})
    _write(
        cycle
        / "execution_validation"
        / "is-trial"
        / "tri-engine-is-trial"
        / "parity.json",
        {
            "status": "PASS",
            "engines": {
                "custom": "custom",
                "vectorbt": "fallback",
                "backtrader": "backtrader",
            },
            "comparisons": [],
        },
    )
    _write(cycle / "attribution" / "is-trial" / "attribution.json", {"status": "PASS"})
    _write(
        cycle / "robustness" / "is-trial" / "robustness-is-trial.json",
        {"status": "MIXED", "robustness_result": "MIXED", "scenarios": []},
    )
    _write(
        cycle / "oos" / "is-trial" / "fresh_state_oos.json",
        {
            "evidence_type": "fresh_state_reexecution",
            "oos_start": "2021-01-04",
            "oos_end": "2021-01-08",
            "oos_nav": [{"date": "2021-01-04", "value": 100}],
            "initial_cash": 100,
            "initial_positions": {},
        },
    )
    _write(
        cycle / "oos" / "is-trial" / "legacy_sliced_oos.json",
        {"evidence_type": "legacy_sliced"},
    )
    if frozen:
        freeze_root = cycle / "freeze"
        _write(
            freeze_root / "trial_inventory.json",
            {"effective_trials": 1, "records": [], "research_id": "fixture-cycle"},
        )
        source_files = [
            file
            for file in cycle.rglob("*")
            if file.is_file() and freeze_root not in file.parents
        ]
        hashes = {
            file.relative_to(root).as_posix(): hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for file in source_files
        }
        _write(freeze_root / "artifact_hashes.json", {"artifacts": hashes})
        inventory = sorted(
            file.relative_to(root).as_posix()
            for file in cycle.rglob("*")
            if file.is_file()
        )
        freeze = freeze_root / "research_freeze_manifest.json"
        _write(
            freeze,
            {
                "candidate_config": {"strategy_id": "is-trial"},
                "code_revision": "fixture-code",
                "git_dirty": False,
                "freeze_version": "fixture-v1",
                "candidate_fingerprint": "fixture-fingerprint",
                "dependency_snapshot": {"pyfolio": None},
                "artifact_hashes": hashes,
                "complete_inventory": inventory,
                "effective_trials": 1,
            },
        )
    return root, cycle


def _snapshot(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in path.rglob("*")
        if file.is_file()
    }


def test_loader_is_deterministic_and_preserves_rejects(tmp_path: Path):
    root, _ = _fixture(tmp_path)
    first = load_research_report_model(root, "fixture-cycle")
    second = load_research_report_model(root, "fixture-cycle")
    assert first.to_dict() == second.to_dict()
    assert len(first.factor_evidence.data["rows"]) == 2
    assert any(row["verdict"] == "REJECT" for row in first.factor_evidence.data["rows"])
    assert first.acceptance.data["strategy_verdict"] == "REJECT"
    assert first.execution_validation.data["parity_status"] == "FALLBACK"
    assert first.performance.data["portfolio_exposure"]["status"] == "AVAILABLE"
    assert first.statistics.data["total_selection_relevant_trials"] == 17
    assert first.statistics.data["dsr_effective_strategy_trials"] == 8
    assert first.statistics.data["diagnostic_trial_count"] == 7
    assert first.statistics.data["trial_count_consistency"] == "PASS"
    assert first.attribution.status == "UNAVAILABLE"
    assert first.attribution.data["execution_status"] == "PASS"
    assert first.attribution.data["evidence_status"] == "UNAVAILABLE"


def test_loader_missing_optional_is_honest_and_mandatory_is_explicit(tmp_path: Path):
    root, cycle = _fixture(tmp_path)
    (cycle / "attribution" / "is-trial" / "attribution.json").unlink()
    (cycle / "robustness" / "is-trial" / "robustness-is-trial.json").unlink()
    (
        cycle
        / "execution_validation"
        / "is-trial"
        / "tri-engine-is-trial"
        / "parity.json"
    ).unlink()
    model = load_research_report_model(root, "fixture-cycle")
    assert model.attribution.status == "UNAVAILABLE"
    assert model.robustness.status == "UNAVAILABLE"
    assert model.execution_validation.status == "UNAVAILABLE"
    (cycle / "experiment_registry.json").unlink()
    with pytest.raises(ReportLoaderError, match="experiment_registry.json"):
        load_research_report_model(root, "fixture-cycle")


def test_all_figures_are_headless_non_leaking_and_deterministic(tmp_path: Path):
    root, _ = _fixture(tmp_path)
    model = load_research_report_model(root, "fixture-cycle")
    first = build_figures(model, tmp_path / "out-1" / "figures")
    second = build_figures(model, tmp_path / "out-2" / "figures")
    assert [item.status for item in first] == ["GENERATED"] * 7
    assert all(item.path and item.path.stat().st_size > 0 for item in first)
    assert [item.path.read_bytes() for item in first] == [
        item.path.read_bytes() for item in second
    ]
    assert first[2].metadata["rolling_window"] == 126
    assert first[3].metadata["annualization_sessions"] == 252
    assert first[5].metadata["net_equals_gross"] is True
    assert first[6].metadata["oos_evidence_type"] == "fresh_state_reexecution"
    assert plt.get_fignums() == []


def test_renderers_keep_verdicts_and_cross_check_vocabulary(tmp_path: Path):
    root, _ = _fixture(tmp_path)
    model = load_research_report_model(root, "fixture-cycle")
    markdown = render_markdown(model)
    rendered_html = render_html(model)
    # 16 v1 headings + 6 Research Cycle v2 sections (UNAVAILABLE for a v1 cycle).
    assert markdown.count("\n## ") == 22
    assert rendered_html.count("<h2>") == 22
    # v2 sections render gracefully as UNAVAILABLE for a v1 cycle.
    assert "Market Breadth Risk Overlay" in markdown
    assert "Performance Metric Definition" in rendered_html
    for rendered in (markdown, rendered_html):
        assert "Platform" in rendered and "PASS" in rendered
        assert "Strategy" in rendered and "REJECT" in rendered
        assert "DEFINITION_DIFFERENCE" in rendered
        if rendered is markdown:
            assert "Parity verdict: FALLBACK" in rendered
        assert "0.5" in rendered and "0.6" in rendered
        assert "Factor Selection Trials: 9" in rendered
        assert "Strategy Selection Trials: 8" in rendered
        assert "Total Selection-Relevant Trials: 17" in rendered
        assert "DSR Effective Strategy Trials: 8" in rendered
        assert "Diagnostic Trials: 7" in rendered
        assert "Effective trials" not in rendered
        assert "Pipeline Status: PASS" in rendered
        assert "Evidence Status: UNAVAILABLE" in rendered
        assert "Canonical metrics are authoritative." in rendered
        assert "Pyfolio is an independent diagnostic." in rendered
        assert (
            "DEFINITION_DIFFERENCE does not indicate execution parity failure."
            in rendered
        )


def test_trial_counts_are_named_consistently_in_outputs(tmp_path: Path):
    root, _ = _fixture(tmp_path)
    result = generate_research_report(root, "fixture-cycle")
    markdown = result.files["research_report.md"].read_text(encoding="utf-8")
    rendered_html = result.files["research_report.html"].read_text(encoding="utf-8")
    summary = json.loads(result.files["summary.json"].read_text(encoding="utf-8"))
    provenance_data = json.loads(
        result.files["provenance.json"].read_text(encoding="utf-8")
    )
    expected = {
        "factor_selection_trial_count": 9,
        "strategy_selection_trial_count": 8,
        "total_selection_relevant_trials": 17,
        "dsr_effective_strategy_trials": 8,
        "diagnostic_trial_count": 7,
        "trial_count_consistency": "PASS",
    }
    labels = {
        "factor_selection_trial_count": "Factor Selection Trials",
        "strategy_selection_trial_count": "Strategy Selection Trials",
        "total_selection_relevant_trials": "Total Selection-Relevant Trials",
        "dsr_effective_strategy_trials": "DSR Effective Strategy Trials",
        "diagnostic_trial_count": "Diagnostic Trials",
    }
    assert all(
        f"{label}: {expected[key]}" in markdown
        for key, label in labels.items()
    )
    assert "Factor Selection Trials: 9" in rendered_html
    assert "Strategy Selection Trials: 8" in rendered_html
    assert "Total Selection-Relevant Trials: 17" in rendered_html
    assert "DSR Effective Strategy Trials: 8" in rendered_html
    assert "Diagnostic Trials: 7" in rendered_html
    assert "Effective trials" not in markdown + rendered_html
    for key, value in expected.items():
        assert summary[key] == value
        assert provenance_data[key] == value
    assert summary["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert provenance_data["report_schema_version"] == REPORT_SCHEMA_VERSION


def test_trial_count_mismatch_is_reported_without_rewriting_evidence(tmp_path: Path):
    root, cycle = _fixture(tmp_path)
    matrix_path = cycle / "acceptance_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["sections"]["statistics"]["evidence"][
        "total_selection_relevant_experiments"
    ] = 99
    _write(matrix_path, matrix)
    model = load_research_report_model(root, "fixture-cycle")
    assert model.statistics.data["total_selection_relevant_trials"] == 99
    assert model.statistics.data["trial_count_consistency"] == "MISMATCH"


def test_freeze_manifest_sha_uses_exact_bytes(tmp_path: Path):
    root, cycle = _fixture(tmp_path, frozen=True)
    manifest = cycle / "freeze" / "research_freeze_manifest.json"
    original = manifest.read_bytes()
    expected = hashlib.sha256(original).hexdigest()
    model = load_research_report_model(root, "fixture-cycle")
    assert freeze_manifest_sha256(manifest) == expected
    assert model.reproducibility.data["source_freeze_manifest_sha256"] == expected
    manifest.write_bytes(original.replace(b"fixture-v1", b"fixture-v2", 1))
    try:
        changed = load_research_report_model(root, "fixture-cycle")
        assert changed.reproducibility.data["source_freeze_manifest_sha256"] != expected
    finally:
        manifest.write_bytes(original)


def test_dependency_resolution_prefers_installed_distribution(
    monkeypatch: pytest.MonkeyPatch,
):
    def reloaded_only(name: str) -> str:
        if name == "pyfolio-reloaded":
            return "0.9.7"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "version", reloaded_only)
    assert provenance.resolve_dependency_version(
        "pyfolio", ("pyfolio-reloaded", "pyfolio")
    ) == {
        "import_name": "pyfolio",
        "distribution": "pyfolio-reloaded",
        "version": "0.9.7",
    }

    def none_installed(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "version", none_installed)
    assert provenance.resolve_dependency_version(
        "pyfolio", ("pyfolio-reloaded", "pyfolio")
    ) == {"import_name": "pyfolio", "distribution": None, "version": None}


def test_barra_execution_and_evidence_statuses(tmp_path: Path):
    root, cycle = _fixture(tmp_path)
    attribution_path = cycle / "attribution" / "is-trial" / "attribution.json"
    model = load_research_report_model(root, "fixture-cycle")
    assert model.attribution.data["execution_status"] == "PASS"
    assert model.attribution.data["evidence_status"] == "UNAVAILABLE"

    _write(
        attribution_path,
        {
            "status": "PASS",
            "available_descriptors": ["value"],
            "unavailable_descriptors": ["size"],
        },
    )
    model = load_research_report_model(root, "fixture-cycle")
    assert model.attribution.status == "PARTIAL"
    assert model.attribution.data["execution_status"] == "PASS"
    assert model.attribution.data["evidence_status"] == "PARTIAL"

    _write(
        attribution_path,
        {
            "status": "PASS",
            "available_descriptors": ["value"],
            "unavailable_descriptors": [],
        },
    )
    model = load_research_report_model(root, "fixture-cycle")
    assert model.attribution.status == "AVAILABLE"
    assert model.attribution.data["evidence_status"] == "AVAILABLE"

    _write(attribution_path, {"status": "INSUFFICIENT_DATA"})
    model = load_research_report_model(root, "fixture-cycle")
    assert model.attribution.status == "INSUFFICIENT_DATA"
    assert model.attribution.data["execution_status"] == "INSUFFICIENT_DATA"
    assert model.attribution.data["evidence_status"] == "INSUFFICIENT_DATA"


def test_frozen_cycle_without_manifest_fails_fast(tmp_path: Path):
    root, cycle = _fixture(tmp_path)
    (cycle / "freeze").mkdir()
    with pytest.raises(ReportLoaderError, match="missing source freeze manifest"):
        generate_research_report(root, "fixture-cycle")


def test_frozen_report_is_external_and_does_not_change_cycle(tmp_path: Path):
    root, cycle = _fixture(tmp_path, frozen=True)
    freeze = cycle / "freeze" / "research_freeze_manifest.json"
    assert verify_research_freeze(root, "fixture-cycle")["status"] == "PASS"
    before = _snapshot(cycle)
    result = generate_research_report(root, "fixture-cycle")
    after = _snapshot(cycle)
    freeze_hash = hashlib.sha256(
        freeze.read_bytes()
    ).hexdigest()[:12]
    assert result.output_mode == "external"
    assert (
        result.output_dir
        == root / "reports" / "research" / "fixture-cycle" / freeze_hash
    )
    assert before == after
    assert verify_research_freeze(root, "fixture-cycle")["status"] == "PASS"
    assert result.model.oos.data["evidence_type"] == "fresh_state_reexecution"
    assert result.model.reproducibility.data["total_selection_relevant_trials"] == 1
    assert result.model.reproducibility.data["source_environment"][
        "dependency_snapshot_recorded_in_freeze"
    ]["pyfolio"] is None
    assert result.model.reproducibility.data["report_generator_environment"]["pyfolio"][
        "import_name"
    ] == "pyfolio"
    provenance = json.loads(
        (result.output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["pre_freeze"] is False
    assert provenance["source_freeze_manifest_sha256"] == freeze_manifest_sha256(freeze)
    assert provenance["freeze_version"] == "fixture-v1"
    assert provenance["candidate_fingerprint"] == "fixture-fingerprint"
    assert provenance["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert provenance["source_environment"]["dependency_snapshot_recorded_in_freeze"][
        "pyfolio"
    ] is None
    assert (result.output_dir / "figures" / "07_is_vs_oos_nav.png").exists()

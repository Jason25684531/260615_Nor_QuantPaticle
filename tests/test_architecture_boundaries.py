from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from run_research_report import main as compatibility_main
from twse_factor_lab.application.architecture import (
    ArchitectureValidationError,
    load_runner_inventory,
    run_architecture_checks,
    validate_cleanup_ledger,
    validate_dependency_direction,
    validate_frozen_write_policy,
    validate_runner_inventory,
)
from twse_factor_lab.application.commands.research_report import (
    main as application_main,
)
from twse_factor_lab.application.support import (
    load_config,
    project_root_for_config,
    repository_root,
    resolve_path,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "architecture" / "runner_inventory.json"


def test_runner_inventory_covers_every_root_runner_and_job():
    records = validate_runner_inventory(ROOT, INVENTORY)
    assert len(records) == 34
    assert {record.lifecycle for record in records} >= {
        "supported",
        "compatibility",
        "diagnostic",
        "operational-job",
    }


def test_runner_inventory_records_command_contract_fields():
    for record in load_runner_inventory(INVENTORY):
        assert record.owner
        assert record.invocation
        assert record.output_namespace
        assert record.retention_class
        assert record.write_authority
        assert record.retirement_condition
        assert record.cohort


def test_application_command_is_single_owner_for_research_report():
    assert compatibility_main is application_main
    assert any(
        record.path == "run_research_report.py"
        and record.owner == "twse_factor_lab.application.commands.research_report"
        for record in load_runner_inventory(INVENTORY)
    )


def test_existing_source_obeys_dependency_direction():
    validate_dependency_direction(ROOT)


def test_unclassified_runner_fails_inventory(tmp_path):
    inventory_path = tmp_path / "runner_inventory.json"
    inventory_path.write_text(INVENTORY.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "run_unclassified.py").write_text("", encoding="utf-8")
    with pytest.raises(ArchitectureValidationError, match="unclassified"):
        validate_runner_inventory(tmp_path, inventory_path)


def test_prohibited_domain_import_fails_boundary_check(tmp_path):
    source = tmp_path / "src" / "twse_factor_lab" / "data"
    source.mkdir(parents=True)
    (source / "bad.py").write_text(
        "from twse_factor_lab.application.commands import research_report\n",
        encoding="utf-8",
    )
    with pytest.raises(ArchitectureValidationError, match="prohibited dependency"):
        validate_dependency_direction(tmp_path)


def test_retained_evidence_cannot_be_marked_for_deletion(tmp_path):
    payload = json.loads(
        (ROOT / "docs" / "architecture" / "cleanup_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    next(
        item
        for item in payload["candidates"]
        if item["path"] == "final_acceptance_handoff.json"
    )["status"] = "delete"
    path = tmp_path / "cleanup_ledger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArchitectureValidationError, match="retained evidence"):
        validate_cleanup_ledger(path)


def test_full_architecture_checks_pass():
    run_architecture_checks(ROOT)


def test_application_support_preserves_config_path_contract(tmp_path):
    config_path = tmp_path / "config" / "strategy.yaml"
    config_path.parent.mkdir()
    config_path.write_text("paths:\n  sample: data/sample.parquet\n", encoding="utf-8")
    assert load_config(config_path)["paths"]["sample"] == "data/sample.parquet"
    assert project_root_for_config(config_path) == tmp_path
    assert (
        resolve_path(config_path, "data/sample.parquet")
        == tmp_path / "data/sample.parquet"
    )
    assert (
        resolve_path(config_path, tmp_path / "absolute.parquet")
        == tmp_path / "absolute.parquet"
    )
    assert repository_root(ROOT / "src" / "twse_factor_lab") == ROOT


def test_missing_migration_target_fails(tmp_path):
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    inventory["runners"] = [
        {
            "path": "run_only.py",
            "lifecycle": "supported",
            "owner": "run_only.py",
            "invocation": "python run_only.py",
            "output_namespace": "data/research/",
            "compatibility_status": "direct",
            "migration_target": "twse_factor_lab.application.commands.missing",
        }
    ]
    (tmp_path / "run_only.py").write_text("", encoding="utf-8")
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ArchitectureValidationError, match="migration target"):
        validate_runner_inventory(tmp_path, path)


def test_application_target_cannot_claim_frozen_namespace():
    report = next(
        record
        for record in load_runner_inventory(INVENTORY)
        if record.path == "run_research_report.py"
    )
    with pytest.raises(ArchitectureValidationError, match="frozen namespace"):
        validate_frozen_write_policy(
            (replace(report, output_namespace="reports/final/"),)
        )

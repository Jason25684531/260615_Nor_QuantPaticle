import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_rc1_manifests_have_required_contracts():
    reproducibility = json.loads((ROOT / "reproducibility_manifest.json").read_text())
    inventory = json.loads((ROOT / "final_artifact_inventory.json").read_text())
    required_manifest = {
        "git_commit", "config_sha256", "effective_trials", "artifact_hashes"
    }
    assert required_manifest <= reproducibility.keys()
    fields = {
        "name", "path", "stage", "type", "exists", "size", "schema_status",
        "date_range_status", "sha256", "canonical", "notes",
    }
    assert all(fields <= item.keys() for item in inventory["artifacts"])


def test_rc1_status_and_strategy_verdict_are_explicit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    handoff = json.loads((ROOT / "final_acceptance_handoff.json").read_text())
    assert "RC1: **PASS / CLOSED**" in readme and "Strategy: **REJECTED**" in readme
    assert handoff["strategy_acceptance"] == "REJECTED"
    assert handoff["d4_robustness_verdict"] == "REJECT"


def test_rc1_ci_and_matrix_are_present():
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    matrix = (ROOT / "reports/final/final_acceptance_matrix.md").read_text(
        encoding="utf-8"
    )
    assert "python -m pytest" in ci and "python -m ruff check ." in ci
    assert "openspec validate --all --strict" in ci
    columns = ("gate", "stage", "evidence", "status", "artifact", "limitation", "notes")
    assert all(column in matrix for column in columns)

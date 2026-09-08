"""Deterministic JSON persistence for research-cycle governance artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from twse_factor_lab.governance.isolation import (
    RESEARCH_NAMESPACE,
    assert_research_id_allowed,
    assert_write_allowed,
)
from twse_factor_lab.governance.schemas import (
    EXPERIMENT_STATUSES,
    DatasetManifest,
    ExperimentRecord,
    GovernanceError,
    ResearchManifest,
)

RESEARCH_MANIFEST_FILE = "research_manifest.json"
DATASET_MANIFESTS_FILE = "dataset_manifests.json"
EXPERIMENT_REGISTRY_FILE = "experiment_registry.json"


def research_dir(root: str | Path, research_id: str) -> Path:
    return Path(root) / RESEARCH_NAMESPACE / research_id


def _write_json(path: Path, root: str | Path, payload: Any) -> Path:
    target = assert_write_allowed(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def _read_json(path: Path, description: str) -> Any:
    if not path.exists():
        raise GovernanceError(f"{description} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_research_manifest(manifest: ResearchManifest, root: str | Path) -> Path:
    manifest.validate()
    assert_research_id_allowed(manifest.research_id)
    path = research_dir(root, manifest.research_id) / RESEARCH_MANIFEST_FILE
    if path.exists():
        raise GovernanceError(
            f"duplicate research_id: manifest already exists at {path}"
        )
    return _write_json(path, root, asdict(manifest))


def load_research_manifest(root: str | Path, research_id: str) -> ResearchManifest:
    path = research_dir(root, research_id) / RESEARCH_MANIFEST_FILE
    manifest = ResearchManifest(**_read_json(path, "research manifest"))
    manifest.validate()
    return manifest


def add_dataset_manifest(
    dataset: DatasetManifest, root: str | Path, research_id: str
) -> Path:
    dataset.validate()
    load_research_manifest(root, research_id)
    path = research_dir(root, research_id) / DATASET_MANIFESTS_FILE
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if any(entry["dataset_id"] == dataset.dataset_id for entry in entries):
        raise GovernanceError(f"duplicate dataset_id: {dataset.dataset_id!r}")
    entries.append(asdict(dataset))
    return _write_json(path, root, entries)


def load_dataset_manifests(
    root: str | Path, research_id: str
) -> list[DatasetManifest]:
    path = research_dir(root, research_id) / DATASET_MANIFESTS_FILE
    if not path.exists():
        return []
    return [
        DatasetManifest(**entry)
        for entry in json.loads(path.read_text(encoding="utf-8"))
    ]


def register_experiment(record: ExperimentRecord, root: str | Path) -> Path:
    record.validate()
    load_research_manifest(root, record.research_id)
    path = research_dir(root, record.research_id) / EXPERIMENT_REGISTRY_FILE
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if any(entry["experiment_id"] == record.experiment_id for entry in entries):
        raise GovernanceError(f"duplicate experiment_id: {record.experiment_id!r}")
    entries.append(asdict(record))
    return _write_json(path, root, entries)


def load_experiment_registry(
    root: str | Path, research_id: str
) -> list[ExperimentRecord]:
    path = research_dir(root, research_id) / EXPERIMENT_REGISTRY_FILE
    if not path.exists():
        return []
    return [
        ExperimentRecord(**entry)
        for entry in json.loads(path.read_text(encoding="utf-8"))
    ]


def update_experiment_status(
    root: str | Path, research_id: str, experiment_id: str, status: str
) -> Path:
    """Update only the status of one experiment; results are never overwritten."""
    if status not in EXPERIMENT_STATUSES:
        raise GovernanceError(f"unknown experiment status: {status!r}")
    path = research_dir(root, research_id) / EXPERIMENT_REGISTRY_FILE
    entries = _read_json(path, "experiment registry")
    for entry in entries:
        if entry["experiment_id"] == experiment_id:
            entry["status"] = status
            return _write_json(path, root, entries)
    raise GovernanceError(f"unknown experiment_id: {experiment_id!r}")

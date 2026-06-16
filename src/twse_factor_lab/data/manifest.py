"""Manifest helpers for Week 2 processed artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def build_manifest_entry(
    *,
    artifact_name: str,
    path: str,
    frame: pd.DataFrame,
    source_inputs: list[str],
    schema_version: str,
    created_at: datetime | None = None,
    notes: str = "",
) -> dict[str, Any]:
    timestamp = created_at or datetime.now(UTC)
    return {
        "artifact_name": artifact_name,
        "path": path,
        "source_inputs": source_inputs,
        "schema_version": schema_version,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "created_at": timestamp.isoformat(),
        "notes": notes,
    }


def write_manifest(entries: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"artifacts": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("artifacts", []))


def append_manifest_entries(entries: list[dict[str, Any]], path: str | Path) -> Path:
    merged: dict[str, dict[str, Any]] = {
        entry["artifact_name"]: entry for entry in read_manifest(path)
    }
    for entry in entries:
        merged[entry["artifact_name"]] = entry
    return write_manifest(list(merged.values()), path)

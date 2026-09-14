"""CSV tables derived from the normalized report model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .model import ResearchReportModel


def _value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def build_factor_evidence_table(model: ResearchReportModel) -> list[dict[str, Any]]:
    return list(model.factor_evidence.data.get("rows", []))


def build_strategy_trials_table(model: ResearchReportModel) -> list[dict[str, Any]]:
    return list(model.strategy_trials.data.get("rows", []))


def build_performance_crosscheck_table(
    model: ResearchReportModel,
) -> list[dict[str, Any]]:
    cross_check = model.performance.data.get("cross_check", {})
    checks = cross_check.get("checks", []) if isinstance(cross_check, dict) else []
    return list(checks) if isinstance(checks, list) else []


def build_robustness_table(model: ResearchReportModel) -> list[dict[str, Any]]:
    data = model.robustness.data
    scenarios = data.get("scenarios", []) if isinstance(data, dict) else []
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        row = {
            "scenario_id": scenario.get("scenario_id"),
            "status": scenario.get("status"),
            "robustness_result": data.get("robustness_result"),
        }
        row.update(scenario.get("config", {}))
        row.update(scenario.get("metrics", {}))
        rows.append(row)
    if not rows and data:
        rows.append({"robustness_result": data.get("robustness_result")})
    return rows


def build_statistical_acceptance_table(
    model: ResearchReportModel,
) -> list[dict[str, Any]]:
    data = model.statistics.data
    return [dict(data)] if data else []


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _value(row.get(key)) for key in fields})
    return path


def write_tables(model: ResearchReportModel, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    tables = {
        "factor_evidence": build_factor_evidence_table(model),
        "strategy_trials": build_strategy_trials_table(model),
        "performance_crosscheck": build_performance_crosscheck_table(model),
        "robustness": build_robustness_table(model),
        "statistical_acceptance": build_statistical_acceptance_table(model),
    }
    return {
        name: _write_csv(output_dir / f"{name}.csv", rows)
        for name, rows in tables.items()
    }


__all__ = [
    "build_factor_evidence_table",
    "build_performance_crosscheck_table",
    "build_robustness_table",
    "build_statistical_acceptance_table",
    "build_strategy_trials_table",
    "write_tables",
]

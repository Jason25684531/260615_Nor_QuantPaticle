"""Orchestrate read-only loading, plotting, tabulation, and rendering."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from twse_factor_lab.governance.isolation import (
    IsolationError,
    assert_research_cycle_writable,
)

from .figures import FigureResult, build_figures
from .loader import load_research_report_model
from .model import ResearchReportModel
from .provenance import REPORT_SCHEMA_VERSION, build_provenance, freeze_manifest_sha256
from .render_html import render_html
from .render_md import render_markdown
from .tables import write_tables


@dataclass(frozen=True)
class ReportResult:
    research_id: str
    output_mode: str
    output_dir: Path
    model: ResearchReportModel
    figures: tuple[FigureResult, ...]
    tables: dict[str, Path]
    files: dict[str, Path]


def _frozen(root: Path, research_id: str) -> bool:
    try:
        assert_research_cycle_writable(research_id, root)
    except IsolationError:
        return True
    return False


def _freeze_output_dir(root: Path, research_id: str, research_dir: Path) -> Path:
    manifest = research_dir / "freeze" / "research_freeze_manifest.json"
    digest = freeze_manifest_sha256(manifest)
    return root / "reports" / "research" / research_id / digest[:12]


def generate_research_report(
    root: str | Path,
    research_id: str,
    output_mode: str | None = None,
) -> ReportResult:
    """Generate a report while keeping frozen research-cycle evidence untouched."""

    root = Path(root).resolve()
    model = load_research_report_model(root, research_id)
    research_dir = root / "data" / "research" / research_id
    is_frozen = _frozen(root, research_id)
    if is_frozen:
        mode = "external"
        output_dir = _freeze_output_dir(root, research_id, research_dir)
    elif output_mode in (None, "auto", "research-cycle"):
        mode = "research-cycle"
        output_dir = research_dir / "report"
    else:
        mode = "external"
        output_dir = root / "reports" / "research" / research_id / "pre-freeze"

    source_files = dict(model.performance.data.get("source_files", {}))
    source_files["repository_root"] = str(root)
    performance_data = dict(model.performance.data)
    performance_data["source_files"] = source_files
    model = replace(
        model, performance=replace(model.performance, data=performance_data)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_results = tuple(build_figures(model, output_dir / "figures"))
    table_paths = write_tables(model, output_dir / "tables")
    markdown = render_markdown(model, figure_results, table_paths)
    html = render_html(model, figure_results, table_paths)
    files = {
        "research_report.md": output_dir / "research_report.md",
        "research_report.html": output_dir / "research_report.html",
        "provenance.json": output_dir / "provenance.json",
        "summary.json": output_dir / "summary.json",
    }
    files["research_report.md"].write_text(markdown, encoding="utf-8")
    files["research_report.html"].write_text(html, encoding="utf-8")
    provenance = build_provenance(model, root)
    files["provenance.json"].write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "research_id": research_id,
        "output_mode": mode,
        "output_dir": str(output_dir),
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "platform_verdict": model.acceptance.data.get("research_platform_verdict"),
        "strategy_verdict": model.acceptance.data.get("strategy_verdict"),
        "factor_selection_trial_count": model.statistics.data.get(
            "factor_selection_trial_count"
        ),
        "strategy_selection_trial_count": model.statistics.data.get(
            "strategy_selection_trial_count"
        ),
        "total_selection_relevant_trials": model.statistics.data.get(
            "total_selection_relevant_trials"
        ),
        "dsr_effective_strategy_trials": model.statistics.data.get(
            "dsr_effective_strategy_trials"
        ),
        "diagnostic_trial_count": model.statistics.data.get("diagnostic_trial_count"),
        "trial_count_consistency": model.statistics.data.get(
            "trial_count_consistency"
        ),
        "model": model.to_dict(),
        "figures": [result.to_dict(output_dir) for result in figure_results],
        "tables": {
            name: path.relative_to(output_dir).as_posix()
            for name, path in table_paths.items()
        },
    }
    files["summary.json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ReportResult(
        research_id, mode, output_dir, model, figure_results, table_paths, files
    )


__all__ = ["ReportResult", "generate_research_report"]

"""Small stdlib-only HTML renderer for the normalized report model."""

# ruff: noqa: E501

from __future__ import annotations

import html
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .model import ResearchReportModel
from .provenance import REPORT_SCHEMA_VERSION


def _text(value: Any) -> str:
    if value is None:
        return "UNAVAILABLE"
    return html.escape(str(value))


def _table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{_text(row.get(column))}</td>" for column in columns)
            + "</tr>"
        )
    if not body:
        body.append(f"<tr><td>{'UNAVAILABLE'}</td></tr>")
    head = "".join(f"<th>{_text(column)}</th>" for column in columns)
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _section(title: str, body: str) -> str:
    return f"<section><h2>{html.escape(title)}</h2>{body}</section>"


def render_html(
    model: ResearchReportModel,
    figures: Iterable[Any] | None = None,
    tables: Any = None,
) -> str:
    """Render HTML without reading source artifacts or using a template dependency."""

    research = model.research.data
    dataset = model.dataset.data
    acceptance = model.acceptance.data
    candidate = model.locked_candidate.data.get("config", {})
    metrics = model.performance.data.get("canonical_metrics", {})
    cross_check = model.performance.data.get("cross_check", {})
    factor_rows = model.factor_evidence.data.get("rows", [])
    trial_rows = model.strategy_trials.data.get("rows", [])
    stats = model.statistics.data
    oos = model.oos.data
    exec_data = model.execution_validation.data
    image_html = []
    if figures is not None:
        for result in figures:
            if getattr(result, "status", None) == "GENERATED" and getattr(
                result, "path", None
            ):
                name = html.escape(Path(result.path).name)
                image_html.append(
                    f'<figure><img src="figures/{name}" '
                    f'alt="{html.escape(result.figure_id)}">'
                    f"<figcaption>{html.escape(result.figure_id)}"
                    "</figcaption></figure>"
                )
            else:
                image_html.append(
                    f"<p>{html.escape(getattr(result, 'figure_id', 'figure'))}: "
                    "UNAVAILABLE</p>"
                )
    parts = [
        '<!doctype html><html><head><meta charset="utf-8"><title>Research Report</title>',
        "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:auto}section{margin:2em 0;padding:1em;border:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:.9em}th,td{border:1px solid #ccc;padding:.35em;text-align:left;vertical-align:top}img{max-width:100%}.platform{border-left:5px solid #2878c8}.strategy{border-left:5px solid #c84b28}</style></head><body>",
        "<h1>Research Report</h1>",
    ]
    summary = (
        f"<p>Research ID: {_text(research.get('research_id'))}<br>"
        f"Dataset ID: {_text(dataset.get('dataset_version'))}<br>"
        f"IS Period: {_text(research.get('is_start'))} to {_text(research.get('is_end'))}<br>"
        f"OOS Period: {_text(research.get('oos_start'))} to {_text(research.get('oos_end'))}<br>"
        f"Locked Candidate: {_text(candidate.get('strategy_id'))}</p>"
        f'<div class="platform"><h3>PLATFORM VALIDITY</h3><p>Research Platform Verdict: {_text(acceptance.get("research_platform_verdict"))}</p></div>'
        f'<div class="strategy"><h3>STRATEGY QUALITY</h3><p>Strategy Verdict: {_text(acceptance.get("strategy_verdict"))}</p></div>'
    )
    parts.append(_section("Executive Summary", summary))
    parts.append(
        _section(
            "Research Definition",
            f"<p>Hypothesis: {_text(research.get('hypothesis'))}<br>Universe: {_text(research.get('universe'))}</p>",
        )
    )
    parts.append(
        _section(
            "Dataset / PIT Information",
            f"<p>Dataset version: {_text(dataset.get('dataset_version'))}<br>PIT manifests: {_text(dataset.get('manifests'))}</p>",
        )
    )
    parts.append(
        _section(
            "Factor Evidence",
            _table(
                factor_rows,
                [
                    "factor",
                    "family",
                    "direction",
                    "primary_horizon",
                    "mean_ic",
                    "icir",
                    "positive_ic_ratio",
                    "quantile_spread",
                    "coverage",
                    "turnover",
                    "rank_autocorrelation",
                    "verdict",
                ],
            ),
        )
    )
    parts.append(
        _section(
            "Strategy Trials + Locked Candidate",
            f"<p>Candidate config: {_text(candidate)}</p>"
            + _table(
                trial_rows,
                [
                    "trial_id",
                    "factors",
                    "weights",
                    "top_n",
                    "rebalance",
                    "buffer",
                    "cost",
                    "is_return",
                    "is_sharpe",
                    "mdd",
                    "selection_status",
                ],
            ),
        )
    )
    parts.append(
        _section(
            "Tri-engine Validation",
            f"<p>Status: {_text(exec_data.get('parity_status', exec_data.get('status')))}<br>Engines: {_text(exec_data.get('engines'))}<br>Comparisons: {_text(exec_data.get('comparisons'))}</p>",
        )
    )
    metric_text = "<br>".join(f"{_text(k)}: {_text(v)}" for k, v in metrics.items())
    parts.append(_section("Performance Summary", f"<p>{metric_text}</p>"))
    parts.append(
        _section(
            "Pyfolio Tear Sheet",
            f"<p>Status: {_text(model.pyfolio.status)}<br>"
            f"Transactions: {_text(model.performance.data.get('transactions'))}</p>"
            + "".join(image_html),
        )
    )
    checks = cross_check.get("checks", []) if isinstance(cross_check, dict) else []
    parts.append(
        _section(
            "Canonical vs Pyfolio Cross-check",
            _table(checks, ["metric", "canonical", "pyfolio", "difference", "status"])
            + (
                "<p>Canonical metrics are authoritative.<br>"
                "Pyfolio is an independent diagnostic.<br>"
                "DEFINITION_DIFFERENCE does not indicate execution parity failure.</p>"
                if any(
                    check.get("status") == "DEFINITION_DIFFERENCE" for check in checks
                )
                else ""
            ),
        )
    )
    attribution = (
        f"<p>Pipeline Status: {_text(model.attribution.data.get('execution_status'))}<br>"
        f"Evidence Status: {_text(model.attribution.data.get('evidence_status'))}<br>"
        f"Method: {_text(model.attribution.data.get('method'))}<br>"
        f"Available descriptors: {_text(model.attribution.data.get('available_descriptors'))}<br>"
        f"Unavailable descriptors: {_text(model.attribution.data.get('unavailable_descriptors'))}<br>"
        f"PIT industry status: {_text(model.attribution.data.get('pit_industry_status'))}</p>"
    )
    if model.attribution.data.get("evidence_status") == "UNAVAILABLE":
        attribution += "<p>Pipeline execution PASS, attribution evidence unavailable due to missing descriptors.</p>"
    parts.append(_section("Barra-style Attribution", attribution))
    parts.append(
        _section(
            "Robustness",
            f"<p>Status: {_text(model.robustness.status)}<br>Verdict: {_text(model.robustness.data.get('robustness_result'))}</p>",
        )
    )
    parts.append(
        _section(
            "Fresh-state OOS",
            f"<p>Status: {_text(model.oos.status)}<br>OOS = FRESH-STATE RE-EXECUTION: {_text(oos.get('evidence_type'))}<br>OOS metrics: {_text(oos.get('oos_metrics'))}</p>",
        )
    )
    parts.append(
        _section(
            "Statistical Acceptance",
            f"<p>Status: {_text(model.statistics.status)}<br>Factor Selection Trials: {_text(stats.get('factor_selection_trial_count'))}<br>Strategy Selection Trials: {_text(stats.get('strategy_selection_trial_count'))}<br>Total Selection-Relevant Trials: {_text(stats.get('total_selection_relevant_trials'))}<br>DSR Effective Strategy Trials: {_text(stats.get('dsr_effective_strategy_trials'))}<br>Diagnostic Trials: {_text(stats.get('diagnostic_trial_count'))}<br>Trial Count Consistency: {_text(stats.get('trial_count_consistency'))}<br>PSR: {_text(stats.get('psr'))}<br>DSR: {_text(stats.get('dsr'))}</p>",
        )
    )
    parts.append(
        _section(
            "Final Acceptance",
            f"<p>Research Platform: {_text(acceptance.get('research_platform_verdict'))}<br>Strategy Research: {_text(acceptance.get('strategy_verdict'))}</p>",
        )
    )
    parts.append(
        _section(
            "Reproducibility / Freeze",
            f"<p>Status: {_text(model.reproducibility.status)}<br>Source Freeze Manifest: {_text(model.reproducibility.data.get('source_freeze_manifest'))}<br>Source Freeze Manifest SHA256: {_text(model.reproducibility.data.get('source_freeze_manifest_sha256'))}<br>Source Code Revision: {_text(model.reproducibility.data.get('source_code_revision'))}<br>Source Git Dirty: {_text(model.reproducibility.data.get('source_git_dirty'))}<br>Freeze Version: {_text(model.reproducibility.data.get('freeze_version'))}<br>Candidate Fingerprint: {_text(model.reproducibility.data.get('candidate_fingerprint'))}<br>Total Selection-Relevant Trials: {_text(model.reproducibility.data.get('total_selection_relevant_trials'))}<br>Report Schema Version: {REPORT_SCHEMA_VERSION}<br>Source Environment: {_text(model.reproducibility.data.get('source_environment'))}<br>Report Generator Environment: {_text(model.reproducibility.data.get('report_generator_environment'))}</p>",
        )
    )
    for title, section in (
        ("Universe / Data Coverage", model.universe_coverage),
        ("Factor Ablation", model.factor_ablation),
        ("Cost & Turnover Analysis", model.cost_turnover),
        ("Performance Metric Definition", model.performance_metric_definition),
        ("Market Breadth Risk Overlay", model.breadth_overlay),
        ("Validation (Walk-forward)", model.validation),
    ):
        data = section.data
        body = f"<p>Status: {_text(section.status)}</p>"
        scalars = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
        if scalars:
            body += "<p>" + "<br>".join(
                f"{_text(k)}: {_text(v)}" for k, v in scalars.items()
            ) + "</p>"
        rows = data.get("rows") or data.get("folds")
        if isinstance(rows, list) and rows:
            columns = list(rows[0].keys())
            body += _table(rows, columns)
        parts.append(_section(title, body))
    limitations = model.limitations.data.get("items", [])
    parts.append(
        _section(
            "Limitations",
            "<ul>"
            + "".join(f"<li>{_text(item)}</li>" for item in limitations)
            + "</ul>",
        )
    )
    parts.append("</body></html>\n")
    return "".join(parts)


__all__ = ["render_html"]

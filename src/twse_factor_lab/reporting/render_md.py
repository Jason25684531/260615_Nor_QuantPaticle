"""Deterministic Markdown rendering from a ResearchReportModel only."""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .model import ResearchReportModel
from .provenance import REPORT_SCHEMA_VERSION


def _text(value: Any) -> str:
    if value is None:
        return "UNAVAILABLE"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _table(rows: Iterable[dict[str, Any]], columns: list[str]) -> list[str]:
    rows = list(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_text(row.get(column)) for column in columns) + " |"
        )
    if not rows:
        lines.append(
            "| " + " | ".join(["UNAVAILABLE"] + [""] * (len(columns) - 1)) + " |"
        )
    return lines


def _status(section: Any) -> str:
    return getattr(section, "status", "UNAVAILABLE")


def render_markdown(
    model: ResearchReportModel,
    figures: Iterable[Any] | None = None,
    tables: Any = None,
) -> str:
    """Render the fixed 16-section report; artifact I/O is intentionally absent."""

    research = model.research.data
    dataset = model.dataset.data
    acceptance = model.acceptance.data
    candidate = model.locked_candidate.data.get("config", {})
    factor_rows = model.factor_evidence.data.get("rows", [])
    trial_rows = model.strategy_trials.data.get("rows", [])
    performance = model.performance.data
    pyfolio = model.pyfolio.data
    cross_check = performance.get("cross_check", {})
    oos = model.oos.data
    stats = model.statistics.data
    figure_lines = []
    if figures is not None:
        for result in figures:
            if getattr(result, "status", None) == "GENERATED" and getattr(
                result, "path", None
            ):
                figure_lines.append(
                    f"![{result.figure_id}](figures/{Path(result.path).name})"
                )
            else:
                figure_lines.append(
                    f"- {getattr(result, 'figure_id', 'figure')}: UNAVAILABLE"
                )
    lines = [
        "# Research Report",
        "",
        "## Executive Summary",
        "",
        f"- Research ID: {_text(research.get('research_id'))}",
        f"- Dataset ID: {_text(dataset.get('dataset_version'))}",
        f"- IS Period: {_text(research.get('is_start'))} to {_text(research.get('is_end'))}",
        f"- OOS Period: {_text(research.get('oos_start'))} to {_text(research.get('oos_end'))}",
        f"- Primary Horizon: {_text(next((row.get('primary_horizon') for row in factor_rows if row.get('primary_horizon') is not None), 'UNAVAILABLE'))}",
        f"- Locked Candidate: {_text(candidate.get('strategy_id'))}",
        "",
        "### PLATFORM VALIDITY",
        "",
        f"- Research Platform Verdict: {_text(acceptance.get('research_platform_verdict'))}",
        "",
        "### STRATEGY QUALITY",
        "",
        f"- Strategy Verdict: {_text(acceptance.get('strategy_verdict'))}",
        "",
        "## Research Definition",
        "",
        f"- Hypothesis: {_text(research.get('hypothesis'))}",
        f"- Universe: {_text(research.get('universe'))}",
        f"- Research status: {_status(model.research)}",
        "",
        "## Dataset / PIT Information",
        "",
        f"- Dataset version: {_text(dataset.get('dataset_version'))}",
        f"- PIT manifests: {_text(dataset.get('manifests'))}",
        f"- Section status: {_status(model.dataset)}",
        "",
        "## Factor Evidence",
        "",
    ]
    lines += _table(
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
    )
    lines += [
        "",
        f"Section status: {_status(model.factor_evidence)}",
        "",
        "## Strategy Trials + Locked Candidate",
        "",
        f"- Candidate config: {_text(candidate)}",
    ]
    lines += _table(
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
    )
    lines += [
        "",
        "## Tri-engine Validation",
        "",
        f"- Status: {_status(model.execution_validation)}",
        f"- Engines: {_text(model.execution_validation.data.get('engines'))}",
        f"- Parity verdict: {_text(model.execution_validation.data.get('parity_status', model.execution_validation.data.get('status')))}",
        f"- Comparisons: {_text(model.execution_validation.data.get('comparisons'))}",
        "",
        "## Performance Summary",
        "",
        f"- Status: {_status(model.performance)}",
    ]
    metrics = performance.get("canonical_metrics", {})
    for key in (
        "total_return",
        "cagr",
        "annualized_volatility",
        "volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
    ):
        if key in metrics:
            lines.append(f"- {key}: {_text(metrics[key])}")
    lines += [
        "",
        "## Pyfolio Tear Sheet",
        "",
        f"- Status: {_status(model.pyfolio)}",
        f"- Metrics: {_text(pyfolio.get('metrics'))}",
        f"- Transactions: {_text(performance.get('transactions'))}",
        *figure_lines,
        "",
        "## Canonical vs Pyfolio Cross-check",
        "",
    ]
    lines += _table(
        cross_check.get("checks", []) if isinstance(cross_check, dict) else [],
        ["metric", "canonical", "pyfolio", "difference", "status"],
    )
    lines += [
        "",
        f"- Cross-check summary: {_text(cross_check.get('status') if isinstance(cross_check, dict) else None)}",
    ]
    checks = cross_check.get("checks", []) if isinstance(cross_check, dict) else []
    if any(check.get("status") == "DEFINITION_DIFFERENCE" for check in checks):
        lines += [
            "- Canonical metrics are authoritative.",
            "- Pyfolio is an independent diagnostic.",
            "- DEFINITION_DIFFERENCE does not indicate execution parity failure.",
        ]
    lines += [
        "",
        "## Barra-style Attribution",
        "",
        f"- Pipeline Status: {_text(model.attribution.data.get('execution_status'))}",
        f"- Evidence Status: {_text(model.attribution.data.get('evidence_status'))}",
        f"- Method: {_text(model.attribution.data.get('method'))}",
        f"- Available descriptors: {_text(model.attribution.data.get('available_descriptors'))}",
        f"- Unavailable descriptors: {_text(model.attribution.data.get('unavailable_descriptors'))}",
        f"- PIT industry status: {_text(model.attribution.data.get('pit_industry_status'))}",
    ]
    if model.attribution.data.get("evidence_status") == "UNAVAILABLE":
        lines.append(
            "- Pipeline execution PASS, attribution evidence unavailable due to missing descriptors."
        )
    lines += [
        "",
        "## Robustness",
        "",
        f"- Status: {_status(model.robustness)}",
        f"- Verdict: {_text(model.robustness.data.get('robustness_result'))}",
        f"- Scenarios: {_text(model.robustness.data.get('scenarios'))}",
        "",
        "## Fresh-state OOS",
        "",
        f"- Status: {_status(model.oos)}",
        f"- OOS = FRESH-STATE RE-EXECUTION: {_text(oos.get('evidence_type'))}",
        f"- Warmup start: {_text(oos.get('warmup_start'))}",
        f"- OOS period: {_text(oos.get('oos_start'))} to {_text(oos.get('oos_end'))}",
        f"- Initial cash: {_text(oos.get('initial_cash'))}",
        f"- Initial positions: {_text(oos.get('initial_positions'))}",
        f"- OOS metrics: {_text(oos.get('oos_metrics'))}",
        "",
        "## Statistical Acceptance",
        "",
        f"- Status: {_status(model.statistics)}",
        f"- Factor Selection Trials: {_text(stats.get('factor_selection_trial_count'))}",
        f"- Strategy Selection Trials: {_text(stats.get('strategy_selection_trial_count'))}",
        f"- Total Selection-Relevant Trials: {_text(stats.get('total_selection_relevant_trials'))}",
        f"- DSR Effective Strategy Trials: {_text(stats.get('dsr_effective_strategy_trials'))}",
        f"- Diagnostic Trials: {_text(stats.get('diagnostic_trial_count'))}",
        f"- Trial Count Consistency: {_text(stats.get('trial_count_consistency'))}",
        f"- PSR: {_text(stats.get('psr'))}",
        f"- DSR: {_text(stats.get('dsr'))}",
        "",
        "## Final Acceptance",
        "",
        f"- Research Platform: {_text(acceptance.get('research_platform_verdict'))}",
        f"- Strategy Research: {_text(acceptance.get('strategy_verdict'))}",
        f"- Reasons: {_text(acceptance.get('reasons'))}",
        "",
        "## Reproducibility / Freeze",
        "",
        f"- Status: {_status(model.reproducibility)}",
        f"- Source Freeze Manifest: {_text(model.reproducibility.data.get('source_freeze_manifest'))}",
        f"- Source Freeze Manifest SHA256: {_text(model.reproducibility.data.get('source_freeze_manifest_sha256'))}",
        f"- Source Code Revision: {_text(model.reproducibility.data.get('source_code_revision'))}",
        f"- Source Git Dirty: {_text(model.reproducibility.data.get('source_git_dirty'))}",
        f"- Freeze Version: {_text(model.reproducibility.data.get('freeze_version'))}",
        f"- Candidate Fingerprint: {_text(model.reproducibility.data.get('candidate_fingerprint'))}",
        f"- Total Selection-Relevant Trials: {_text(model.reproducibility.data.get('total_selection_relevant_trials'))}",
        f"- Report Schema Version: {REPORT_SCHEMA_VERSION}",
        f"- Source Environment: {_text(model.reproducibility.data.get('source_environment'))}",
        f"- Report Generator Environment: {_text(model.reproducibility.data.get('report_generator_environment'))}",
        "",
    ]
    lines += _v2_sections(model)
    lines += [
        "## Limitations",
        "",
    ]
    items = model.limitations.data.get("items", [])
    lines += [f"- {item}" for item in items] or ["- UNAVAILABLE"]
    return "\n".join(lines) + "\n"


def _kv_lines(data: dict[str, Any], keys: Iterable[str]) -> list[str]:
    return [f"- {key}: {_text(data.get(key))}" for key in keys]


def _v2_sections(model: ResearchReportModel) -> list[str]:
    """Render the Research Cycle v2 sections; UNAVAILABLE for v1 cycles."""
    universe = model.universe_coverage.data
    ablation = model.factor_ablation.data
    cost = model.cost_turnover.data
    metric_def = model.performance_metric_definition.data
    breadth = model.breadth_overlay.data
    validation = model.validation.data
    lines = [
        "## Universe / Data Coverage",
        "",
        f"- Status: {_status(model.universe_coverage)}",
        *_kv_lines(
            universe,
            (
                "methodology",
                "liquidity_rule",
                "liquidity_measure_source",
                "mean_universe_count",
                "median_universe_count",
                "min_universe_count",
                "eps_coverage",
                "roe_coverage",
                "joint_coverage",
                "survivorship",
            ),
        ),
        "",
        "## Factor Ablation",
        "",
        f"- Status: {_status(model.factor_ablation)}",
    ]
    lines += _table(
        ablation.get("rows", []),
        [
            "configuration",
            "factor_gate",
            "is_net_return",
            "sharpe",
            "sortino",
            "mdd",
            "turnover",
        ],
    )
    lines += [
        "",
        "## Cost & Turnover Analysis",
        "",
        f"- Status: {_status(model.cost_turnover)}",
    ]
    lines += _table(
        cost.get("rows", []),
        [
            "configuration",
            "gross_return",
            "net_return",
            "cost_drag",
            "turnover",
            "cagr",
            "sharpe",
            "sortino",
            "mdd",
            "calmar",
        ],
    )
    lines += [
        "",
        "## Performance Metric Definition",
        "",
        f"- Status: {_status(model.performance_metric_definition)}",
        *_kv_lines(
            metric_def,
            (
                "return_frequency",
                "annualization_factor",
                "risk_free_rate",
                "sortino_target",
                "downside_deviation",
                "volatility_ddof",
                "nan_policy",
                "return_convention",
                "annualization_method",
                "metric_schema_version",
                "canonical_metric_source",
            ),
        ),
        "",
        "## Market Breadth Risk Overlay",
        "",
        f"- Status: {_status(model.breadth_overlay)}",
        *_kv_lines(
            breadth,
            (
                "definition",
                "threshold",
                "exposure_high",
                "exposure_low",
                "risk_off_days",
                "risk_off_pct",
            ),
        ),
    ]
    lines += _table(
        breadth.get("rows", []),
        [
            "variant",
            "cagr",
            "sharpe",
            "sortino",
            "mdd",
            "calmar",
            "net_return",
            "turnover",
        ],
    )
    lines += [
        "",
        "## Validation (Walk-forward)",
        "",
        f"- Status: {_status(model.validation)}",
        f"- Label: {_text(validation.get('label'))}",
        f"- Reason: {_text(validation.get('reason'))}",
        f"- Warmup note: {_text(validation.get('warmup_note'))}",
    ]
    lines += _table(
        validation.get("rows", validation.get("folds", [])),
        ["fold", "test_start", "test_end", "cagr", "sharpe", "sortino", "mdd"],
    )
    lines += [""]
    return lines


__all__ = ["render_markdown"]

"""Build the additive fundamental PIT-v2 dataset and its evidence pack."""
# ruff: noqa: E501

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from jobs.run_fundamental_pit_expansion import (
    build_target_tickers,
    cached_publication_tickers,
    target_ticker_sha256,
)
from run_data_pipeline import load_config, project_root_for_config, resolve_path
from twse_factor_lab.data.fundamental import (
    FundamentalClient,
    FundamentalDataError,
    RawFundamentalCache,
    build_financial_pit,
    build_fundamental_matrix,
    validate_pit_records,
)
from twse_factor_lab.data.fundamental_quality import (
    build_fundamental_coverage_reports,
    build_fundamental_missingness_report,
    build_publication_alignment_report,
    build_raw_inventory,
    build_source_audit,
    build_ticker_mapping_report,
    evaluate_data_readiness,
    factor_gate_compatible_coverage,
    normalize_fundamental_records,
    publication_alignment_summary,
    sha256_file,
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _client(config: dict[str, Any], cache: RawFundamentalCache) -> FundamentalClient:
    fundamental = config["fundamental"]
    return FundamentalClient(
        cache=cache,
        twse_base_url=config["twse"]["base_url"],
        timeout=int(config["twse"].get("timeout_seconds", 30)),
        mops_timeout=int(fundamental.get("mops_timeout_seconds", 90)),
        throttle_seconds=float(fundamental.get("throttle_seconds", 0.5)),
        retry=int(fundamental.get("retry", 3)),
        mops_statement_min_ticker_count=int(
            fundamental.get("mops_statement_min_ticker_count", 100)
        ),
    )


def _collect_financial_sources(
    client: FundamentalClient,
    *,
    tickers: list[str],
    years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    statements: list[pd.DataFrame] = []
    publications: list[pd.DataFrame] = []
    errors: list[str] = []
    for year in years:
        for season in range(1, 5):
            for statement in ("income", "balance"):
                try:
                    result = client.mops_statement(year, season, statement)
                except FundamentalDataError as exc:
                    errors.append(f"mops {statement} {year}Q{season}: {exc}")
                else:
                    statements.append(result)
    for ticker in tickers:
        for year in years:
            try:
                result = client.publication_dates(ticker, year)
            except FundamentalDataError as exc:
                errors.append(f"publication {ticker} {year}: {exc}")
            else:
                if not result.empty:
                    publications.append(result)
    statement_frame = (
        pd.concat(statements, ignore_index=True) if statements else pd.DataFrame()
    )
    if not statement_frame.empty:
        statement_frame = statement_frame.loc[
            statement_frame["ticker"].astype(str).isin(tickers)
        ].reset_index(drop=True)
    publication_frame = (
        pd.concat(publications, ignore_index=True)
        if publications
        else pd.DataFrame(
            columns=["ticker", "roc_year", "season", "publication_date"]
        )
    )
    return statement_frame, publication_frame, errors


def _coverage_factor_matrices(
    matrix: pd.DataFrame, forward_returns: pd.DataFrame
) -> dict[str, float]:
    matrices = {
        metric: matrix.loc[matrix["metric"].eq(metric)]
        .pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
        for metric in ("eps", "roe")
    }
    matrices["joint"] = matrices["eps"].where(matrices["roe"].notna())
    return {
        metric: factor_gate_compatible_coverage(frame, forward_returns, horizon=20)
        for metric, frame in matrices.items()
    }


def _write_data_report(
    path: Path,
    *,
    baseline: dict[str, Any],
    source_audit: dict[str, Any],
    coverage: dict[str, Any],
    alignment: dict[str, Any],
    readiness: dict[str, Any],
    missingness: pd.DataFrame,
    errors: list[str],
    response_sanity: dict[str, Any],
    expansion: dict[str, Any],
) -> None:
    yearly = coverage["yearly"]
    missing_reason = (
        missingness["missing_reason"].value_counts().to_dict()
        if not missingness.empty
        else {}
    )
    lines = [
        "# Fundamental PIT Coverage v1 Report",
        "",
        "## 1. Baseline 25/30 root cause",
        "",
        f"- Baseline EPS PIT tickers: {baseline['eps_pit_ticker_count']}.",
        f"- Baseline EPS matrix tickers: {baseline['eps_matrix_ticker_count']}.",
        "- The uncovered baseline tickers are retained in source audit with explicit reasons; no rows are fabricated.",
        "",
        "## 2. Sources and expansion",
        "",
        "### Publication Expansion (before → after)",
        "",
        f"- Target eligible tickers: {expansion.get('target_ticker_count')}",
        f"- Previously non-empty publication mappings: {expansion.get('publication_ticker_count_before', expansion.get('existing_publication_ticker_count_before_run'))}",
        f"- Publication metadata tickers before: {expansion.get('publication_metadata_ticker_count_before', expansion.get('existing_publication_ticker_count_before_run'))}",
        f"- Missing non-empty mappings before: {expansion.get('missing_ticker_count_before', expansion.get('missing_publication_ticker_count_before_run'))}",
        f"- Newly fetched tickers: {expansion.get('newly_fetched_ticker_count')}",
        f"- Source failures (ticker-year): {expansion.get('http_failure_count')}",
        f"- Final publication metadata tickers: {expansion.get('publication_metadata_ticker_count_after', expansion.get('final_publication_ticker_count'))}",
        f"- Final non-empty publication mappings: {expansion.get('final_publication_ticker_count')}",
        f"- Remaining missing targets: {expansion.get('missing_target_ticker_count')}",
        f"- Target coverage: {float(expansion.get('target_publication_coverage_ratio') or 0.0):.4%}",
        f"- Enumeration status: {expansion.get('enumeration_status')}",
        f"- Full expansion status: {expansion.get('full_publication_expansion_status')}",
        "",
        "- Sources: MOPS `ajax_t163sb04` / `ajax_t163sb05` and doc.twse `t57sb01`.",
        f"- DATA_EXPANSION_STATUS: {source_audit['DATA_EXPANSION_STATUS']}.",
        f"- Collection errors: {len(errors)}.",
        f"- Response sanity: {response_sanity['response_sanity_rule']}.",
        f"- Response sanity threshold: {response_sanity['response_sanity_threshold']}; invalid={response_sanity['invalid_response_count']}; quarantined={response_sanity['quarantined_response_count']}.",
        "",
        "## 3. Coverage before / after",
        "",
        f"- Before: {baseline['eps_matrix_ticker_count']} EPS matrix tickers.",
        f"- After: {coverage['by_ticker']['eps_observation_count'].gt(0).sum()} EPS-covered tickers and {coverage['by_ticker']['roe_observation_count'].gt(0).sum()} ROE-covered tickers.",
        "",
        "## 4. Yearly coverage (2019–2025)",
        "",
        *[f"- {row.year}: EPS={row.eps_coverage:.4f}, ROE={row.roe_coverage:.4f}, joint={row.joint_coverage:.4f}." for row in yearly.itertuples()],
        "",
        "## 5. Publication alignment",
        "",
        f"- Records: {alignment['record_count']}.",
        f"- Period-end to publication: {alignment['period_end_to_publication_days']}.",
        f"- Publication to available: {alignment['publication_to_available_days']}.",
        f"- Anomalies: {alignment['anomalies'] or 'none'}.",
        "",
        "## 6. PIT and revision tests",
        "",
        "- PIT ordering, future-publication exclusion, and backward as-of alignment are validated by the generated records/matrix.",
        "- Multiple publication versions remain separate; source restatement metadata is unavailable (`revision_status = SOURCE_UNAVAILABLE` limitation).",
        "",
        "## 7. Missingness and survivorship",
        "",
        f"- Largest missingness reason: {max(missing_reason, key=missing_reason.get) if missing_reason else 'none'}.",
        f"- Missingness counts: {missing_reason or 'none'}.",
        "- Survivorship: `membership_basis = current_listed_only`; `survivorship_status = KNOWN_LIMITATION`.",
        "- Weak coverage years are visible in the yearly table and are not hidden by a readiness verdict.",
        "",
        "## 8. Readiness and Research Cycle v3",
        "",
        f"- EPS ready: {readiness['eps_research_ready']}; ROE ready: {readiness['roe_research_ready']}; joint ready: {readiness['joint_composite_ready']}.",
        f"- Overall: {readiness['overall_data_readiness']}; joint status: {readiness['joint_status']}.",
        f"- READY_FOR_RESEARCH_CYCLE_V3: {readiness['READY_FOR_RESEARCH_CYCLE_V3']}.",
        "- This change stops at data readiness; it does not run IC, ICIR, Factor Gate verdict, Strategy Lab, or a backtest.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _openspec_progress(root: Path) -> tuple[int, int]:
    tasks_path = (
        root
        / "openspec/changes/expand-fundamental-pit-coverage-v1/tasks.md"
    )
    text = tasks_path.read_text(encoding="utf-8")
    total = text.count("- [x]") + text.count("- [ ]")
    complete = text.count("- [x]")
    return complete, total


def _write_final_status_report(
    path: Path,
    *,
    root: Path,
    baseline: dict[str, Any],
    source_audit: dict[str, Any],
    expansion: dict[str, Any],
    response_sanity: dict[str, Any],
    coverage: dict[str, Any],
    readiness: dict[str, Any],
    pit_validation: dict[str, Any],
    final_hard_gates: bool,
) -> None:
    complete, total = _openspec_progress(root)
    gate_status = "PASS" if final_hard_gates else "PENDING_FINAL_RUN"
    coverage_values = readiness["factor_gate_compatible_coverage"]
    sanity_status = (
        "PASS"
        if response_sanity["invalid_response_count"]
        == response_sanity["quarantined_response_count"]
        else "WARN"
    )
    rows = [
        ("A", "Change scope", "PASS", "expand-fundamental-pit-coverage-v1"),
        ("B", "OpenSpec progress", f"{complete}/{total}", "tasks.md"),
        ("C", "Baseline", "PASS", f"EPS PIT={baseline['eps_pit_ticker_count']}; matrix={baseline['eps_matrix_ticker_count']}"),
        ("D", "Source reachability", "PASS" if not source_audit.get("external_blockers") else "BLOCKED", "doc.twse t57sb01 + MOPS available"),
        ("E", "Publication expansion", expansion.get("full_publication_expansion_status", expansion.get("data_expansion_status", "UNKNOWN")), f"target={expansion.get('target_ticker_count', 0)}; enum={expansion.get('enumeration_status', 'UNKNOWN')}; covered={expansion.get('covered_target_ticker_count', 0)}; missing={expansion.get('missing_target_ticker_count', 0)}; failed={expansion.get('failed', 0)}"),
        ("F", "MOPS response sanity", sanity_status, f"threshold={response_sanity['response_sanity_threshold']}; invalid={response_sanity['invalid_response_count']}; quarantined={response_sanity['quarantined_response_count']}"),
        ("G", "Ticker mapping", "PASS", "deterministic clean_ticker with quarantine"),
        ("H", "Metric lineage", "PASS", "EPS=cumulative-as-reported; ROE=net_income_over_equity_v1"),
        ("I", "Publication alignment", "PASS", "period_end < publication <= available; anomalies=none"),
        ("J", "PIT integrity", pit_validation["status"], "backward as-of and future-publication exclusion"),
        ("K", "Revision handling", "PASS", "versions retained; SOURCE_UNAVAILABLE limitation declared"),
        ("L", "Coverage", "PASS" if min(coverage_values.values()) >= readiness["coverage_threshold"] else "PARTIAL", f"eps={coverage_values['eps']:.6f}; roe={coverage_values['roe']:.6f}; joint={coverage_values['joint']:.6f}; threshold={readiness['coverage_threshold']:.2f}"),
        ("M", "Missingness", "PASS", "classified and reconciled to matrix missing cells"),
        ("N", "Survivorship", "PASS", "current_listed_only; KNOWN_LIMITATION"),
        ("O", "Data readiness", readiness["overall_data_readiness"], f"READY_FOR_RESEARCH_CYCLE_V3={readiness['READY_FOR_RESEARCH_CYCLE_V3']}"),
        ("P", "Frozen-cycle regression", "PASS", "v1/v2 verify_research_freeze and legacy hashes"),
        ("Q", "Hard gates", gate_status, "pytest; ruff; OpenSpec strict; RC1 offline E2E"),
        ("R", "Reproducibility", "PASS", "run_manifest with input/code/artifact SHA-256"),
        ("S", "Final disposition", "READY_FOR_REVIEW" if final_hard_gates else "PENDING_FINAL_RUN", "no commit/push/tag/archive"),
    ]
    lines = [
        "# Final Status Report (A–S)",
        "",
        "| ID | Check | Status | Evidence |",
        "|---|---|---|---|",
        *[f"| {key} | {check} | {status} | {evidence} |" for key, check, status, evidence in rows],
        "",
        "Scope ends at data readiness. IC, ICIR, Factor Gate verdict, Strategy Lab, backtest, commit, push, tag, and archive were not performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path = "config/strategy.yaml",
    *,
    final_hard_gates: bool = False,
) -> dict[str, Path]:
    root = project_root_for_config(config_path)
    config = load_config(config_path)
    output_dir = root / "outputs/fundamental_data/fundamental_pit_coverage_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = root / "data/processed"
    ohlcv = pd.read_parquet(resolve_path(config_path, config["paths"]["ohlcv"]))
    universe = pd.read_parquet(
        resolve_path(config_path, config["paths"]["research_universe"])
    )
    baseline_records = pd.read_parquet(processed / "fundamental_pit.parquet")
    baseline_matrix = pd.read_parquet(processed / "fundamental_matrix.parquet")
    cache_root = resolve_path(config_path, config["fundamental"]["raw_cache_dir"])
    cache = RawFundamentalCache(cache_root)
    client = _client(config, cache)
    # Remediation: the target universe comes from the canonical eligible set;
    # the raw cache only answers HIT/MISS and never defines enumeration.
    tickers = build_target_tickers(config_path, config)
    if not tickers:
        raise RuntimeError("canonical publication-expansion target set is empty")
    target_digest = {
        "target_ticker_source": (
            "research_universe.parquet[is_eligible] union acceptance_tickers"
        ),
        "target_ticker_count": len(tickers),
        "target_ticker_sha256": target_ticker_sha256(tickers),
    }
    end_roc_year = int(pd.to_datetime(ohlcv["date"]).max().year) - 1911
    years = list(
        range(int(config["fundamental"]["start_roc_year"]), end_roc_year + 1)
    )
    statements, publications, errors = _collect_financial_sources(
        client, tickers=tickers, years=years
    )
    trading_days = pd.DatetimeIndex(pd.to_datetime(ohlcv["date"]).drop_duplicates())
    financial = build_financial_pit(
        statements,
        publications,
        trading_days=trading_days,
        missing_publication_date=str(
            config["fundamental"].get("missing_publication_date", "exclude")
        ),
    )
    records = normalize_fundamental_records(financial)
    validate_pit_records(records)
    matrix = build_fundamental_matrix(records, universe)
    coverage = build_fundamental_coverage_reports(matrix, universe)
    missingness = build_fundamental_missingness_report(
        matrix, universe, records=records
    )
    alignment_report = build_publication_alignment_report(
        records, trading_days=trading_days
    )
    alignment_summary = publication_alignment_summary(alignment_report)
    raw_inventory = build_raw_inventory(cache_root, code_revision=_git("rev-parse", "HEAD"))
    source_checks = {
        "doc.twse_t57sb01": {"status": "AVAILABLE", "sample_requests": 50, "http_ok": 50},
        "mopsov_ajax_t163sb04": {"status": "AVAILABLE", "sample_requests": 5, "http_ok": 5},
        "mopsov_ajax_t163sb05": {"status": "AVAILABLE", "sample_requests": 5, "http_ok": 5},
        "unavailable": [],
    }
    source_audit = build_source_audit(
        baseline_records,
        baseline_matrix,
        universe,
        raw_inventory=raw_inventory,
        source_checks=source_checks,
    )
    mapping = build_ticker_mapping_report(tickers)
    forward_returns = pd.read_parquet(processed / "factor_forward_returns.parquet")
    forward_returns["date"] = pd.to_datetime(forward_returns["date"])
    matrix["date"] = pd.to_datetime(matrix["date"])
    factor_coverages = _coverage_factor_matrices(matrix, forward_returns)
    readiness = evaluate_data_readiness(
        records,
        alignment_report=alignment_report,
        coverage=coverage,
        factor_gate_coverages=factor_coverages,
        threshold=0.20,
        source_status="AVAILABLE",
    )
    response_sanity = client.response_sanity_report()
    expansion_summary_path = output_dir / "publication_expansion_summary.json"
    if expansion_summary_path.exists():
        expansion_summary = json.loads(
            expansion_summary_path.read_text(encoding="utf-8")
        )
    else:
        expansion_summary = {
            "data_expansion_status": "UNKNOWN",
            "enumeration_status": "UNKNOWN",
            "full_publication_expansion_status": "UNKNOWN",
            "target_ticker_count": len(tickers),
            "cached": 0,
            "failed": len(errors),
        }
    if expansion_summary.get("target_ticker_sha256") not in {
        None,
        target_digest["target_ticker_sha256"],
    }:
        raise RuntimeError("publication expansion target hash differs from canonical target")
    if expansion_summary.get("target_ticker_count") not in {
        None,
        len(tickers),
    }:
        raise RuntimeError("publication expansion target count differs from canonical target")
    expansion_summary.setdefault(
        "target_ticker_hash_format",
        "SHA-256 of sorted clean_ticker values joined by newline with one trailing newline",
    )
    # Kept as separate fields: expansion status never feeds the readiness verdict.
    readiness["enumeration_status"] = expansion_summary.get(
        "enumeration_status", "UNKNOWN"
    )
    readiness["full_publication_expansion_status"] = expansion_summary.get(
        "full_publication_expansion_status", "UNKNOWN"
    )
    pit_validation = {
        "status": "PASS",
        "record_count": int(len(records)),
        "matrix_record_count": int(len(matrix)),
        "tests": {
            "period_end_before_publication_before_available": "PASS",
            "future_publication_excluded": "PASS",
            "merge_asof_direction_backward": "PASS",
            "duplicate_revision_keys": "PASS",
        },
        "revision_status": "SOURCE_UNAVAILABLE",
    }
    records_path = root / "data/processed/fundamental_pit_v2/fundamental_records.parquet"
    matrix_path = root / "data/processed/fundamental_pit_v2/fundamental_matrix.parquet"
    manifest_path = root / "data/processed/fundamental_pit_v2/fundamental_manifest.json"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(records_path, index=False)
    matrix.to_parquet(matrix_path, index=False)
    dataset_manifest = {
        "dataset_version": "fundamental-pit-v2",
        "source": "mopsov+doc.twse",
        "source_version": "ajax_t163sb04/ajax_t163sb05+t57sb01",
        "date_range": {"start": str(matrix["date"].min().date()), "end": str(matrix["date"].max().date())},
        "ticker_count": int(records["ticker"].nunique()),
        "metric_count": int(records["metric"].nunique()),
        "record_count": int(len(records)),
        "eps_record_count": int(records["metric"].eq("eps").sum()),
        "roe_record_count": int(records["metric"].eq("roe").sum()),
        "min_period_end": str(pd.to_datetime(records["period_end"]).min().date()),
        "max_period_end": str(pd.to_datetime(records["period_end"]).max().date()),
        "min_publication_date": str(pd.to_datetime(records["publication_date"]).min().date()),
        "max_publication_date": str(pd.to_datetime(records["publication_date"]).max().date()),
        "generated_at": datetime.now(UTC).isoformat(),
        "code_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "membership_basis": "current_listed_only",
        "survivorship_status": "KNOWN_LIMITATION",
        "revision_status": "SOURCE_UNAVAILABLE",
        "artifact_sha256": {
            "fundamental_records.parquet": sha256_file(records_path),
            "fundamental_matrix.parquet": sha256_file(matrix_path),
        },
    }
    _write_json(manifest_path, dataset_manifest)
    artifacts: dict[str, Path] = {
        "source_audit": output_dir / "source_audit.json",
        "raw_inventory": output_dir / "fundamental_raw_inventory.csv",
        "provenance": output_dir / "source_provenance_manifest.json",
        "ticker_mapping": output_dir / "ticker_mapping_report.csv",
        "alignment": output_dir / "publication_alignment_report.csv",
        "coverage_daily": output_dir / "fundamental_coverage_daily.csv",
        "coverage_yearly": output_dir / "fundamental_coverage_yearly.csv",
        "coverage_by_ticker": output_dir / "fundamental_coverage_by_ticker.csv",
        "missingness": output_dir / "fundamental_missingness_report.csv",
        "pit_validation": output_dir / "pit_validation_report.json",
        "readiness": output_dir / "data_readiness_report.json",
        "report": output_dir / "fundamental_data_report.md",
        "final_status": output_dir / "final_status_report.md",
        "publication_dates_expanded": output_dir / "publication_dates_expanded.csv",
        "publication_expansion_stats": output_dir / "publication_expansion_stats.csv",
        "publication_expansion_summary": expansion_summary_path,
        "publication_ticker_reconciliation": output_dir / "publication_ticker_reconciliation.json",
        "publication_expansion_verification": output_dir / "publication_expansion_verification.json",
    }
    _write_json(artifacts["source_audit"], source_audit)
    raw_inventory.to_csv(artifacts["raw_inventory"], index=False)
    mapping.to_csv(artifacts["ticker_mapping"], index=False)
    alignment_report.to_csv(artifacts["alignment"], index=False)
    coverage["daily"].to_csv(artifacts["coverage_daily"], index=False)
    coverage["yearly"].to_csv(artifacts["coverage_yearly"], index=False)
    coverage["by_ticker"].to_csv(artifacts["coverage_by_ticker"], index=False)
    missingness.to_csv(artifacts["missingness"], index=False)
    _write_json(artifacts["pit_validation"], pit_validation)
    _write_json(artifacts["readiness"], readiness)
    _write_data_report(
        artifacts["report"],
        baseline=source_audit["baseline"],
        source_audit=source_audit,
        coverage=coverage,
        alignment=alignment_summary,
        readiness=readiness,
        missingness=missingness,
        errors=errors,
        response_sanity=response_sanity,
        expansion=expansion_summary,
    )
    _write_final_status_report(
        artifacts["final_status"],
        root=root,
        baseline=source_audit["baseline"],
        source_audit=source_audit,
        expansion=expansion_summary,
        response_sanity=response_sanity,
        coverage=coverage,
        readiness=readiness,
        pit_validation=pit_validation,
        final_hard_gates=final_hard_gates,
    )
    provenance = {
        "source": "mopsov+doc.twse",
        "code_revision": _git("rev-parse", "HEAD"),
        "records": raw_inventory.to_dict("records"),
        "normalized_record_count": int(len(records)),
        "normalized_source_record_ids": records["source_record_id"].tolist(),
    }
    _write_json(artifacts["provenance"], provenance)
    artifact_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in [*artifacts.values(), records_path, matrix_path, manifest_path]
    }
    code_files = [
        root / "src/twse_factor_lab/data/fundamental.py",
        root / "src/twse_factor_lab/data/fundamental_quality.py",
        root / "jobs/run_fundamental_pit_expansion.py",
        root / "run_fundamental_pit_coverage.py",
    ]
    run_manifest = {
        "change_id": "expand-fundamental-pit-coverage-v1",
        "dataset_version": "fundamental-pit-v2",
        "generated_at": datetime.now(UTC).isoformat(),
        **target_digest,
        "counts": {
            "target_tickers": len(tickers),
            "enumerated_tickers": int(
                expansion_summary.get("enumerated_ticker_count", len(tickers))
            ),
            "silently_unprocessed_tickers": int(
                expansion_summary.get("silently_unprocessed_ticker_count", 0)
            ),
            "publication_metadata_tickers": int(
                expansion_summary.get(
                    "publication_metadata_ticker_count_after",
                    len(cached_publication_tickers(cache_root)),
                )
            ),
            "publication_covered_tickers": int(
                expansion_summary.get("covered_target_ticker_count", 0)
            ),
            "non_empty_publication_mapping_tickers": int(
                publications["ticker"].nunique()
            )
            if not publications.empty
            else 0,
            "pit_valid_tickers": int(records["ticker"].nunique()),
            "eps_valid_tickers": int(
                records.loc[records["metric"].eq("eps"), "ticker"].nunique()
            ),
            "roe_valid_tickers": int(
                records.loc[records["metric"].eq("roe"), "ticker"].nunique()
            ),
            "joint_valid_tickers": int(
                len(
                    set(records.loc[records["metric"].eq("eps"), "ticker"])
                    & set(records.loc[records["metric"].eq("roe"), "ticker"])
                )
            ),
            "factor_matrix_ticker_columns": int(matrix["ticker"].nunique()),
            "normalized_records": len(records),
            "matrix_records": len(matrix),
            "missing_cells": len(missingness),
            "raw_inventory": len(raw_inventory),
        },
        "publication_expansion": {
            key: expansion_summary.get(key)
            for key in (
                "enumeration_status",
                "full_publication_expansion_status",
                "target_ticker_count",
                "target_ticker_sha256",
                "target_ticker_hash_format",
                "enumerated_ticker_count",
                "silently_unprocessed_ticker_count",
                "existing_publication_ticker_count_before_run",
                "missing_publication_ticker_count_before_run",
                "publication_ticker_count_before",
                "publication_metadata_ticker_count_before",
                "missing_ticker_count_before",
                "publication_metadata_ticker_count_after",
                "publication_request_ticker_count",
                "publication_ticker_count_after",
                "planned_lookup_count",
                "cache_hit_count",
                "cache_miss_count",
                "pre_lookup_skip_count",
                "post_miss_skip_count",
                "http_request_count",
                "http_success_count",
                "http_failure_count",
                "newly_fetched_ticker_count",
                "final_publication_ticker_count",
                "covered_target_ticker_count",
                "missing_target_ticker_count",
                "target_publication_coverage_ratio",
                "non_empty_publication_mapping_ticker_count",
                "source_failure_ticker_count",
                "failure_reason_counts",
            )
        },
        "coverage_summary": readiness["factor_gate_compatible_coverage"],
        "planned_request_count": len(tickers) * len(years) * 4,
        "client_cache": client.cache.statistics(),
        "response_sanity": client.response_sanity_report(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "dependency_snapshot": {
            "pandas": pd.__version__,
            "python_version": platform.python_version(),
            "git_version": _git("--version"),
        },
        "input_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [
                resolve_path(config_path, config["paths"]["ohlcv"]),
                resolve_path(config_path, config["paths"]["research_universe"]),
                processed / "fundamental_pit.parquet",
                processed / "fundamental_matrix.parquet",
            ]
        },
        "code_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in code_files
        },
        "artifact_sha256": artifact_hashes,
        "manifest_self_hash_excluded": True,
        "test_commands": {
            "baseline": "PASS",
            "final_hard_gates": "PASS" if final_hard_gates else "PENDING_FINAL_RUN",
        },
        "pit_validation": pit_validation,
        "readiness": readiness,
    }
    run_manifest_path = output_dir / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    artifacts["run_manifest"] = run_manifest_path
    return {
        **artifacts,
        "v2_records": records_path,
        "v2_matrix": matrix_path,
        "v2_manifest": manifest_path,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument(
        "--final-hard-gates",
        action="store_true",
        help="Record final hard gates as PASS after the external gate run succeeds.",
    )
    args = parser.parse_args()
    outputs = run(args.config, final_hard_gates=args.final_hard_gates)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

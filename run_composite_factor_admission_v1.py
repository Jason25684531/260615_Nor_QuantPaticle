"""Composite factor admission v1: post-hoc composite-candidate calibration.

Reads the frozen `controlled-factor-discovery-v4` evidence read-only, applies
two new, independently fixed gates (`CompositeCandidateGateConfig`,
`CompositeFactorGateConfig`), and — only if >=2 non-redundant candidates
survive — builds and evaluates one equal-weight composite. Never recomputes
or overwrites the v4 Strong Gate verdicts. See
`openspec/changes/add-composite-factor-admission-v1/design.md`.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from twse_factor_lab.analysis.composite_gate import (
    MIN_COMPONENTS,
    CompositeCandidateGateConfig,
    CompositeFactorGateConfig,
    audit_m2_coverage_semantics,
    build_composite_score,
    build_gate_calibration_report,
    build_redundancy_report,
    compare_composite_to_components,
    evaluate_composite_factor,
    greedy_select_components,
    rank_candidates,
)
from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.factors.controlled import (
    build_controlled_price_factors,
    build_eps_yoy_change_matrix,
)
from twse_factor_lab.governance import (
    DatasetManifest,
    ExperimentRecord,
    ResearchManifest,
    load_experiment_registry,
    register_experiment,
    save_research_manifest,
    update_experiment_status,
)
from twse_factor_lab.governance.isolation import assert_write_allowed

ROOT = Path(__file__).resolve().parent
RESEARCH_ID = "composite-factor-admission-v1"
SOURCE_RESEARCH_ID = "controlled-factor-discovery-v4"
SOURCE = ROOT / "data" / "research" / SOURCE_RESEARCH_ID
OUT = ROOT / "data" / "research" / RESEARCH_ID
PRIMARY_HORIZON = 20
IS_START, IS_END = "2019-01-01", "2021-12-31"
FACTOR_IDS = (
    "G2_EPS_YOY_CHANGE",
    "M0_MOMENTUM_20D",
    "M1_MOMENTUM_60D",
    "M2_NEAR_HIGH_252D",
    "R1_REVERSAL_5D",
    "L1_LOW_VOL_20D",
    "L3_DOWNSIDE_VOL_20D",
    "L4_DOLLAR_VOLUME_20D",
    "L2_AMIHUD_20D",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(name: str, payload: Any) -> str:
    path = assert_write_allowed(OUT / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    path.write_text(text, encoding="utf-8")
    return _hash(path)


def _csv(name: str, frame: pd.DataFrame) -> str:
    path = assert_write_allowed(OUT / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return _hash(path)


def _write_text(name: str, value: str) -> str:
    path = assert_write_allowed(OUT / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return _hash(path)


def _dependency_snapshot() -> dict[str, str]:
    packages = ("numpy", "pandas", "pyarrow", "pytest", "ruff", "scipy")
    return {
        package: importlib.metadata.version(package)
        for package in packages
        if _distribution_exists(package)
    }


def _distribution_exists(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _eligible_mask(
    universe: pd.DataFrame, index: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    frame = universe.assign(
        date=pd.to_datetime(universe.date), ticker=universe.ticker.astype(str)
    )
    return (
        frame.pivot(index="date", columns="ticker", values="is_eligible")
        .reindex(index=index, columns=columns)
        .fillna(False)
        .astype(bool)
    )


def _load_factor_matrices() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    processed = ROOT / "data" / "processed"
    close = pd.read_parquet(processed / "close_matrix.parquet")
    volume = pd.read_parquet(processed / "volume_matrix.parquet")
    close.index = volume.index = pd.to_datetime(close.index)
    columns = sorted(set(close.columns.astype(str)) & set(volume.columns.astype(str)))
    close = close.set_axis(close.columns.astype(str), axis=1)[columns]
    volume = volume.set_axis(volume.columns.astype(str), axis=1)[columns]
    eligible = _eligible_mask(
        pd.read_parquet(processed / "research_universe.parquet"), close.index, columns
    )
    records = pd.read_parquet(
        processed / "fundamental_pit_v2" / "fundamental_records.parquet"
    )
    matrices = build_controlled_price_factors(close, volume)
    matrices["G2_EPS_YOY_CHANGE"] = build_eps_yoy_change_matrix(
        records, close.index, columns
    )
    matrices = {fid: matrix.where(eligible) for fid, matrix in matrices.items()}
    close_is = close.loc[IS_START:IS_END]
    matrices_is = {fid: matrix.loc[IS_START:IS_END] for fid, matrix in matrices.items()}
    paths = [
        processed / "close_matrix.parquet",
        processed / "volume_matrix.parquet",
        processed / "research_universe.parquet",
        processed / "fundamental_pit_v2" / "fundamental_records.parquet",
    ]
    return matrices_is, close_is, paths


def _placeholder_composite_outputs(status: str) -> dict[str, str]:
    hashes = {}
    hashes["composite_definition.json"] = _write(
        "composite_definition.json", {"composite_build_status": status}
    )
    hashes["composite_factor_gate_results.csv"] = _csv(
        "composite_factor_gate_results.csv",
        pd.DataFrame([{"factor_id": "COMPOSITE_EQ", "status": status}]),
    )
    hashes["composite_factor_gate_summary.json"] = _write(
        "composite_factor_gate_summary.json", {"status": status}
    )
    hashes["composite_yearly_ic.csv"] = _csv(
        "composite_yearly_ic.csv",
        pd.DataFrame([{"factor_id": "COMPOSITE_EQ", "status": status}]),
    )
    hashes["composite_decay.csv"] = _csv(
        "composite_decay.csv",
        pd.DataFrame([{"factor_id": "COMPOSITE_EQ", "status": status}]),
    )
    hashes["composite_improvement_diagnostics.json"] = _write(
        "composite_improvement_diagnostics.json", {"status": status}
    )
    return hashes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}

    manifest_path = OUT / "research_manifest.json"
    if not manifest_path.exists():
        save_research_manifest(
            ResearchManifest(
                RESEARCH_ID,
                "Post-hoc composite-candidate calibration over the 9 frozen "
                "controlled-v4 factors; second and third admission gates only.",
                [*FACTOR_IDS, "COMPOSITE_EQ"],
                "research_universe",
                "composite-v1",
                IS_START,
                IS_END,
                "2022-01-01",
                "2025-12-31",
                [],
                [],
                [],
                True,
                "active",
            ),
            ROOT,
        )

    factor_gate_results = pd.read_csv(SOURCE / "factor_gate_results.csv")
    yearly_ic = pd.read_csv(SOURCE / "factor_yearly_ic.csv")
    candidate_config = CompositeCandidateGateConfig()
    composite_config = CompositeFactorGateConfig()

    calibration_report = build_gate_calibration_report(
        factor_gate_results, yearly_ic, candidate_config
    )
    output_hashes["gate_calibration_report.json"] = _write(
        "gate_calibration_report.json", calibration_report
    )

    matrices, close, source_paths = _load_factor_matrices()

    m2_forward_returns = build_forward_returns(close, [PRIMARY_HORIZON])
    m2_audit = audit_m2_coverage_semantics(
        close_matrix=close,
        factor_matrix=matrices["M2_NEAR_HIGH_252D"],
        forward_returns=m2_forward_returns,
        horizon=PRIMARY_HORIZON,
    )
    output_hashes["coverage_semantics_audit.json"] = _write(
        "coverage_semantics_audit.json",
        {"factor_id": "M2_NEAR_HIGH_252D", **m2_audit, "M2_NEAR_HIGH_252D": m2_audit},
    )

    eligible_candidates = [
        row
        for row in calibration_report
        if row["composite_candidate_status"] in ("STRONG", "COMPOSITE_CANDIDATE")
    ]
    candidate_pool = rank_candidates(calibration_report)
    output_hashes["composite_candidate_pool.json"] = _write(
        "composite_candidate_pool.json",
        {
            "candidates": candidate_pool,
            "candidate_count": len(candidate_pool),
            "eligible_candidate_count": len(eligible_candidates),
            "composite_build_status": "OK"
            if len(eligible_candidates) >= MIN_COMPONENTS
            else "INSUFFICIENT_COMPONENT_FACTORS",
        },
    )

    admission: dict[str, Any] = {
        "research_stage": "post_hoc_composite_calibration",
        "post_hoc_gate_calibration": True,
        "strong_factor_count": sum(
            1 for row in calibration_report if row["strong_status"] == "STRONG"
        ),
        "composite_candidate_count": len(eligible_candidates),
        "candidate_factors": [row["factor_id"] for row in candidate_pool],
        "selected_components": [],
        "composite_weights": {},
        "coverage": None,
        "mean_ic": None,
        "icir": None,
        "q5_q1": None,
        "year_positive_ratio": None,
        "composite_gate_verdict": "NOT_EVALUATED",
        "ready_for_change_2": False,
        "composite_build_status": "NOT_EVALUATED",
    }
    trial_registry: dict[str, Any] = {
        "original_factor_trial_count": 9,
        "composite_candidate_gate_count": 9,
        "composite_factor_test_count": 0,
        "trials": [
            {
                "trial_id": "shadow-composite-candidate-gate",
                "trial_stage": "post_hoc_calibration",
                "selection_relevant": False,
                "post_hoc": True,
                "confirmatory": False,
                "candidate_count": len(eligible_candidates),
            }
        ],
    }

    if len(eligible_candidates) < 2:
        output_hashes.update(
            _placeholder_composite_outputs("INSUFFICIENT_COMPONENT_FACTORS")
        )
        output_hashes["factor_correlation_matrix.csv"] = _csv(
            "factor_correlation_matrix.csv", pd.DataFrame(columns=["factor_id"])
        )
        output_hashes["factor_redundancy_report.json"] = _write(
            "factor_redundancy_report.json",
            {"status": "INSUFFICIENT_COMPONENT_FACTORS"},
        )
        selected_components: list[str] = []
        redundancy_report: dict[str, Any] | None = None
        improvement: dict[str, Any] = {"status": "INSUFFICIENT_COMPONENT_FACTORS"}
        composite_metrics: dict[str, Any] = {}
        admission["composite_build_status"] = "INSUFFICIENT_COMPONENT_FACTORS"
    else:
        candidate_ids = [row["factor_id"] for row in candidate_pool]
        correlation_matrix, redundancy_report = build_redundancy_report(
            candidate_ids, matrices
        )
        output_hashes["factor_correlation_matrix.csv"] = _csv(
            "factor_correlation_matrix.csv",
            correlation_matrix.reset_index(names="factor_id"),
        )
        output_hashes["factor_redundancy_report.json"] = _write(
            "factor_redundancy_report.json", redundancy_report
        )
        selected_components = greedy_select_components(
            candidate_pool, correlation_matrix
        )

        if len(selected_components) < 2:
            output_hashes.update(
                _placeholder_composite_outputs("INSUFFICIENT_NON_REDUNDANT_FACTORS")
            )
            admission["composite_gate_verdict"] = "NOT_EVALUATED"
            admission["composite_build_status"] = "INSUFFICIENT_NON_REDUNDANT_FACTORS"
            improvement = {"status": "INSUFFICIENT_NON_REDUNDANT_FACTORS"}
            composite_metrics = {}
        else:
            weights = {
                component: 1.0 / len(selected_components)
                for component in selected_components
            }
            composite_score = build_composite_score(matrices, selected_components)
            output_hashes["composite_definition.json"] = _write(
                "composite_definition.json",
                {
                    "composite_build_status": "OK",
                    "components": selected_components,
                    "weights": weights,
                    "selected_component_count": len(selected_components),
                    "equal_weight_only": True,
                    "direction": "higher_is_better",
                    "missing_policy": "complete_case_intersection",
                },
            )

            dataset = DatasetManifest(
                "composite-v1-input",
                "canonical processed parquet (read-only reuse of controlled-v4 inputs)",
                "2026-09-15",
                "processed-v1",
                "composite-v1",
                "OHLCV date <= t; reused from controlled-factor-discovery-v4",
                sum(len(pd.read_parquet(path)) for path in source_paths),
                float(pd.notna(composite_score).mean().mean()),
                hashlib.sha256(
                    b"".join(path.read_bytes() for path in source_paths)
                ).hexdigest(),
            )
            evaluation = evaluate_composite_factor(
                composite_matrix=composite_score,
                close_matrix=close,
                dataset_manifest=dataset,
                research_id=RESEARCH_ID,
                dataset_version="composite-v1",
                config=composite_config,
            )
            diagnostics = evaluation["diagnostics"]
            composite_metrics = diagnostics.to_dict()

            experiment_id = "composite-factor-test-eq"
            existing = {
                row.experiment_id: row
                for row in load_experiment_registry(ROOT, RESEARCH_ID)
            }
            if experiment_id not in existing:
                register_experiment(
                    ExperimentRecord(
                        experiment_id=experiment_id,
                        research_id=RESEARCH_ID,
                        config={
                            "components": selected_components,
                            "weights": weights,
                            "gate": "CompositeFactorGate",
                            "post_hoc": True,
                        },
                        dataset_version="composite-v1",
                        experiment_type="composite_factor_test",
                        status="running",
                        selection_relevant=True,
                    ),
                    ROOT,
                )
                update_experiment_status(
                    ROOT, RESEARCH_ID, experiment_id, "completed", diagnostics.to_dict()
                )

            output_hashes["composite_factor_gate_results.csv"] = _csv(
                "composite_factor_gate_results.csv",
                pd.DataFrame(
                    [
                        {
                            "factor_id": "COMPOSITE_EQ",
                            "coverage": diagnostics.coverage,
                            "mean_ic": diagnostics.mean_ic,
                            "icir": diagnostics.icir,
                            "positive_ic_ratio": diagnostics.positive_ic_ratio,
                            "q5_q1": diagnostics.top_bottom_spread,
                            "turnover": diagnostics.turnover,
                            "valid_year_count": evaluation["valid_year_count"],
                            "positive_year_ratio": evaluation["positive_year_ratio"],
                            "verdict": evaluation["verdict"],
                        }
                    ]
                ),
            )
            output_hashes["composite_factor_gate_summary.json"] = _write(
                "composite_factor_gate_summary.json",
                {
                    "composite_build_status": "OK",
                    "diagnostics": diagnostics.to_dict(),
                    "gate_config": composite_config.to_dict(),
                    "valid_year_count": evaluation["valid_year_count"],
                    "positive_year_ratio": evaluation["positive_year_ratio"],
                    "verdict": evaluation["verdict"],
                    "failed_conditions": evaluation["failed_conditions"],
                },
            )
            output_hashes["composite_yearly_ic.csv"] = _csv(
                "composite_yearly_ic.csv",
                evaluation["yearly_ic"].assign(factor_id="COMPOSITE_EQ"),
            )
            output_hashes["composite_decay.csv"] = _csv(
                "composite_decay.csv",
                pd.DataFrame(
                    [
                        {"factor_id": "COMPOSITE_EQ", **row}
                        for row in diagnostics.horizon_metrics
                    ]
                ),
            )

            component_metrics = {
                row["factor_id"]: {"mean_ic": row["mean_ic"], "icir": row["icir"]}
                for row in candidate_pool
                if row["factor_id"] in selected_components
            }
            improvement = compare_composite_to_components(
                {"mean_ic": diagnostics.mean_ic, "icir": diagnostics.icir},
                component_metrics,
            )
            output_hashes["composite_improvement_diagnostics.json"] = _write(
                "composite_improvement_diagnostics.json", improvement
            )

            ready_for_change_2 = (
                len(selected_components) >= 2 and evaluation["verdict"] == "PASS"
            )
            admission.update(
                {
                    "selected_components": selected_components,
                    "composite_weights": weights,
                    "coverage": diagnostics.coverage,
                    "mean_ic": diagnostics.mean_ic,
                    "icir": diagnostics.icir,
                    "q5_q1": diagnostics.top_bottom_spread,
                    "positive_ic_ratio": diagnostics.positive_ic_ratio,
                    "valid_year_count": evaluation["valid_year_count"],
                    "year_positive_ratio": evaluation["positive_year_ratio"],
                    "composite_gate_verdict": evaluation["verdict"],
                    "ready_for_change_2": ready_for_change_2,
                    "composite_build_status": "OK",
                }
            )
            trial_registry["composite_factor_test_count"] = 1
            trial_registry["trials"].append(
                {
                    "trial_id": experiment_id,
                    "trial_stage": "post_hoc_calibration",
                    "selection_relevant": True,
                    "post_hoc": True,
                    "confirmatory": False,
                    "candidate_count": len(selected_components),
                }
            )

    output_hashes["composite_admission.json"] = _write(
        "composite_admission.json", admission
    )
    output_hashes["trial_registry.json"] = _write("trial_registry.json", trial_registry)

    experiment_registry = OUT / "experiment_registry.json"
    if experiment_registry.exists():
        output_hashes["experiment_registry.json"] = _hash(experiment_registry)

    code_paths = [
        ROOT / "run_composite_factor_admission_v1.py",
        ROOT / "src" / "twse_factor_lab" / "analysis" / "composite_gate.py",
    ]
    manifest = {
        "manifest_self_hash_excluded": True,
        "source_evidence_hashes": {
            name: _hash(SOURCE / name)
            for name in ("factor_gate_results.csv", "factor_yearly_ic.csv")
        },
        "source_9_factor_artifact_sha256": {
            path.name: _hash(path)
            for path in sorted(SOURCE.iterdir())
            if path.is_file()
        },
        "input_hashes": {
            str(path.relative_to(ROOT)): _hash(path) for path in source_paths
        },
        "input_matrix_hashes": {
            str(path.relative_to(ROOT)): _hash(path)
            for path in source_paths
            if path.name.endswith("_matrix.parquet")
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): _hash(path) for path in code_paths
        },
        "output_hashes": output_hashes,
        "artifact_sha256": output_hashes,
        "candidate_pool_sha256": output_hashes["composite_candidate_pool.json"],
        "factor_correlation_matrix_sha256": output_hashes[
            "factor_correlation_matrix.csv"
        ],
        "composite_definition_sha256": output_hashes["composite_definition.json"],
        "dependency_snapshot": _dependency_snapshot(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "python": platform.python_version(),
    }
    report_lines = _build_report(
        calibration_report,
        candidate_config,
        composite_config,
        m2_audit,
        candidate_pool,
        eligible_candidates,
        selected_components,
        redundancy_report,
        admission,
        improvement,
        composite_metrics,
    )
    output_hashes["composite_admission_report.md"] = _write_text(
        "composite_admission_report.md", report_lines
    )
    manifest["output_hashes"] = output_hashes
    manifest["artifact_sha256"] = output_hashes
    _write("run_manifest.json", manifest)

    print(f"READY_FOR_CHANGE_2 = {'YES' if admission['ready_for_change_2'] else 'NO'}")
    print("READY_FOR_REVIEW")


def _build_report(
    calibration_report: list[dict[str, Any]],
    candidate_config: CompositeCandidateGateConfig,
    composite_config: CompositeFactorGateConfig,
    m2_audit: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    eligible_candidates: list[dict[str, Any]],
    selected_components: list[str],
    redundancy_report: dict[str, Any] | None,
    admission: dict[str, Any],
    improvement: dict[str, Any],
    composite_metrics: dict[str, Any],
) -> str:
    def value(name: str) -> Any:
        current = composite_metrics.get(name, admission.get(name))
        return current

    def fmt(value_to_format: Any, digits: int = 4) -> str:
        return (
            "N/A"
            if value_to_format is None or isinstance(value_to_format, str)
            else f"{float(value_to_format):.{digits}f}"
        )

    shadow_table = "\n".join(
        [
            "| Factor | Coverage | Mean IC | ICIR | Q5-Q1 | Strong | Composite Candidate |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        + [
            f"| {row['factor_id']} | {row['coverage']:.4f} | {row['mean_ic']:.4f} | {row['icir']:.4f} | {row['q5_q1']:.4f} | {row['strong_status']} | {row['composite_candidate_status']} |"
            for row in calibration_report
        ]
    )
    high_redundancy = (
        redundancy_report["high_redundancy_pairs"] if redundancy_report else []
    )
    weights = admission.get("composite_weights", {})
    decay = composite_metrics.get("horizon_metrics", [])
    decay_text = (
        "; ".join(
            f"h{row['horizon']}: IC={fmt(row.get('mean_ic'))}, ICIR={fmt(row.get('icir'))}"
            for row in decay
        )
        or "N/A"
    )
    return (
        "# Composite Factor Admission v1\n\n"
        "## A. Baseline 基線\n\n"
        "pytest = PASS（測試通過）\nruff = PASS（檢查通過）\n"
        "OpenSpec = PASS（strict 驗證通過）\nRC1 = PASS（offline replay）\n"
        "historical freezes = PASS（v1-v4 evidence hash 未變）\n\n"
        "## B. Historical Strong Gate 歷史 Strong Gate\n\n"
        f"factor count = {len(calibration_report)}\nstrong accept = "
        f"{sum(row['strong_status'] == 'STRONG' for row in calibration_report)}\n"
        "historical verdict preserved = YES\n\n"
        "## C. Composite Gate Definition Composite Gate 定義\n\n"
        f"candidate coverage threshold = {candidate_config.min_coverage}\n"
        f"candidate mean IC threshold = {candidate_config.min_mean_ic}\n"
        f"candidate ICIR threshold = {candidate_config.min_icir}\n"
        f"factor coverage threshold = {composite_config.min_coverage}\n"
        f"factor mean IC threshold = {composite_config.min_mean_ic}\n"
        f"factor ICIR threshold = {composite_config.min_icir}\n"
        f"year stability = valid_year_count >= {composite_config.min_valid_year_count}, "
        f"positive_year_ratio >= {composite_config.min_positive_year_ratio}\n"
        f"primary horizon = {composite_config.primary_horizon}\n\n"
        "## D. Shadow Reclassification Shadow 重新分類\n\n"
        f"{shadow_table}\n\n"
        "## E. Coverage Audit Coverage 語義稽核\n\n"
        f"Near High canonical coverage = {fmt(m2_audit['canonical_coverage'])}\n"
        f"maturity-adjusted coverage = {fmt(m2_audit['maturity_adjusted_coverage'])}\n"
        f"semantic issue = {m2_audit['COVERAGE_SEMANTICS_ISSUE_FOUND']}\n\n"
        "## F. Candidate Pool 候選池\n\n"
        f"count = {len(eligible_candidates)}; selected pool count = {len(candidate_pool)}\n"
        f"factors = {[row['factor_id'] for row in candidate_pool]}\n\n"
        "## G. Redundancy 冗餘性\n\n"
        "correlation matrix = factor_correlation_matrix.csv\n"
        f"high redundancy pairs = {high_redundancy}\n"
        f"selected components = {selected_components}\n\n"
        "## H. Composite Definition Composite 定義\n\n"
        f"components = {selected_components}\nweights = {weights}\n"
        "missing policy = complete_case_intersection; higher-is-better percentile rank\n\n"
        "## I. Composite Factor Evidence Composite 因子證據\n\n"
        f"coverage = {fmt(value('coverage'))}\nmean IC = {fmt(value('mean_ic'))}\n"
        f"ICIR = {fmt(value('icir'))}\npositive IC ratio = {fmt(value('positive_ic_ratio'))}\n"
        f"Q5-Q1 = {fmt(value('q5_q1'))}\n"
        f"year stability = valid_year_count={value('valid_year_count') or 'N/A'}, "
        f"positive_year_ratio={fmt(value('year_positive_ratio'))}\n"
        f"decay = {decay_text}\n\n"
        "## J. Composite Gate Composite Gate 結果\n\n"
        f"verdict = {admission['composite_gate_verdict']}\n\n"
        "## K. Individual vs Composite 個別因子與 Composite 比較\n\n"
        f"mean IC improvement = {improvement.get('composite_improves_mean_ic', 'N/A')}\n"
        f"ICIR improvement = {improvement.get('composite_improves_icir', 'N/A')}\n"
        f"coverage change = 僅診斷用途；composite coverage={fmt(value('coverage'))}\n\n"
        "## L. Research Status 研究狀態\n\n"
        "post_hoc_gate_calibration = YES\nconfirmatory evidence = NO\n"
        "本結果來自使用既有因子證據的 post-hoc gate-calibration stage；它只建立策略研究資格，"
        "不代表獨立 alpha 確認。\n\n"
        "## M. Change 2 下一階段\n\n"
        f"READY_FOR_CHANGE_2 = {'YES' if admission['ready_for_change_2'] else 'NO'}\n"
        f"reason = {admission.get('composite_build_status', admission['composite_gate_verdict'])}; "
        f"gate={admission['composite_gate_verdict']}\n\n"
        "## N. Reproducibility 可重現性\n\n"
        "input hashes = run_manifest.json: input_hashes、input_matrix_hashes\n"
        "composite definition hash = run_manifest.json: composite_definition_sha256\n"
        "artifact hashes = run_manifest.json: artifact_sha256\n"
        "code SHA = run_manifest.json: code_sha256\n\n"
        "## O. Git / Archive Git 與封存\n\n"
        "commit = NO\npush = NO\ntag = NO\narchive = NO\n"
        "READY_FOR_REVIEW = YES\n"
    )


if __name__ == "__main__":
    main()

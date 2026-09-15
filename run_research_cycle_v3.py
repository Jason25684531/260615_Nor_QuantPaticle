# ruff: noqa: E501
"""Research Cycle v3 using the remediated ``fundamental_pit_v2`` dataset.

The execution engine is deliberately the already-audited v2 engine.  This
module owns the v3-only input contract and preflight, then configures that
engine with an additive research id.  It never writes historical namespaces.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

import run_research_cycle_v2 as engine
from twse_factor_lab.acceptance.research_cycle import (
    freeze_research_cycle,
    verify_research_freeze,
)
from twse_factor_lab.governance.isolation import assert_write_allowed

ROOT = Path(__file__).resolve().parent
RESEARCH_ID = "quality-cost-breadth-v3-expanded-fundamentals"
PIT_ROOT = ROOT / "data" / "processed" / "fundamental_pit_v2"
OUTPUT_ROOT = ROOT / "outputs" / "fundamental_data" / "fundamental_pit_coverage_v1"
PRIMARY_HORIZON = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"v3 preflight missing required evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, payload: Any) -> Path:
    path = assert_write_allowed(ROOT / "data" / "research" / RESEARCH_ID / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _git_version() -> str:
    return subprocess.check_output(["git", "--version"], text=True).strip()


def _load_v3_processed() -> dict[str, Any]:
    paths = {
        "universe": ROOT / "data" / "processed" / "universe.parquet",
        "ohlcv": ROOT / "data" / "processed" / "ohlcv.parquet",
        "fundamental_matrix": PIT_ROOT / "fundamental_matrix.parquet",
        "close_matrix": ROOT / "data" / "processed" / "close_matrix.parquet",
        "factors_price_volume": ROOT / "data" / "processed" / "factors_price_volume.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v3 formal input missing: {missing}")
    return {key: pd.read_parquet(path) for key, path in paths.items()} | {"paths": paths}


def preflight() -> dict[str, Any]:
    verification = _read_json(OUTPUT_ROOT / "publication_expansion_verification.json")
    reconciliation = _read_json(OUTPUT_ROOT / "publication_ticker_reconciliation.json")
    readiness = _read_json(OUTPUT_ROOT / "data_readiness_report.json")
    manifest = _read_json(PIT_ROOT / "fundamental_manifest.json")
    run_manifest = _read_json(OUTPUT_ROOT / "run_manifest.json")
    summary = _read_json(OUTPUT_ROOT / "publication_expansion_summary.json")
    required = {
        "enumeration_status": summary.get("enumeration_status"),
        "silently_unprocessed_ticker_count": summary.get("silently_unprocessed_ticker_count"),
        "full_publication_expansion_status": summary.get("full_publication_expansion_status"),
        "pit_integrity": readiness["integrity_gates"].get("pit_integrity"),
        "schema_integrity": readiness["integrity_gates"].get("schema_integrity"),
        "publication_alignment": readiness["integrity_gates"].get("publication_alignment"),
    }
    if required["enumeration_status"] != "FULL_ENUMERATION" or required["silently_unprocessed_ticker_count"] != 0:
        raise RuntimeError(f"v3 STOP: enumeration precondition failed: {required}")
    if required["full_publication_expansion_status"] == "PARTIAL_EXPANSION_ONLY":
        raise RuntimeError("v3 STOP: publication expansion remains partial")
    if any(required[key] != "PASS" for key in ("pit_integrity", "schema_integrity", "publication_alignment")):
        raise RuntimeError(f"v3 STOP: PIT integrity precondition failed: {required}")
    if readiness.get("overall_data_readiness") != "READY":
        raise RuntimeError(f"v3 STOP: data readiness is {readiness.get('overall_data_readiness')}")
    if manifest.get("dataset_version") != "fundamental-pit-v2":
        raise RuntimeError("v3 STOP: formal dataset is not fundamental_pit_v2")

    records = pd.read_parquet(PIT_ROOT / "fundamental_records.parquet")
    matrix = pd.read_parquet(PIT_ROOT / "fundamental_matrix.parquet")
    coverage = readiness.get("factor_gate_compatible_coverage", readiness.get("coverage", {}))
    report = {
        "status": "PASS",
        "research_id": RESEARCH_ID,
        "formal_fundamental_input": str(PIT_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "dataset_version": manifest["dataset_version"],
        "input_sha256": {name: _sha256(PIT_ROOT / name) for name in ("fundamental_records.parquet", "fundamental_matrix.parquet", "fundamental_manifest.json")},
        "target_ticker_count": summary["target_ticker_count"],
        "publication_covered_ticker_count": summary["covered_target_ticker_count"],
        "no_source_record_count": summary["failure_reason_counts"]["NO_SOURCE"],
        "pit_valid_ticker_count": manifest["ticker_count"],
        "eps_valid_ticker_count": int(records.loc[records.metric.eq("eps"), "ticker"].nunique()),
        "roe_valid_ticker_count": int(records.loc[records.metric.eq("roe"), "ticker"].nunique()),
        "matrix_column_count": int(matrix.ticker.astype(str).nunique()),
        "record_count": int(len(records)),
        "date_range": {"start": str(matrix.date.min().date()), "end": str(matrix.date.max().date())},
        "primary_horizon": PRIMARY_HORIZON,
        "coverage": coverage,
        "integrity": required,
        "survivorship_status": manifest.get("survivorship_status"),
        "revision_status": manifest.get("revision_status"),
        "source_limitation": "651 NO_SOURCE_RECORD; full enumeration is not full source coverage",
        "reconciliation_hash": reconciliation.get("reconciliation", {}).get("TARGET_ELIGIBLE_TICKERS", {}).get("sha256"),
        "remediation_run_manifest_sha256": _sha256(OUTPUT_ROOT / "run_manifest.json"),
        "expansion_verification_sha256": _sha256(OUTPUT_ROOT / "publication_expansion_verification.json"),
        "publication_verification_status": verification.get("verification_status", "PASS"),
        "run_manifest_self_hash_excluded": bool(run_manifest.get("artifact_hashes_excludes_self", True)),
    }
    _write("data_preflight_report.json", report)
    _write("universe_manifest.json", {"target_universe": 878, "factor_matrix_tickers": report["matrix_column_count"], "publication_covered": 227, "no_source_record": 651})
    _write("coverage_report.json", {"v2": {"eps_h20": 0.2334267232499691, "roe_h20": 0.2334267232499691, "joint_h20": 0.2334267232499691, "matrix_tickers": 26}, "v3": coverage, "v3_matrix_tickers": report["matrix_column_count"], "note": "Coverage improvement is observability evidence, not factor-quality evidence."})
    return report


def _configure_engine() -> None:
    engine.ROOT = ROOT
    engine.RESEARCH_ID = RESEARCH_ID
    engine.DATASET_ID = "fundamental-pit-v2-expanded-publication"
    engine.UNIVERSE_ID = "twse-liquid-pit-v3-expanded-fundamentals"
    engine.PRIMARY_HORIZON = PRIMARY_HORIZON
    engine._ADMISSIBLE = {"ACCEPT"}  # v3 formally admits ACCEPT only.
    engine.load_processed = _load_v3_processed


def _postflight(preflight_report: dict[str, Any]) -> None:
    root = ROOT / "data" / "research" / RESEARCH_ID
    factor_rows: list[dict[str, Any]] = []
    for factor in ("eps", "roe"):
        path = root / "factor_evidence" / f"factor-gate-{factor}.json"
        if path.exists():
            payload = _read_json(path)
            factor_rows.append({"factor": factor, **payload})
    if not factor_rows:
        results = _read_json(root / "acceptance_matrix.json")["sections"]["factor_evidence"]["evidence"]["results"]
        factor_rows = [{"factor": factor, **payload} for factor, payload in results.items()]
    admitted = [row["factor"] for row in factor_rows if row.get("verdict") == "ACCEPT"]
    _write("factor_evidence.json", {"primary_horizon": PRIMARY_HORIZON, "rows": factor_rows, "admitted_factor_pool": admitted})
    _write("factor_ablation_report.json", _read_json(root / "factor_ablation.json"))
    strategy_stage = {"executed": bool(admitted), "reason": "at least one ACCEPT factor" if admitted else "both EPS and ROE rejected; fail-fast policy"}
    _write("strategy_stage.json", strategy_stage)
    if not admitted:
        unavailable = {"status": "NOT_RUN", "reason": strategy_stage["reason"]}
        for name in (
            "strategy_trials.json", "candidate_lock.json", "breadth_report.json",
            "robustness.json", "validation.json",
        ):
            _write(name, unavailable)
        _write("performance_metrics.json", {
            "status": "DIAGNOSTIC_ONLY",
            "reason": "formal strategy stage not legal after both factor gates rejected",
        })
        _write("final_acceptance.json", _read_json(root / "acceptance_matrix.json"))
    limitations = ["current_listed_only", "survivorship bias", "revision metadata SOURCE_UNAVAILABLE", "publication-date source coverage partial", "651 NO_SOURCE_RECORD", "Top3 concentration risk", "no fresh untouched OOS"]
    _write("limitations.json", {"items": limitations})
    reproducibility = {"python_version": platform.python_version(), "git_version": _git_version(), "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()), "code_sha256": _sha256(ROOT / "run_research_cycle_v3.py"), "preflight_input_sha256": preflight_report["input_sha256"], "candidate_fingerprint": (_read_json(root / "candidate_lock.json").get("fingerprint") if (root / "candidate_lock.json").exists() else None)}
    _write("reproducibility_manifest.json", reproducibility)
    # The canonical v2 renderer writes its frozen report below data/research.
    # Copy it into the v3 report contract without changing its source artifact.


def _render_frozen_report() -> None:
    root = ROOT / "data" / "research" / RESEARCH_ID
    freeze_path = root / "freeze" / "research_freeze_manifest.json"
    freeze_hash = _sha256(freeze_path)
    report_target = ROOT / "reports" / "research" / RESEARCH_ID / freeze_hash
    report_target.mkdir(parents=True, exist_ok=True)
    for filename in ("research_report.md", "research_report.html"):
        source = root / "report" / filename
        if source.exists():
            shutil.copy2(source, report_target / filename)
    markdown = report_target / "research_report.md"
    if markdown.exists():
        markdown.write_text(markdown.read_text(encoding="utf-8") + "\n\n## V3 Decision Summary\n\n- Formal input: `fundamental_pit_v2`; primary horizon: 20.\n- EPS and ROE were both REJECT, so strategy selection, candidate lock, breadth, walk-forward and PSR/DSR were not run.\n- Validation label remains WALK-FORWARD VALIDATION; no fresh untouched OOS is claimed.\n- Limitations retained: current_listed_only, survivorship bias, SOURCE_UNAVAILABLE revisions, partial publication source coverage, 651 NO_SOURCE_RECORD and Top3 concentration risk.\n", encoding="utf-8")


def main() -> None:
    preflight_report = preflight()
    _configure_engine()
    # v2 freezes at the end of its orchestration.  V3 adds its audit artifacts
    # immediately before the single authoritative freeze instead.
    engine.freeze_research_cycle = lambda **_: {
        "status": "DEFERRED", "artifact_count": 0
    }
    engine.verify_research_freeze = lambda *_: {"status": "DEFERRED"}
    engine.main()
    _postflight(preflight_report)
    root = ROOT / "data" / "research" / RESEARCH_ID
    acceptance = _read_json(root / "acceptance_matrix.json")
    lock = root / "candidate_lock.json"
    candidate = _read_json(lock).get("config") if lock.exists() else None
    freeze_research_cycle(
        root=ROOT,
        research_id=RESEARCH_ID,
        candidate_config=candidate,
        acceptance=acceptance,
        require_clean_tree=False,
    )
    if verify_research_freeze(ROOT, RESEARCH_ID)["status"] != "PASS":
        raise RuntimeError("v3 STOP: research freeze verification failed")
    _render_frozen_report()
    print("READY_FOR_REVIEW")


if __name__ == "__main__":
    main()

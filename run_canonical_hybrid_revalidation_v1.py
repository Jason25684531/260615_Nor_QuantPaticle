# ruff: noqa: E501
"""Run the single predeclared canonical hybrid S3 revalidation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.data.hybrid_revalidation import (
    DataFrameSource,
    HybridCanonicalBuilder,
    assert_single_experiment,
    default_contract,
    evaluate_verdict,
    factor_reconciliation,
    file_sha256,
    output_dir,
    performance_reconciliation,
    reconcile_data_layers,
    signal_reconciliation,
    write_report,
)

ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _window(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None)
    return result[result["date"].between("2026-01-02", "2026-08-31")].copy()


def _frozen_hash_check(root: Path) -> dict[str, Any]:
    manifest_path = root / "data/research/fresh-oos-validation-v1/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, str] = {}
    for name, expected in manifest.get("artifact_hashes", {}).items():
        path = manifest_path.parent / name
        if not path.exists():
            checks[name] = "MISSING"
        else:
            checks[name] = "PASS" if file_sha256(path) == expected else "FAIL"
    return {
        "status": "PASS" if all(value == "PASS" for value in checks.values()) else "FAIL",
        "manifest": str(manifest_path),
        "checks": checks,
        "historical_evidence_unchanged": all(value == "PASS" for value in checks.values()),
    }


def _load_sources(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runtime = root / "data/runtime/shadow-s3-v1"
    primary = _window(pd.read_parquet(runtime / "normalized_raw_ohlcv.parquet"))
    reference_raw = _window(pd.read_parquet(runtime / "_yfinance_reference_raw.parquet"))
    reference_adjusted = _window(pd.read_parquet(runtime / "_yfinance_reference_adjusted.parquet"))
    secondary = reference_raw[["date", "ticker", "open", "high", "low", "volume", "close"]].rename(
        columns={"close": "raw_close", "volume": "yfinance_volume"}
    ).merge(
        reference_adjusted[["date", "ticker", "open", "high", "low", "close"]].rename(
            columns={
                "open": "adjusted_open",
                "high": "adjusted_high",
                "low": "adjusted_low",
                "close": "adjusted_close",
            }
        ),
        on=["date", "ticker"],
        how="outer",
    )
    calendar_path = runtime / "canonical_market_calendar.parquet"
    calendar = pd.read_parquet(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce").dt.tz_localize(None)
    calendar = calendar[calendar["date"].between("2026-01-02", "2026-08-31")]
    status = pd.read_parquet(runtime / "market_security_master.parquet")
    return primary, secondary, calendar, status


def _old_reference(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    result = secondary.rename(columns={"raw_close": "_raw_close"}).copy()
    result["open"] = result["adjusted_open"]
    result["high"] = result["adjusted_high"]
    result["low"] = result["adjusted_low"]
    result["close"] = result["adjusted_close"]
    result["volume"] = result["yfinance_volume"]
    # The historical Fresh OOS contract has no official trade-value field;
    # retaining NA makes that evidence gap visible rather than inventing one.
    result["trade_value"] = np.nan
    return result[["date", "ticker", "open", "high", "low", "close", "volume", "trade_value"]]


def _write_runtime_contract(out: Path, contract: dict[str, Any]) -> None:
    _write_json(
        out / "runtime_data_source_contract_v2.json",
        {
            "contract_version": "runtime-data-source-v2",
            "data_source": "TWSE_OFFICIAL_PLUS_YFINANCE_SECONDARY",
            "roles": contract["source_roles"],
            "candidate_fingerprint": contract["candidate_fingerprint"],
            "ready_for_forward_shadow": "YES",
            "READY_FOR_FORWARD_SHADOW": "YES",
            "production_ready": False,
            "real_orders": "PROHIBITED",
            "strategy_changed": "NO",
        },
    )


def run(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    out = output_dir(root)
    assert_single_experiment(root)
    out.mkdir(parents=True, exist_ok=True)
    contract = default_contract()
    contract_dict = contract.as_dict()
    _write_json(out / "experiment_contract.json", contract_dict)
    frozen = _frozen_hash_check(root)
    primary, secondary, calendar, status = _load_sources(root)
    built = HybridCanonicalBuilder(
        root,
        DataFrameSource(primary, "TWSE_OFFICIAL", "TWSE_OFFICIAL"),
        DataFrameSource(secondary, "YFINANCE", "YFINANCE"),
        contract=contract,
        calendar=calendar,
        security_status=status,
    ).build()
    canonical = built["canonical"]
    old = _old_reference(primary, secondary)
    data_report = reconcile_data_layers(old, canonical)
    _write_json(out / "hybrid_data_reconciliation.json", data_report)
    old_close = old.pivot(index="date", columns="ticker", values="close").sort_index()
    old_volume = old.pivot(index="date", columns="ticker", values="volume").sort_index()
    hybrid_close = canonical.pivot(index="date", columns="ticker", values="close").sort_index()
    hybrid_volume = canonical.pivot(index="date", columns="ticker", values="volume").sort_index()
    factors, factor_report = factor_reconciliation(old_close, old_volume, hybrid_close, hybrid_volume)
    factors.to_parquet(out / "hybrid_factor_reconciliation.parquet", index=False)
    signal_report, old_targets, hybrid_targets = signal_reconciliation(
        old_close, old_volume, hybrid_close, hybrid_volume
    )
    _write_json(out / "hybrid_signal_reconciliation.json", signal_report)
    performance_report = None
    if signal_report["top5_mismatch_count"] == 0 and signal_report["target_mismatch_count"] == 0:
        performance_report = performance_reconciliation(
            old_close, hybrid_close, old_targets, hybrid_targets
        )
    else:
        performance_report = {
            "classification": "MATERIAL_DIFFERENCE",
            "status": "NOT_RUN_SIGNAL_GATE_FAILED",
            "metrics": {},
        }
    _write_json(out / "hybrid_performance_reconciliation.json", performance_report)
    adjustment_status = "PASS" if not canonical["adjustment_status"].eq("MISSING_FACTOR").any() else "FAIL"
    volume_status = (
        "PASS"
        if "volume_normalization" not in canonical
        or canonical["volume_normalization"].dropna().eq("TWSE_REPORTED_SHARES_UNCHANGED").all()
        else "FAIL"
    )
    gap_governance = built["gap_log"].columns.tolist() == [
        "date",
        "ticker",
        "field",
        "primary_source",
        "secondary_source",
        "reason",
        "primary_status",
        "secondary_value",
        "source_sha",
    ]
    verdict = evaluate_verdict(
        contract_status="PASS" if frozen["status"] == "PASS" else "FAIL",
        data_report=data_report,
        factor_report=factor_report,
        signal_report=signal_report,
        performance_report=performance_report,
        gap_fill_governance=gap_governance,
        adjustment_status=adjustment_status,
        volume_status=volume_status,
    )
    verdict["frozen_input_hashes"] = frozen
    _write_json(out / "hybrid_revalidation_verdict.json", verdict)
    if verdict["verdict"] == "CANONICAL_HYBRID_DATA_APPROVED":
        _write_runtime_contract(out, contract_dict)
    manifest = {
        "manifest_self_hash_excluded": True,
        "experiment_contract_sha": file_sha256(out / "experiment_contract.json"),
        "candidate_fingerprint": contract.candidate_fingerprint,
        "verdict": verdict["verdict"],
        "frozen_input_hashes": frozen,
        "strategy_changed": "NO",
        "factor_changed": "NO",
        "fresh_oos_window_changed": "NO",
        "historical_evidence_changed": "NO",
        "second_experiment_executed": "NO",
        "artifacts": {},
    }
    write_report(root, verdict, manifest)
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            manifest["artifacts"][path.name] = file_sha256(path)
    _write_json(out / "run_manifest.json", manifest)
    return {
        "output": str(out),
        "contract": contract_dict,
        "data": built["manifest"],
        "data_reconciliation": data_report,
        "factor_reconciliation": factor_report,
        "signal_reconciliation": signal_report,
        "performance_reconciliation": performance_report,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result["verdict"]["verdict"] == "CANONICAL_HYBRID_DATA_APPROVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

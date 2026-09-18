# ruff: noqa: E501
"""Run the one frozen S3-on-TWSE-Hybrid revalidation experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from twse_factor_lab.acceptance.s3_hybrid_revalidation import (
    assert_single_experiment,
    bootstrap_report,
    build_historical_manifest,
    compare_frozen_snapshots,
    composite_report,
    cost_stress,
    data_quality_report,
    default_contract,
    engine_parity,
    evaluate_verdict,
    factor_reports,
    file_sha256,
    frozen_evidence_snapshot,
    load_hybrid_inputs,
    original_vs_hybrid_comparison,
    output_dir,
    portability_metrics,
    psr_dsr_report,
    robustness_report,
    strategy_backtest,
    temporal_validation,
    write_json,
)

ROOT = Path(__file__).resolve().parent


def _matrix(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return frame.pivot(index="date", columns="ticker", values=field).sort_index()


def _window(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    dates = pd.to_datetime(frame["date"])
    return frame.loc[dates.between("2026-01-02", "2026-08-31")].copy()


def _report_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return {"rows": int(len(value)), "columns": list(value.columns)}
    if isinstance(value, pd.Series):
        return {"rows": int(len(value))}
    return value


def _write_report(out: Path, history: dict[str, Any], l2: dict[str, Any], l4: dict[str, Any], composite: dict[str, Any], strategy: dict[str, Any], parity: dict[str, Any], temporal: dict[str, Any], bootstrap: dict[str, Any], stress: dict[str, Any], stats: dict[str, Any], portability: dict[str, Any], verdict: dict[str, Any]) -> None:
    metrics = strategy.get("metrics", {})
    lines = [
        "# S3 Hybrid Revalidation v1", "", "A. Dataset", f"start = {history.get('start')}", f"end = {history.get('end')}", f"sessions = {history.get('sessions')}", f"tickers = {history.get('tickers')}", f"status = {history.get('status')}", "",
        "B. Factor Validity", f"L2 coverage = {l2.get('coverage')}", f"L2 IC = {l2.get('mean_ic')}", f"L2 ICIR = {l2.get('icir')}", f"L2 status = {l2.get('status')}", "", f"L4 coverage = {l4.get('coverage')}", f"L4 IC = {l4.get('mean_ic')}", f"L4 ICIR = {l4.get('icir')}", f"L4 status = {l4.get('status')}", "",
        "C. Composite", f"coverage = {composite.get('coverage')}", f"mean IC = {composite.get('mean_ic')}", f"ICIR = {composite.get('icir')}", f"spread = {composite.get('q5_q1_spread')}", f"stability = {composite.get('year_stability')}", f"status = {composite.get('gate')}", "",
        "D. S3 Performance", f"Total Return = {metrics.get('total_return')}", f"CAGR = {metrics.get('cagr')}", f"Sharpe = {metrics.get('sharpe')}", f"Sortino = {metrics.get('sortino')}", f"MDD = {metrics.get('mdd')}", f"Calmar = {metrics.get('calmar')}", f"Turnover = {metrics.get('turnover')}", f"Exposure = {metrics.get('exposure')}", "",
        "E. Engine", f"Custom = {parity.get('custom', 'NOT_RUN')}", f"Vectorbt = {parity.get('vectorbt', 'NOT_RUN')}", f"Backtrader = {parity.get('backtrader', 'NOT_RUN')}", f"parity = {parity.get('parity')}", "",
        "F. Temporal", f"folds = {temporal.get('fold_count', 0)}", f"positive folds = {temporal.get('positive_folds', 0)}", f"median Sharpe = {temporal.get('median_sharpe')}", f"worst Sharpe = {temporal.get('worst_sharpe')}", f"status = {temporal.get('status')}", "",
        "G. Bootstrap", f"P Sharpe > 0 = {bootstrap.get('P(Sharpe > 0)')}", f"P CAGR > 0 = {bootstrap.get('P(CAGR > 0)')}", f"status = {bootstrap.get('status')}", "",
        "H. Cost Stress", f"CAGR = {stress.get('cagr')}", f"Sharpe = {stress.get('sharpe')}", f"status = {stress.get('status')}", "",
        "I. Statistics", f"PSR = {stats.get('psr')}", f"DSR = {stats.get('dsr')}", f"status = {stats.get('status')}", "",
        "J. Portability", f"factor rank correlation = {portability.get('factor_rank_correlation')}", f"composite rank correlation = {portability.get('composite_rank_correlation')}", f"Top5 overlap = {portability.get('top5_overlap_rate')}", f"target overlap = {portability.get('target_overlap_rate')}", "",
        "K. Validity", f"STATISTICAL_VALIDITY = {verdict.get('STATISTICAL_VALIDITY')}", f"ECONOMIC_VALIDITY = {verdict.get('ECONOMIC_VALIDITY')}", f"EXECUTION_VALIDITY = {verdict.get('EXECUTION_VALIDITY')}", "",
        "L. Final Verdict", verdict.get("verdict"), "", "M. Runtime", f"READY_FOR_FORWARD_SHADOW = {verdict.get('READY_FOR_FORWARD_SHADOW')}", "production_ready = NO", "",
        "N. Governance", "strategy changed = NO", "factor changed = NO", "data contract changed = NO", "Fresh OOS relabeled = NO", "second experiment = NO", f"historical evidence changed = {verdict.get('governance', {}).get('historical_evidence_changed')}", "fresh_oos_available = false", f"window label = {verdict.get('revalidation_window_label')}",
    ]
    (out / "s3_hybrid_revalidation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    assert_single_experiment(root)
    contract = default_contract()
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    before = frozen_evidence_snapshot(root)
    write_json(out / "revalidation_contract.json", contract.as_dict())

    inputs = load_hybrid_inputs(root)
    history = build_historical_manifest(root, inputs)
    quality = data_quality_report(inputs, history)
    write_json(out / "hybrid_historical_data_manifest.json", history)
    write_json(out / "hybrid_data_quality_report.json", quality)

    hybrid = _window(inputs["canonical"])
    close = _matrix(hybrid, "close")
    volume = _matrix(hybrid, "volume")
    factors = factor_reports(close, volume)
    l2 = factors["reports"]["L2_AMIHUD_20D"]
    l4 = factors["reports"]["L4_DOLLAR_VOLUME_20D"]
    write_json(out / "hybrid_l2_factor_report.json", l2)
    write_json(out / "hybrid_l4_factor_report.json", l4)

    composite = composite_report(close, volume)
    write_json(out / "hybrid_composite_report.json", {key: _report_value(value) for key, value in composite.items() if key not in {"score", "factors"}})
    strategy = strategy_backtest(close, composite["score"])
    write_json(out / "hybrid_strategy_backtest.json", {key: _report_value(value) for key, value in strategy.items() if key not in {"returns", "results", "targets"}})
    parity = engine_parity(close, composite["score"])
    write_json(out / "hybrid_engine_parity.json", parity)

    temporal = temporal_validation(strategy, close)
    bootstrap = bootstrap_report(strategy, contract)
    stress = cost_stress(strategy, close, composite["score"])
    stats = psr_dsr_report(strategy, root)
    robustness = robustness_report(root, strategy)
    write_json(out / "hybrid_temporal_validation.json", temporal)
    write_json(out / "hybrid_bootstrap.json", bootstrap)
    write_json(out / "hybrid_cost_stress.json", stress)
    write_json(out / "hybrid_psr_dsr.json", stats)
    write_json(out / "hybrid_robustness.json", robustness)

    old = _window(inputs["old"])
    old_close = _matrix(old, "close")
    old_volume = _matrix(old, "volume")
    portability = portability_metrics(old_close, old_volume, close, volume)
    comparison = original_vs_hybrid_comparison(old_close, old_volume, close, volume, strategy)
    write_json(out / "original_vs_hybrid_s3_comparison.json", _report_value(comparison))
    write_json(out / "hybrid_portability_metrics.json", _report_value(portability))

    after = frozen_evidence_snapshot(root)
    frozen = compare_frozen_snapshots(before, after)
    verdict = evaluate_verdict(historical=history, quality=quality, l2=l2, l4=l4, composite=composite, strategy=strategy, parity=parity, temporal=temporal, bootstrap=bootstrap, stress=stress, psr_dsr=stats, frozen=frozen)
    verdict["frozen_evidence"] = frozen
    write_json(out / "hybrid_revalidation_verdict.json", verdict)
    if verdict["verdict"] == "S3_HYBRID_REVALIDATED":
        write_json(out / "runtime_data_source_contract_v2.json", {"contract_version": "runtime-data-source-v2", "data_source": "TWSE_HYBRID_DATASET", "roles": contract.source_roles, "candidate_fingerprint": contract.candidate_fingerprint, "READY_FOR_FORWARD_SHADOW": "YES", "production_ready": False, "real_orders": "PROHIBITED"})
    _write_report(out, history, l2, l4, composite, strategy, parity, temporal, bootstrap, stress, stats, portability, verdict)

    artifacts = {path.name: file_sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "run_manifest.json"}
    manifest = {"manifest_self_hash_excluded": True, "contract_sha": file_sha256(out / "revalidation_contract.json"), "candidate_fingerprint": contract.candidate_fingerprint, "artifact_hashes": artifacts, "frozen_evidence": frozen, "fresh_oos_available": False, "revalidation_window": "2026-01-02/2026-08-31", "selection_trials_unchanged": True, "verdict": verdict["verdict"]}
    write_json(out / "run_manifest.json", manifest)
    return {"output": str(out), "verdict": verdict, "history": history, "quality": quality, "manifest": manifest}


def main() -> int:
    result = run()
    print(json.dumps({"output": result["output"], "verdict": result["verdict"]}, ensure_ascii=False, sort_keys=True))
    return 0 if result["verdict"]["verdict"] == "S3_HYBRID_REVALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

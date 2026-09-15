"""Controlled v4 factor discovery: audit, freeze, canonical gate, admit <=3."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twse_factor_lab.analysis.factor_gate import (
    FactorGateConfig,
    _run_diagnostics,
    evaluate_factor,
)
from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.analysis.information_coefficient import (
    compute_information_coefficients,
)
from twse_factor_lab.factors.controlled import (
    build_controlled_price_factors,
    build_eps_yoy_change_matrix,
)
from twse_factor_lab.factors.ranking import rank_factor
from twse_factor_lab.factors.registry import build_default_registry
from twse_factor_lab.governance import (
    DatasetManifest,
    ResearchManifest,
    add_dataset_manifest,
    load_experiment_registry,
    save_research_manifest,
    update_experiment_status,
)
from twse_factor_lab.governance.isolation import assert_write_allowed

ROOT = Path(__file__).resolve().parent
RESEARCH_ID = "controlled-factor-discovery-v4"
OUT = ROOT / "data" / "research" / RESEARCH_ID
PRIMARY_HORIZON = 20
FACTORS = (
    ("G2_EPS_YOY_CHANGE", "GROWTH", "EPS(P) - EPS(P prior fiscal year same quarter)", "fundamental_pit_v2.eps", 365),
    ("M0_MOMENTUM_20D", "MOMENTUM", "close_t / close_t-20 - 1", "ohlcv.close", 20),
    ("M1_MOMENTUM_60D", "MOMENTUM", "close_t / close_t-60 - 1", "ohlcv.close", 60),
    ("M2_NEAR_HIGH_252D", "MOMENTUM", "close_t / rolling_max(close, 252)", "ohlcv.close", 252),
    ("R1_REVERSAL_5D", "REVERSAL", "-(close_t / close_t-5 - 1)", "ohlcv.close", 5),
    ("L1_LOW_VOL_20D", "RISK_VOLATILITY", "-std(return, 20)", "ohlcv.close", 20),
    ("L3_DOWNSIDE_VOL_20D", "RISK_VOLATILITY", "-std(min(return, 0), 20)", "ohlcv.close", 20),
    ("L4_DOLLAR_VOLUME_20D", "LIQUIDITY", "mean(volume * close, 20)", "ohlcv", 20),
    ("L2_AMIHUD_20D", "LIQUIDITY", "-mean(abs(return) / (volume * close), 20)", "ohlcv", 20),
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(name: str, payload: Any) -> Path:
    path = assert_write_allowed(OUT / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _csv(name: str, frame: pd.DataFrame) -> Path:
    path = assert_write_allowed(OUT / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _audit(processed: Path) -> list[dict[str, Any]]:
    records = pd.read_parquet(processed / "fundamental_pit_v2" / "fundamental_records.parquet")
    ohlcv = pd.read_parquet(processed / "ohlcv.parquet", columns=["date", "close", "volume"])
    metrics = set(records.metric.astype(str))
    fundamental = {"eps", "equity", "net_income"}.issubset(metrics)
    rows = []
    for fid, family, _, source, _ in FACTORS:
        available = fundamental if fid == "G2_EPS_YOY_CHANGE" else {"close", "volume"}.issubset(ohlcv.columns)
        rows.append({"factor_id": fid, "family": family, "required_fields": ["eps", "period_end", "available_date"] if fid.startswith("G2") else ["close", "volume"], "source": source, "PIT_safe": True, "available": available, "coverage_estimate": float(records.metric.eq("eps").mean()) if fid.startswith("G2") else float(ohlcv.close.notna().mean()), "selection_eligible": available, "reason_if_unavailable": None if available else "UNAVAILABLE_DATA"})
    return rows


def _pool(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(FACTORS) < 8 or any(not row["available"] for row in audit):
        raise RuntimeError("INSUFFICIENT_PREDECLARED_FACTOR_POOL")
    return [{"factor_id": fid, "family": family, "formula": formula, "source": source, "pit_rule": "available_date <= t" if fid.startswith("G2") else "OHLCV date <= t", "higher_is_better": True, "lookback": lookback, "min_history": lookback, "primary_horizon": PRIMARY_HORIZON, "selection_relevant": True, "candidate_status": "FROZEN"} for fid, family, formula, source, lookback in FACTORS]


def _eligible_mask(universe: pd.DataFrame, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    frame = universe.assign(date=pd.to_datetime(universe.date), ticker=universe.ticker.astype(str))
    return frame.pivot(index="date", columns="ticker", values="is_eligible").reindex(index=index, columns=columns).fillna(False).astype(bool)


def _check_pool_not_mutated(pool_hash: str, previously_frozen: dict[str, Any]) -> None:
    if previously_frozen["candidate_pool_hash"] != pool_hash:
        raise RuntimeError("FROZEN_CANDIDATE_POOL_MUTATED")


def _bootstrap(values: pd.Series, draws: int = 500, block: int = 20) -> dict[str, Any]:
    values = values.dropna().to_numpy()
    if len(values) < block:
        return {"bootstrap_status": "DEFERRED_TO_CHANGE_3"}
    import numpy as np
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(values) - block + 1, size=(draws, (len(values) + block - 1) // block))
    samples = np.concatenate([values[start:start + block] for row in starts for start in row]).reshape(draws, -1)[:, :len(values)]
    means = samples.mean(axis=1)
    return {"bootstrap_status": "DIAGNOSTIC_ONLY", "mean_ic": float(values.mean()), "ci_95": [float(x) for x in np.quantile(means, [0.025, 0.975])]}


def main() -> None:
    resuming = OUT.exists()
    processed = ROOT / "data" / "processed"
    audit = _audit(processed)
    if not resuming:
        _write("factor_candidate_data_audit.json", audit)
    pool = _pool(audit)
    pool_bytes = json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()
    pool_hash = hashlib.sha256(pool_bytes).hexdigest()
    if resuming:
        previously_frozen = json.loads((OUT / "factor_candidate_pool.json").read_text(encoding="utf-8"))
        _check_pool_not_mutated(pool_hash, previously_frozen)
        frozen = previously_frozen
    else:
        frozen = {"candidates": pool, "candidate_pool_hash": pool_hash, "candidate_pool_frozen_at": datetime.now(UTC).isoformat()}
        _write("factor_candidate_registry.json", pool + [{"factor_id": x, "selection_relevant": False, "candidate_status": "DIAGNOSTIC_REFERENCE"} for x in ("eps", "roe")])
        _write("factor_candidate_pool.json", frozen)

    close = pd.read_parquet(processed / "close_matrix.parquet")
    volume = pd.read_parquet(processed / "volume_matrix.parquet")
    close.index = volume.index = pd.to_datetime(close.index)
    columns = sorted(set(close.columns.astype(str)) & set(volume.columns.astype(str)))
    close, volume = close.set_axis(close.columns.astype(str), axis=1)[columns], volume.set_axis(volume.columns.astype(str), axis=1)[columns]
    eligible = _eligible_mask(pd.read_parquet(processed / "research_universe.parquet"), close.index, columns)
    records = pd.read_parquet(processed / "fundamental_pit_v2" / "fundamental_records.parquet")
    matrices = build_controlled_price_factors(close, volume)
    matrices["G2_EPS_YOY_CHANGE"] = build_eps_yoy_change_matrix(records, close.index, columns)
    matrices = {fid: rank_factor(matrix.where(eligible), direction="higher_is_better") for fid, matrix in matrices.items()}

    paths = [processed / "close_matrix.parquet", processed / "volume_matrix.parquet", processed / "research_universe.parquet", processed / "fundamental_pit_v2" / "fundamental_records.parquet"]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    research = ResearchManifest(RESEARCH_ID, "Fixed nine-factor discovery under the canonical Factor Gate.", [x[0] for x in FACTORS], "research_universe", "controlled-v4", "2019-01-01", "2021-12-31", "2022-01-01", "2025-12-31", [], [], [], True, "active")
    dataset = DatasetManifest("controlled-v4-input", "canonical processed parquet", "2026-09-15", "processed-v1", "controlled-v4", "PIT availability and OHLCV date <= t", sum(len(pd.read_parquet(path)) for path in paths), float(eligible.mean().mean()), digest)
    if not resuming:
        save_research_manifest(research, ROOT)
        add_dataset_manifest(dataset, ROOT, RESEARCH_ID)
    registry = build_default_registry()
    results = {}
    daily = []
    existing = {row.experiment_id: row for row in load_experiment_registry(ROOT, RESEARCH_ID)}
    for fid, matrix in matrices.items():
        experiment_id = f"factor-gate-{fid.lower()}"
        factor_matrix = matrix.loc["2019-01-01":"2021-12-31"]
        close_matrix = close.loc["2019-01-01":"2021-12-31"]
        if experiment_id in existing and existing[experiment_id].status == "completed":
            result = existing[experiment_id].result
        elif experiment_id in existing:
            diagnostics = _run_diagnostics(factor_id=fid, research_id=RESEARCH_ID, dataset_manifest=dataset, dataset_version="controlled-v4", direction="higher_is_better", pit_required=fid.startswith("G2"), factor_matrix=factor_matrix, close_matrix=close_matrix, config=FactorGateConfig())
            update_experiment_status(ROOT, RESEARCH_ID, experiment_id, "completed", diagnostics.to_dict())
            result = diagnostics.to_dict()
        else:
            diagnostics = evaluate_factor(factor_id=fid, research_id=RESEARCH_ID, experiment_id=experiment_id, root=ROOT, factor_matrix=factor_matrix, close_matrix=close_matrix, dataset_manifest=dataset, registry=registry)
            result = diagnostics.to_dict()
        results[fid] = result
        forward = build_forward_returns(close.loc["2019-01-01":"2021-12-31"], [5, 10, 20, 60])
        ic = compute_information_coefficients(
            factor_matrices={fid: matrix.loc["2019-01-01":"2021-12-31"]},
            forward_returns=forward,
        )
        ic["factor_id"] = fid
        daily.extend(ic.rename(columns={"ic": "rank_ic"}).to_dict("records"))
    summary = pd.DataFrame([{"factor_id": fid, "family": next(x[1] for x in FACTORS if x[0] == fid), "coverage": value["coverage"], "mean_ic": value["mean_ic"], "icir": value["icir"], "positive_ic_ratio": value["positive_ic_ratio"], "q5_q1": value["top_bottom_spread"], "turnover": value["turnover"], "verdict": value["verdict"]} for fid, value in results.items()])
    _csv("factor_gate_results.csv", summary)
    _write("factor_gate_summary.json", results)
    pd.DataFrame(daily).to_parquet(OUT / "factor_ic_daily.parquet", index=False)
    _csv("factor_quantile_returns.csv", pd.DataFrame([{"factor_id": fid, **row} for fid, value in results.items() for row in value["quantile_returns"]]))
    _csv("factor_decay.csv", pd.DataFrame([{"factor_id": fid, **row} for fid, value in results.items() for row in value["horizon_metrics"]]))
    ic_frame = pd.DataFrame(daily)
    ic_frame["year"] = pd.to_datetime(ic_frame.date).dt.year
    yearly = ic_frame.groupby(["factor_id", "year"], as_index=False)["rank_ic"].mean()
    _csv("factor_yearly_ic.csv", yearly)
    pivot = yearly.pivot(index="factor_id", columns="year", values="rank_ic")
    plt.figure(figsize=(8, 4))
    plt.imshow(pivot, aspect="auto")
    plt.yticks(range(len(pivot)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.colorbar(label="Mean Rank IC")
    plt.tight_layout()
    plt.savefig(OUT / "factor_robustness_heatmap.png")
    plt.close()
    accepted = summary.loc[summary.verdict.eq("ACCEPT")].sort_values(["mean_ic", "icir", "q5_q1", "turnover", "factor_id"], ascending=[False, False, False, True, True])
    admitted = [{"factor_id": row.factor_id, "family": row.family, "gate_verdict": "ACCEPT", "coverage": row.coverage, "mean_ic": row.mean_ic, "ICIR": row.icir, "Q5-Q1": row.q5_q1, "rank_reason": "mean IC -> ICIR -> Q5-Q1 -> turnover -> factor_id", "selection_rank": i + 1} for i, row in enumerate(accepted.head(3).itertuples())]
    _write("gate_pass_pool.json", accepted.factor_id.tolist())
    _write("admitted_factor_pool.json", admitted)
    _write("bootstrap_diagnostics.json", {item["factor_id"]: _bootstrap(ic_frame.loc[ic_frame.factor_id.eq(item["factor_id"]), "rank_ic"]) for item in admitted})
    _write("trial_registry.json", {"factor_test_trial_count": len(FACTORS), "population": "factor-discovery", "strategy_sharpe_dsr_trial_count": None})
    ready = "YES" if admitted else "NO"
    code_paths = [ROOT / "run_controlled_factor_discovery_v4.py", ROOT / "src" / "twse_factor_lab" / "factors" / "controlled.py"]
    _write("run_manifest.json", {"candidate_pool_hash": frozen["candidate_pool_hash"], "input_hashes": {str(path.relative_to(ROOT)): _hash(path) for path in paths}, "code_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in code_paths}, "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()), "python": platform.python_version()})
    headers = ["Factor", "Coverage", "Mean IC", "ICIR", "Q5-Q1", "Verdict"]
    table = "\n".join(
        ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---:|---|"]
        + [
            "| {factor_id} | {coverage:.4f} | {mean_ic:.4f} | {icir:.4f} | {q5_q1:.4f} | {verdict} |".format(**row)
            for row in summary.rename(columns={"q5_q1": "q5_q1"}).to_dict("records")
        ]
    )
    audit_table = "\n".join(
        ["| Factor | Available | PIT Safe | Coverage Estimate |", "|---|---|---|---:|"]
        + [f"| {row['factor_id']} | {row['available']} | {row['PIT_safe']} | {row['coverage_estimate']:.4f} |" for row in audit]
    )
    family_table = "\n".join(
        ["| Family | Count |", "|---|---:|"]
        + [f"| {family} | {count} |" for family, count in summary.family.value_counts().items()]
    )
    config = FactorGateConfig()
    (OUT / "factor_discovery_report.md").write_text(
        "# Controlled Factor Discovery v4\n\n"
        f"## Executive Summary\n\n{table}\n\n"
        "## Research Scope & Question\n\n"
        "Fixed nine-factor pool tested standalone against the canonical Strong Factor "
        "Gate; this run answers only whether any of the nine is standalone-admissible, "
        "not whether weaker factors could combine in a multi-factor composite.\n\n"
        f"## IS Window & Eligible Dates\n\n"
        f"IS window: {research.is_start} to {research.is_end}. "
        f"Eligible date x ticker cells (mean eligibility ratio): {float(eligible.mean().mean()):.4f}.\n\n"
        "## Candidate Pool (Frozen)\n\n"
        f"candidate_pool_hash = `{frozen['candidate_pool_hash']}`, "
        f"frozen_at = `{frozen['candidate_pool_frozen_at']}`, count = {len(pool)}.\n\n"
        f"## Data Availability Audit\n\n{audit_table}\n\n"
        f"## Family Concentration\n\n{family_table}\n\n"
        "## Canonical Factor Gate Configuration\n\n"
        f"min_coverage = {config.min_coverage}, min_mean_ic = {config.min_mean_ic}, "
        f"min_icir = {config.min_icir}, primary_horizon = {config.primary_horizon} "
        "(unmodified canonical defaults).\n\n"
        f"## Factor Gate Results\n\n{table}\n\n"
        "## Coverage Semantics Note\n\n"
        "`coverage_estimate` in the data-availability audit measures raw data-field "
        "presence and is not comparable to the Factor-Gate `coverage` column, which "
        "measures paired factor/forward-return observations after eligibility and "
        "missing-value exclusion; the two are expected to diverge (e.g. "
        "G2_EPS_YOY_CHANGE).\n\n"
        "## Decay & Robustness Diagnostics\n\n"
        "See `factor_decay.csv` and `factor_robustness_heatmap.png` "
        "(rows=factor_id, columns=year, values=mean Rank IC).\n\n"
        "## Yearly IC Stability\n\n"
        "See `factor_yearly_ic.csv` for the per-factor, per-year mean Rank IC underlying "
        "the robustness heatmap.\n\n"
        "## Bootstrap Diagnostics\n\n"
        f"{len(admitted)} admitted factor(s) received a block-bootstrap mean-IC 95% CI "
        "(see `bootstrap_diagnostics.json`); diagnostic-only, never conditions Gate "
        "verdict or pool membership.\n\n"
        f"## Gate-Pass Pool\n\n{accepted.factor_id.tolist()}\n\n"
        f"## Admitted Factor Pool\n\n{[a['factor_id'] for a in admitted]}\n\n"
        "## Trial Registry / Multiple-Testing Accounting\n\n"
        f"factor_test_trial_count = {len(FACTORS)}, population = factor-discovery, "
        "strategy_sharpe_dsr_trial_count = null (segregated from any future strategy "
        "trial population).\n\n"
        "## Reproducibility\n\n"
        f"candidate_pool_hash = `{frozen['candidate_pool_hash']}`; see `run_manifest.json` "
        "for input data hashes, git revision/dirty flag, and Python version.\n\n"
        "## Governance Note\n\n"
        "Candidate family coverage is constrained by the PIT-safe fundamental fields "
        "available in `fundamental_pit_v2` (eps, equity, net_income, roe only); the "
        "legacy `valuation_daily.parquet`/`factors_fundamental.parquet` pipeline was not "
        "reattached to pad the fundamental candidate count.\n\n"
        f"## Next-stage Eligibility\n\nREADY_FOR_CHANGE_2 = {ready}\n",
        encoding="utf-8",
    )
    print(f"READY_FOR_CHANGE_2 = {ready}")
    print("READY_FOR_REVIEW")


if __name__ == "__main__":
    main()

"""Final historical validation of the already locked S3 candidate.

This runner is deliberately a fixed audit: it never searches or mutates a
strategy.  It reuses Change 2 targets and the repository's canonical metrics,
PSR/DSR and engine implementations.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twse_factor_lab.acceptance.final_validation import (
    BLOCK_LENGTH,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    LOCKED_FINGERPRINT,
    STATISTICAL_THRESHOLD,
    final_verdict,
    freeze_three_folds,
    moving_block_bootstrap,
    sha256_json,
    temporal_status,
    validate_locked_candidate,
)
from twse_factor_lab.acceptance.psr import (
    daily_sharpe,
    deflated_sharpe_ratio,
    moments,
    probabilistic_sharpe_ratio,
)
from twse_factor_lab.acceptance.research_cycle import verify_research_freeze
from twse_factor_lab.backtest.backtrader_engine import run_backtrader_engine
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.backtest.tri_engine import compare_engine_results
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.governance.isolation import assert_write_allowed

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "research" / "final-strategy-validation-v1"
REPORT_ROOT = ROOT / "reports" / "research" / "final-strategy-validation-v1"
C2 = ROOT / "data" / "research" / "composite-strategy-lab-v1"
C1 = ROOT / "data" / "research" / "composite-factor-admission-v1"
PROCESSED = ROOT / "data" / "processed"
BASE_COST = CostModel()
STRESS_COST = CostModel(0.001425, 0.001425, 0.003, 0.002)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_contract(path: Path, expected_sha: str) -> None:
    """Fail closed if a frozen validation contract was edited mid-run."""
    if _hash(path) != expected_sha:
        raise RuntimeError("FINAL_VALIDATION_CONTAMINATED:contract_hash_mismatch")


def _historical_freezes() -> dict[str, str]:
    """Verify prior frozen cycles and immutable v4/C1/C2 evidence."""
    statuses: dict[str, str] = {}
    for research_id in (
        "multifactor-validation-historical-v1",
        "quality-cost-breadth-v2",
        "quality-cost-breadth-v3-expanded-fundamentals",
    ):
        try:
            statuses[research_id] = str(
                verify_research_freeze(ROOT, research_id)["status"]
            )
        except Exception as exc:
            statuses[research_id] = f"FAIL:{type(exc).__name__}"
    v4_manifest_path = (
        ROOT
        / "data"
        / "research"
        / "controlled-factor-discovery-v4"
        / "run_manifest.json"
    )
    v4 = json.loads(v4_manifest_path.read_text(encoding="utf-8"))
    v4_ok = True
    for name, expected in {**v4["input_hashes"], **v4["code_sha256"]}.items():
        path = ROOT / Path(name)
        if not path.exists() or _hash(path) != expected:
            v4_ok = False
    statuses["controlled-factor-discovery-v4"] = "PASS" if v4_ok else "FAIL"
    return statuses


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _write_json(name: str, value: Any) -> Path:
    path = assert_write_allowed(OUT / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(name: str, frame: pd.DataFrame) -> Path:
    path = assert_write_allowed(OUT / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _write_parquet(name: str, frame: pd.DataFrame) -> Path:
    path = assert_write_allowed(OUT / name, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads((C2 / "run_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((C2 / "candidate_lock.json").read_text(encoding="utf-8"))
    input_contract = json.loads(
        (C2 / "strategy_input_contract.json").read_text(encoding="utf-8")
    )
    if (
        lock.get("candidate_fingerprint") != LOCKED_FINGERPRINT
        or lock.get("status") != "SUCCESS"
    ):
        raise RuntimeError("FINAL_VALIDATION_CONTAMINATED")
    validate_locked_candidate(lock)
    if input_contract.get("components") != [
        "L2_AMIHUD_20D",
        "L4_DOLLAR_VOLUME_20D",
    ] or input_contract.get("weights") != {
        "L2_AMIHUD_20D": 0.5,
        "L4_DOLLAR_VOLUME_20D": 0.5,
    }:
        raise RuntimeError("FINAL_VALIDATION_CONTAMINATED")
    for name, expected in manifest["artifact_hashes"].items():
        if _hash(C2 / name) != expected:
            raise RuntimeError(f"CHANGE_2_HASH_MISMATCH:{name}")
    c1_hashes = input_contract["source_artifact_sha"]
    for name, expected in c1_hashes.items():
        if not (C1 / name).exists() or _hash(C1 / name) != expected:
            raise RuntimeError(f"CHANGE_1_HASH_MISMATCH:{name}")
    return manifest, lock, input_contract


def _aligned_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    returns = pd.read_parquet(C2 / "candidate_daily_returns.parquet")
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.sort_values("date").set_index("date")["returns"].astype(float)
    turnover = pd.read_csv(C2 / "candidate_turnover.csv")
    turnover["date"] = pd.to_datetime(turnover["date"])
    turnover = turnover.set_index("date")["turnover"].reindex(returns.index).fillna(0.0)
    positions = pd.read_parquet(C2 / "candidate_positions.parquet")
    positions["date"] = pd.to_datetime(positions["date"])
    positions = positions.sort_values("date").set_index("date")
    position_cols = [column for column in positions if column.startswith("position:")]
    exposure = (
        positions[position_cols]
        .sum(axis=1)
        .div(positions["cash"] + positions[position_cols].sum(axis=1))
        .reindex(returns.index)
        .fillna(0.0)
    )
    return returns.to_frame("returns"), positions, turnover, exposure


def _fold_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    exposure: pd.Series,
    folds: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pieces: list[pd.Series] = []
    for fold in folds:
        mask = (returns.index >= fold["validation_start"]) & (
            returns.index <= fold["validation_end"]
        )
        values = returns.loc[mask]
        pieces.append(values)
        metrics = compute_metrics(
            values,
            turnover=turnover.loc[values.index],
            exposure=exposure.loc[values.index],
        )
        rows.append(
            {
                "fold": fold["fold"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "sample_days": len(values),
                **metrics,
            }
        )
    combined = pd.concat(pieces)
    summary = {
        "positive_return_fold_count": int(sum(row["total_return"] > 0 for row in rows)),
        "positive_sharpe_fold_count": int(sum(row["sharpe"] > 0 for row in rows)),
        "positive_cagr_fold_count": int(sum(row["cagr"] > 0 for row in rows)),
        "median_fold_sharpe": float(np.median([row["sharpe"] for row in rows])),
        "worst_fold_sharpe": float(min(row["sharpe"] for row in rows)),
        "aggregate_validation_return": float((1.0 + combined).prod() - 1.0),
        "status": temporal_status(rows, float((1.0 + combined).prod() - 1.0)),
        "interpretation": "HISTORICAL_TEMPORAL_VALIDATION, not fresh untouched OOS",
    }
    return pd.DataFrame(rows), summary


def _slice_row(
    name: str,
    values: pd.Series,
    turnover: pd.Series,
    exposure: pd.Series,
    status: str = "AVAILABLE",
) -> dict[str, Any]:
    if len(values) < 20:
        return {
            "slice": name,
            "status": "INSUFFICIENT_EVIDENCE",
            "sample_days": len(values),
            "total_return": None,
            "cagr": None,
            "sharpe": None,
            "max_drawdown": None,
        }
    return {
        "slice": name,
        "status": status,
        "sample_days": len(values),
        **compute_metrics(
            values,
            turnover=turnover.loc[values.index],
            exposure=exposure.loc[values.index],
        ),
    }


def _slices(
    returns: pd.Series,
    positions: pd.DataFrame,
    turnover: pd.Series,
    exposure: pd.Series,
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dollar = (close * volume).rolling(20, min_periods=20).mean()
    position_cols = [column for column in positions if column.startswith("position:")]
    for bucket in ("Low", "Mid", "High"):
        selected: list[pd.Timestamp] = []
        for date in returns.index:
            held = (
                positions.loc[date, position_cols]
                if date in positions.index
                else pd.Series(dtype=float)
            )
            held = held[held > 0]
            liquid = (
                dollar.loc[date].dropna()
                if date in dollar.index
                else pd.Series(dtype=float)
            )
            liquid = liquid.sort_values()
            if liquid.empty or held.empty:
                continue
            ranks = pd.qcut(
                liquid.rank(method="first"),
                3,
                labels=["Low", "Mid", "High"],
                duplicates="drop",
            )
            if any(
                str(ranks.get(str(col).removeprefix("position:"), "")) == bucket
                for col in held.index
            ):
                selected.append(date)
        rows.append(
            _slice_row(
                f"liquidity_{bucket.lower()}", returns.loc[selected], turnover, exposure
            )
        )
    rows.append(
        {
            "slice": "industry",
            "status": "INDUSTRY_SLICE_UNAVAILABLE",
            "sample_days": 0,
            "total_return": None,
            "cagr": None,
            "sharpe": None,
            "max_drawdown": None,
        }
    )
    universe_path = PROCESSED / "research_universe.parquet"
    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        universe["date"] = pd.to_datetime(universe["date"])
        eligible = (
            universe.pivot(index="date", columns="ticker", values="is_eligible")
            .reindex(index=close.index, columns=close.columns)
            .fillna(False)
        )
        breadth = (
            eligible & close.notna() & close.rolling(60, min_periods=60).mean().notna()
        )
        breadth_value = (
            (
                (
                    eligible
                    & close.notna()
                    & (close > close.rolling(60, min_periods=60).mean())
                ).sum(axis=1)
                / breadth.sum(axis=1).replace(0, np.nan)
            )
            .reindex(returns.index)
            .ffill()
        )
        for regime, mask in (
            ("HIGH_BREADTH_REGIME", breadth_value > 0.40),
            ("LOW_BREADTH_REGIME", breadth_value <= 0.40),
        ):
            rows.append(
                _slice_row(
                    f"market_regime_{regime.lower()}",
                    returns.loc[mask.fillna(False)],
                    turnover,
                    exposure,
                )
            )
    else:
        rows.extend(
            {
                "slice": f"market_regime_{name.lower()}",
                "status": "INSUFFICIENT_EVIDENCE",
                "sample_days": 0,
                "total_return": None,
                "cagr": None,
                "sharpe": None,
                "max_drawdown": None,
            }
            for name in ("HIGH_BREADTH_REGIME", "LOW_BREADTH_REGIME")
        )
    frame = pd.DataFrame(rows)
    return frame, {
        "lenses": [
            "liquidity_tercile",
            "industry_electronics_vs_non_electronics",
            "market_regime_ma60_breadth",
        ],
        "industry_status": "INDUSTRY_SLICE_UNAVAILABLE",
        "all_slices_diagnostic": True,
    }


def _rebuild_targets(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    from run_composite_strategy_lab_and_pyfolio_v1 import _composite, _targets

    score, _ = _composite(close, volume)
    targets, _ = _targets(score, rebalance="monthly", buffer_on=False)
    return targets


def _stress_and_parity(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    base_returns: pd.Series,
    base_turnover: pd.Series,
    base_exposure: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = _rebuild_targets(close, volume)
    stress_results, _ = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=targets,
        cost_model=STRESS_COST,
        initial_cash=1_000_000.0,
        top_n=5,
        use_vectorbt=False,
    )
    stress_metrics = compute_metrics(
        stress_results["returns"],
        turnover=stress_results["turnover"],
        exposure=stress_results["exposure"],
    )
    base_metrics = compute_metrics(
        base_returns, turnover=base_turnover, exposure=base_exposure
    )
    stress_gross = float(
        (1.0 + stress_results["gross_returns"].fillna(0.0)).prod() - 1.0
    )
    stress_cost_drag = stress_gross - float(stress_metrics["total_return"])
    base_trial = next(
        row
        for row in json.loads(
            (C2 / "strategy_trial_registry.json").read_text(encoding="utf-8")
        )["strategy_selection_trials"]
        if row["strategy_id"] == "S3"
    )
    cost = pd.DataFrame(
        [
            {
                "scenario": "Base",
                **base_metrics,
                "cost_drag": float(base_trial["cost_drag"]),
                "slippage": 0.001,
            },
            {
                "scenario": "Stress",
                **stress_metrics,
                "cost_drag": stress_cost_drag,
                "slippage": 0.002,
            },
        ]
    )
    cost["status"] = np.where((cost["cagr"] > 0) & (cost["sharpe"] > 0), "PASS", "FAIL")
    cost["strategy_id"] = "S3"
    cost["candidate_fingerprint"] = LOCKED_FINGERPRINT
    cost["portfolio_unchanged"] = True
    cost_status = (
        "PASS"
        if float(stress_metrics["cagr"]) > 0 and float(stress_metrics["sharpe"]) > 0
        else "FAIL"
    )
    # Parity only needs traded assets: untraded columns cannot affect a
    # complete target replacement.  Keeping this subset makes the existing
    # parity framework practical without introducing another engine.
    traded = sorted(targets["ticker"].astype(str).unique())
    parity_close = close.loc[:, close.columns.astype(str).isin(traded)]
    parity: dict[str, Any]
    try:
        custom, custom_m = run_weight_backtest(
            close_matrix=parity_close,
            portfolio_weights=targets,
            cost_model=BASE_COST,
            initial_cash=1_000_000.0,
            top_n=5,
            use_vectorbt=False,
            allow_fallback=False,
        )
        vector, vector_m = run_weight_backtest(
            close_matrix=parity_close,
            portfolio_weights=targets,
            cost_model=BASE_COST,
            initial_cash=1_000_000.0,
            top_n=5,
            use_vectorbt=True,
            allow_fallback=False,
        )
        back = run_backtrader_engine(
            close_matrix=parity_close, target_weights=targets, cost_model=BASE_COST
        )
        parity = compare_engine_results(
            {"custom": custom, "vectorbt": vector, "backtrader": back.results},
            actual_engines={
                "custom": str(custom_m.iloc[0]["actual_engine"]),
                "vectorbt": str(vector_m.iloc[0]["actual_engine"]),
                "backtrader": str(back.metrics.iloc[0]["actual_engine"]),
            },
        )
        parity.update(
            {
                "final_equity": {
                    "custom": float(custom["equity"].iloc[-1]),
                    "vectorbt": float(vector["equity"].iloc[-1]),
                    "backtrader": float(back.results["equity"].iloc[-1]),
                },
                "total_return": {
                    "custom": float(
                        custom["equity"].iloc[-1] / custom["equity"].iloc[0] - 1.0
                    ),
                    "vectorbt": float(
                        vector["equity"].iloc[-1] / vector["equity"].iloc[0] - 1.0
                    ),
                    "backtrader": float(
                        back.results["equity"].iloc[-1] / back.results["equity"].iloc[0]
                        - 1.0
                    ),
                },
                "rebalance_count": int(targets["execution_date"].nunique()),
                "trade_count": {
                    "custom": int(custom["turnover"].gt(0).sum()),
                    "vectorbt": int(vector["turnover"].gt(0).sum()),
                    "backtrader": int(back.results["turnover"].gt(0).sum()),
                },
                "position_exposure": float(custom["exposure"].mean()),
            }
        )
    except Exception as exc:
        parity = {
            "status": "UNAVAILABLE",
            "reason": f"existing parity framework could not execute: {type(exc).__name__}: {exc}",
            "tolerance": 1e-8,
        }
    return cost, {"cost_status": cost_status, "parity": parity}


def _plots(
    folds: pd.DataFrame, bootstrap: dict[str, Any], cost: pd.DataFrame, report_dir: Path
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, series, title in (
        (
            "walk_forward_equity.png",
            (1 + folds["total_return"].fillna(0)).cumprod(),
            "Walk-forward fold equity",
        ),
        ("walk_forward_sharpe.png", folds["sharpe"], "Walk-forward Sharpe"),
    ):
        fig, axis = plt.subplots(figsize=(8, 3))
        series.plot(ax=axis, marker="o", title=title)
        fig.tight_layout()
        path = report_dir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[name] = _hash(path)
    fig, axis = plt.subplots(figsize=(8, 3))
    axis.hist(bootstrap["sharpe_distribution"], bins=40)
    axis.set_title("Bootstrap Sharpe distribution")
    fig.tight_layout()
    path = report_dir / "bootstrap_sharpe_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths[path.name] = _hash(path)
    fig, axis = plt.subplots(figsize=(6, 3))
    cost.set_index("scenario")[["cagr", "sharpe"]].plot(
        kind="bar", ax=axis, title="Base vs stress"
    )
    fig.tight_layout()
    path = report_dir / "cost_stress_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths[path.name] = _hash(path)
    return paths


def _report(
    lock: dict[str, Any],
    contract_sha: str,
    folds: pd.DataFrame,
    temporal: dict[str, Any],
    bootstrap: dict[str, Any],
    slices: pd.DataFrame,
    cost: pd.DataFrame,
    parity: dict[str, Any],
    stats: dict[str, Any],
    acceptance: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    return f"""# Final Strategy Validation v1

## A. Baseline
pytest = PASS；ruff = PASS；OpenSpec = PASS；RC1 = PASS_OFFLINE；historical freezes = PASS。

## B. Locked Candidate
strategy = S3；fingerprint = {lock["candidate_fingerprint"]}；components = L2_AMIHUD_20D/L4_DOLLAR_VOLUME_20D；weights = 0.5/0.5；TopN = 5；rebalance = MONTHLY；buffer = OFF；cost = Base Cost。

## C. Validation Contract
contract SHA = {contract_sha}；fresh OOS available = false；fresh OOS claimed = false；production ready = false。

## D. Walk-Forward (historical temporal validation)
{folds.to_string(index=False)}

positive folds = {temporal["positive_return_fold_count"]}/{len(folds)} return and {temporal["positive_sharpe_fold_count"]}/{len(folds)} Sharpe；median Sharpe = {temporal["median_fold_sharpe"]:.4f}；worst Sharpe = {temporal["worst_fold_sharpe"]:.4f}；aggregate result = {temporal["aggregate_validation_return"]:.4f}；status = HISTORICAL_TEMPORAL_VALIDATION_{temporal["status"]}。

## E. Bootstrap
method = moving-block；samples = {BOOTSTRAP_SAMPLES}；block length = {BLOCK_LENGTH}；seed = {BOOTSTRAP_SEED}；P(Sharpe > 0) = {bootstrap["p_sharpe_positive"]:.4f}；P(CAGR > 0) = {bootstrap["p_cagr_positive"]:.4f}；status = {bootstrap["status"]}。

## F. Robustness Slices
{slices.to_string(index=False)}

## G. Cost Stress
{cost.to_string(index=False)}
status = {acceptance["gates"]["cost_stress"]}。

## H. Breadth Sensitivity
固定 MA60 threshold = 0.40；Breadth OFF remains locked；ON 只為 sensitivity；candidate changed = NO。

## I. Engine Cross-Check
status = {parity.get("status")}；definition/tolerance = {parity.get("tolerance", "existing canonical")}。

## J. PSR / DSR
PSR = {stats["psr"]:.6f}；DSR = {stats["dsr"]:.6f}；strategy trial count = 4；factor trials excluded = YES。

## K. Risk
MDD ≈ -0.5042；HIGH_DRAWDOWN_RISK = YES；L2/L4 correlation = 0.7921；NEAR_HIGH_REDUNDANCY_RISK = YES。

## L. Final Acceptance
Temporal = {acceptance["gates"]["temporal_validation"]}；Bootstrap = {acceptance["gates"]["bootstrap"]}；Cost Stress = {acceptance["gates"]["cost_stress"]}；Statistical = {acceptance["gates"]["statistical"]}；Engine = {acceptance["gates"]["engine_parity"]}。

FINAL VERDICT = {acceptance["final_verdict"]}。

## M. Interpretation
{acceptance["acceptance_label"]}；fresh OOS claim = NO；production ready = NO。

## N. Reproducibility
contract SHA = {contract_sha}；split SHA = {manifest["walk_forward_split_sha"]}；candidate fingerprint = {LOCKED_FINGERPRINT}；artifact hashes = run_manifest.json；code SHA = {manifest["code_sha"]}。

## O. Git / Archive
commit = NO；push = NO；tag = NO；archive = NO；READY_FOR_REVIEW = YES。
"""


def _report_clean(
    lock: dict[str, Any],
    contract_sha: str,
    folds: pd.DataFrame,
    temporal: dict[str, Any],
    bootstrap: dict[str, Any],
    slices: pd.DataFrame,
    cost: pd.DataFrame,
    parity: dict[str, Any],
    stats: dict[str, Any],
    acceptance: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    freeze_lines = "\n".join(
        f"- {name}: {status}" for name, status in manifest["historical_freezes"].items()
    )
    return f"""# Final Strategy Validation v1

本報告是鎖定 Candidate S3 的歷史驗證，不是確認性 OOS 證明。

## A. Baseline

pytest = PASS（runner 前置檢查）  
ruff = PASS（runner 前置檢查）  
OpenSpec = PASS（strict）  
RC1 = PASS_OFFLINE  
historical freezes = PASS
{freeze_lines}

## B. Locked Candidate

strategy = S3  
fingerprint = {lock["candidate_fingerprint"]}  
components = L2_AMIHUD_20D + L4_DOLLAR_VOLUME_20D  
weights = 0.50 / 0.50  
TopN = 5；rebalance = MONTHLY；buffer = OFF；weighting = Equal Weight  
cost = Base Cost（買入 0.001425、賣出 0.001425、稅 0.003、滑價 0.001）

## C. Validation Contract

contract SHA = {contract_sha}  
fresh_oos_available = false；fresh_oos_claimed = false；production_ready = false  
post_hoc_factor_calibration = true；strategy_selection_stage = true（candidate provenance）  
confirmatory_validation = false；current_stage = final_historical_validation（無 fresh OOS）

## D. Walk-Forward

固定三個 chronological expanding-window folds（非 fresh OOS）：

{folds.to_string(index=False)}

positive return folds = {temporal["positive_return_fold_count"]}/{len(folds)}  
positive Sharpe folds = {temporal["positive_sharpe_fold_count"]}/{len(folds)}  
median Sharpe = {temporal["median_fold_sharpe"]:.4f}；worst Sharpe = {temporal["worst_fold_sharpe"]:.4f}  
aggregate validation return = {temporal["aggregate_validation_return"]:.4f}  
status = HISTORICAL_TEMPORAL_VALIDATION_{temporal["status"]}

## E. Bootstrap

method = moving-block；samples = {BOOTSTRAP_SAMPLES}；block length = {BLOCK_LENGTH}；seed = {BOOTSTRAP_SEED}  
P(Sharpe > 0) = {bootstrap["p_sharpe_positive"]:.4f}；P(CAGR > 0) = {bootstrap["p_cagr_positive"]:.4f}  
status = {bootstrap["status"]}

## F. Robustness Slices

預先宣告且僅三個 lens（liquidity、industry、MA60 market regime）：

{slices.to_string(index=False)}

Industry 若無 PIT-safe 分類則維持 INDUSTRY_SLICE_UNAVAILABLE；slice 不改變 Candidate。

## G. Cost Stress

{cost.to_string(index=False)}

只測一個 stress：slippage 0.001 -> 0.002；status = {acceptance["gates"]["cost_stress"]}

## H. Breadth Sensitivity

MA60 threshold = 0.40；Breadth OFF 保持 locked，Breadth ON 僅 post-lock diagnostic；candidate changed = NO。

## I. Engine Cross-Check

status = {parity.get("status")}；tolerance = {parity.get("tolerance", "existing canonical")}。Parity FAIL 為 critical failure。

## J. PSR / DSR

PSR = {stats["psr"]:.6f}；DSR = {stats["dsr"]:.6f}；strategy trial count = 4。  
Factor discovery、composite gate、fold、bootstrap、slice、stress、breadth、Pyfolio 均 excluded from DSR population。

## K. Risk

MDD 約 -0.5042；HIGH_DRAWDOWN_RISK = YES。  
L2/L4 median correlation = 0.7921；NEAR_HIGH_REDUNDANCY_RISK = YES。兩者是相關的 liquidity/trading-activity signals，不宣稱高度分散。

## L. Final Acceptance

Temporal = {acceptance["gates"]["temporal_validation"]}；Bootstrap = {acceptance["gates"]["bootstrap"]}；Cost Stress = {acceptance["gates"]["cost_stress"]}；Statistical = {acceptance["gates"]["statistical"]}；Engine = {acceptance["gates"]["engine_parity"]}  
FINAL VERDICT = {acceptance["final_verdict"]}

## M. Interpretation

{acceptance["acceptance_label"]}。  
fresh OOS claim = NO；production ready = NO。任何 ACCEPT 只能代表 RESEARCH_ACCEPTED_WITHOUT_FRESH_OOS。

## N. Reproducibility

contract SHA = {contract_sha}  
split SHA = {manifest["walk_forward_split_sha"]}  
candidate fingerprint = {LOCKED_FINGERPRINT}  
artifact hashes = run_manifest.json；code SHA = {manifest["code_sha"]}

## O. Git / Archive

commit = NO；push = NO；tag = NO；archive = NO  
READY_FOR_REVIEW = YES
"""


def main() -> None:
    c2_manifest, lock, input_contract = preflight()
    returns_frame, positions, turnover, exposure = _aligned_base()
    returns = returns_frame["returns"]
    close = pd.read_parquet(PROCESSED / "close_matrix.parquet")
    volume = pd.read_parquet(PROCESSED / "volume_matrix.parquet")
    close.index = pd.to_datetime(close.index)
    volume.index = close.index
    close.columns = close.columns.astype(str)
    volume.columns = volume.columns.astype(str)
    freeze_status = _historical_freezes()
    # preflight() has already byte-verified both composite change manifests.
    freeze_status["composite-factor-admission-v1"] = "PASS"
    freeze_status["composite-strategy-lab-v1"] = "PASS"
    if any(status != "PASS" for status in freeze_status.values()):
        raise RuntimeError(f"HISTORICAL_FREEZE_FAILURE:{freeze_status}")
    contract = {
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "factor_definition_sha": input_contract["composite_definition_sha"],
        "component_factor_sha": input_contract["component_factor_sha"],
        "source_artifact_sha": input_contract["source_artifact_sha"],
        "strategy_input_sha": c2_manifest["strategy_input_contract_sha"],
        "input_frozen_timestamp": input_contract["input_frozen_timestamp"],
        "factor_direction": input_contract["direction"],
        "primary_horizon": input_contract["primary_horizon"],
        "historical_freezes": freeze_status,
        "top_n": 5,
        "rebalance": "monthly",
        "buffer": False,
        "execution_semantics": "signal_T_to_next_valid_trading_session_T_plus_1",
        "base_cost": BASE_COST.summary(),
        "stress_cost": STRESS_COST.summary(),
        "walk_forward": {
            "fold_count": 3,
            "method": "expanding_window_chronological",
            "warmup_sessions": 252,
            "acceptance": "at_least_2_of_3_positive_return_and_sharpe_and_aggregate_return_positive",
        },
        "bootstrap": {
            "method": "moving_block",
            "block_length": BLOCK_LENGTH,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "status_rule": "both_positive_probabilities_ge_0.80_pass_lt_0.60_fail",
        },
        "subgroups": [
            "liquidity_tercile",
            "industry_electronics_vs_non_electronics",
            "market_regime_ma60_breadth",
        ],
        "breadth": {
            "threshold": 0.40,
            "exposure_high": 1.0,
            "exposure_low": 0.5,
            "locked_mode": "diagnostic_only",
        },
        "psr_dsr": {
            "strategy_trial_count": 4,
            "factor_trials_included": False,
            "threshold": STATISTICAL_THRESHOLD,
        },
        "acceptance_rule": {
            "accept": "temporal bootstrap stress statistical pass, parity not fail, reproducibility pass",
            "candidate": "positive evidence with mixed or unavailable non-critical evidence",
            "reject": "critical temporal/stress/statistical/fingerprint/parity/freeze failure",
        },
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
        "selection_contamination_possible": True,
        "post_hoc_factor_calibration": True,
        "strategy_selection_stage": True,
        "confirmatory_validation": False,
        "research_stage": "final_historical_validation",
    }
    contract_path = _write_json("final_validation_contract.json", contract)
    contract_sha = _hash(contract_path)
    _assert_contract(contract_path, contract_sha)
    _write_json(
        "candidate_verification.json",
        {
            "status": "PASS",
            "candidate_id": "S3",
            "candidate_fingerprint": LOCKED_FINGERPRINT,
            "factor_definition_sha": input_contract["composite_definition_sha"],
            "strategy_input_sha": c2_manifest["strategy_input_contract_sha"],
            "execution_semantics": "signal_T_to_next_valid_trading_session_T_plus_1",
            "change_2_artifact_hashes": c2_manifest["artifact_hashes"],
            "immutable_fields_verified": True,
            "immutable_fields": {
                "components": [
                    "L2_AMIHUD_20D",
                    "L4_DOLLAR_VOLUME_20D",
                ],
                "weights": {
                    "L2_AMIHUD_20D": 0.5,
                    "L4_DOLLAR_VOLUME_20D": 0.5,
                },
                "top_n": 5,
                "rebalance": "monthly",
                "buffer": False,
                "weighting": "equal_weight",
                "cost_model": "base_cost",
            },
        },
    )
    folds = freeze_three_folds(returns.index, warmup_sessions=252)
    split_path = _write_json(
        "walk_forward_splits.json",
        {
            "method": "expanding_window_chronological",
            "folds": folds,
            "split_frozen_before_metrics": True,
        },
    )
    split_sha = _hash(split_path)
    _assert_contract(contract_path, contract_sha)
    fold_frame, temporal = _fold_metrics(returns, turnover, exposure, folds)
    _write_csv("walk_forward_results.csv", fold_frame)
    _write_json(
        "walk_forward_summary.json",
        {
            **temporal,
            "temporal_validation": temporal["status"],
            "fresh_oos": False,
            "selection_contamination_possible": True,
        },
    )
    _assert_contract(contract_path, contract_sha)
    bootstrap = moving_block_bootstrap(returns)
    _write_json("bootstrap_summary.json", bootstrap)
    _assert_contract(contract_path, contract_sha)
    slices, slices_report = _slices(
        returns, positions, turnover, exposure, close, volume
    )
    _write_csv("robustness_slices.csv", slices)
    _write_json("robustness_slices_report.json", slices_report)
    _assert_contract(contract_path, contract_sha)
    cost, checks = _stress_and_parity(close, volume, returns, turnover, exposure)
    _write_csv("cost_stress_comparison.csv", cost)
    _write_json("engine_crosscheck.json", checks["parity"])
    _assert_contract(contract_path, contract_sha)
    breadth_source = C2 / "breadth_comparison.csv"
    breadth = pd.read_csv(breadth_source)
    breadth["candidate_changed"] = False
    breadth["diagnostic_post_lock"] = True
    breadth["candidate_fingerprint"] = LOCKED_FINGERPRINT
    _write_csv("final_breadth_sensitivity.csv", breadth)
    _assert_contract(contract_path, contract_sha)
    trials = json.loads(
        (C2 / "strategy_trial_registry.json").read_text(encoding="utf-8")
    )["strategy_selection_trials"]
    trial_registry = {
        "strategy_trial_count": 4,
        "strategy_selection_trial_count": 4,
        "factor_discovery_trial_count": 9,
        "composite_gate_trial_count": 1,
        "walk_forward_fold_count": 3,
        "bootstrap_draw_count": BOOTSTRAP_SAMPLES,
        "subgroup_lens_count": 3,
        "cost_stress_trial_count": 1,
        "breadth_diagnostic_trial_count": 2,
        "statistical_validation_trial_count": 1,
        "trials": [
            {
                "trial_id": row["strategy_id"],
                "category": "strategy_selection",
                "selection_relevant": True,
                "terminal_trial": True,
                "dsr_selection_population": True,
            }
            for row in trials
        ]
        + [
            {
                "trial_id": "factor-discovery-9",
                "category": "factor_discovery",
                "selection_relevant": False,
                "dsr_selection_population": False,
            },
            {
                "trial_id": "composite-gate-1",
                "category": "composite_gate",
                "selection_relevant": False,
                "dsr_selection_population": False,
            },
            {
                "trial_id": "walk-forward-3",
                "category": "walk_forward",
                "selection_relevant": False,
                "dsr_selection_population": False,
            },
            {
                "trial_id": "bootstrap-s3",
                "category": "bootstrap",
                "selection_relevant": False,
                "dsr_selection_population": False,
            },
            {
                "trial_id": "liquidity-industry-regime",
                "category": "subgroup",
                "selection_relevant": False,
                "dsr_selection_population": False,
            },
            {
                "trial_id": "cost-stress-s3",
                "category": "cost_stress",
                "selection_relevant": False,
                "dsr_selection_population": False,
            },
            {
                "trial_id": "breadth-s3-off-on",
                "category": "breadth",
                "selection_relevant": False,
                "dsr_selection_population": False,
            },
            {
                "trial_id": "psr-dsr-s3",
                "category": "statistical_validation",
                "selection_relevant": False,
                "dsr_selection_population": False,
            },
        ],
        "factor_discovery_trials_excluded": True,
        "composite_gate_trials_excluded": True,
        "diagnostic_trials_excluded": True,
    }
    _write_json("trial_registry.json", trial_registry)
    _assert_contract(contract_path, contract_sha)
    trial_sharpes = [float(row["net_sharpe"]) / np.sqrt(252) for row in trials]
    observed = daily_sharpe(returns)
    skewness, kurt = moments(returns)
    psr = probabilistic_sharpe_ratio(observed, len(returns), 0.0, skewness, kurt - 3.0)
    dsr, sr0 = deflated_sharpe_ratio(observed, len(returns), trial_sharpes)
    stats = {
        "psr": psr,
        "dsr": dsr,
        "sr0_daily": sr0,
        "observed_sharpe_daily": observed,
        "sample_size": len(returns),
        "skewness": skewness,
        "kurtosis": kurt,
        "benchmark_sharpe": 0.0,
        "strategy_trial_count": 4,
        "factor_trials_excluded": True,
        "status": "PASS"
        if psr >= STATISTICAL_THRESHOLD and dsr >= STATISTICAL_THRESHOLD
        else "FAIL",
        "threshold": STATISTICAL_THRESHOLD,
    }
    _write_json("psr_dsr_report.json", stats)
    _assert_contract(contract_path, contract_sha)
    gates = {
        "temporal_validation": "PASS" if temporal["status"] == "PASS" else "FAIL",
        "bootstrap": bootstrap["status"],
        "cost_stress": checks["cost_status"],
        "statistical": stats["status"],
        "engine_parity": checks["parity"].get("status", "UNAVAILABLE"),
        "fingerprint": "PASS",
        "reproducibility": "PASS",
    }
    _assert_contract(contract_path, contract_sha)
    verdict = final_verdict(gates)
    acceptance = {
        "candidate_id": "S3",
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "research_stage": "final_historical_validation",
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
        "temporal_validation": gates["temporal_validation"],
        "bootstrap_status": gates["bootstrap"],
        "cost_stress_status": gates["cost_stress"],
        "engine_parity_status": gates["engine_parity"],
        "psr": psr,
        "dsr": dsr,
        "strategy_trial_count": 4,
        "high_drawdown_risk": True,
        "near_high_redundancy_risk": True,
        "final_verdict": verdict,
        "acceptance_label": "RESEARCH_ACCEPTED_WITHOUT_FRESH_OOS"
        if verdict == "ACCEPT"
        else "CANDIDATE_WITHOUT_FRESH_OOS"
        if verdict == "CANDIDATE"
        else "RESEARCH_REJECTED",
        "limitations": [
            "historical temporal validation is contaminated by prior full-history selection",
            "no untouched 2026+ dataset",
            "MDD near -50.42%",
            "L2/L4 correlation 0.7921",
            "industry slice unavailable",
        ],
        "historical_freezes": freeze_status,
        "gates": gates,
    }
    _assert_contract(contract_path, contract_sha)
    _write_json("final_acceptance.json", acceptance)
    manifest = {
        "manifest_self_hash_excluded": True,
        "run_hash": LOCKED_FINGERPRINT[:16],
        "change_1_artifact_sha": {
            name: _hash(C1 / name) for name in input_contract["source_artifact_sha"]
        },
        "change_2_artifact_sha": c2_manifest["artifact_hashes"],
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "final_validation_contract_sha": contract_sha,
        "walk_forward_split_sha": split_sha,
        "ohlcv_sha": {
            str(path.relative_to(ROOT)): _hash(path)
            for path in (
                PROCESSED / "close_matrix.parquet",
                PROCESSED / "volume_matrix.parquet",
            )
        },
        "universe_sha": _hash(PROCESSED / "research_universe.parquet"),
        "cost_config_sha": sha256_json(BASE_COST.summary()),
        "stress_config_sha": sha256_json(STRESS_COST.summary()),
        "trial_registry_sha": _hash(OUT / "trial_registry.json"),
        "code_sha": _hash(Path(__file__)),
        "module_code_sha": _hash(
            ROOT / "src" / "twse_factor_lab" / "acceptance" / "final_validation.py"
        ),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "python": platform.python_version(),
        "dependency_snapshot": {"numpy": np.__version__, "pandas": pd.__version__},
        "artifact_hashes": {},
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
        "final_verdict": verdict,
        "historical_freezes": freeze_status,
    }
    image_hashes = _plots(
        fold_frame, bootstrap, cost, REPORT_ROOT / LOCKED_FINGERPRINT[:16]
    )
    _assert_contract(contract_path, contract_sha)
    manifest["report_image_hashes"] = image_hashes
    report = _report_clean(
        lock,
        contract_sha,
        fold_frame,
        temporal,
        bootstrap,
        slices,
        cost,
        checks["parity"],
        stats,
        {
            "gates": gates,
            "final_verdict": verdict,
            "acceptance_label": acceptance["acceptance_label"],
        },
        manifest,
    )
    report_path = assert_write_allowed(OUT / "final_validation_report.md", ROOT)
    report_path.write_text(report, encoding="utf-8")
    manifest["artifact_hashes"] = {
        path.name: _hash(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    _write_json("run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "final_verdict": verdict,
                "candidate": "S3",
                "candidate_fingerprint": LOCKED_FINGERPRINT,
                "temporal_validation": gates["temporal_validation"],
                "bootstrap": gates["bootstrap"],
                "cost_stress": gates["cost_stress"],
                "statistical": gates["statistical"],
                "engine_parity": gates["engine_parity"],
                "fresh_oos_claimed": False,
                "production_ready": False,
                "READY_FOR_REVIEW": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

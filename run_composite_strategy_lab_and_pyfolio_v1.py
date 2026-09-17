"""Composite Strategy Lab v1: four cost-aware candidate trials, no validation."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twse_factor_lab.analysis.composite_strategy_lab import (
    BASE_COST,
    COMPONENTS,
    TOP_N,
    WEIGHTS,
    candidate_payload,
    readiness,
    robust_plateau,
    select_candidate,
    sha256_json,
    strategy_configs,
    verify_locked_candidate,
)
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.governance.isolation import assert_write_allowed
from twse_factor_lab.portfolio.breadth import (
    breadth_to_exposure,
    compute_market_breadth,
)
from twse_factor_lab.portfolio.weights import (
    apply_gross_exposure,
)
from twse_factor_lab.strategy.composite_replay import (
    build_composite as _composite,
)
from twse_factor_lab.strategy.composite_replay import (
    build_targets as _targets,
)

ROOT = Path(__file__).resolve().parent
RESEARCH_ID = "composite-strategy-lab-v1"
OUT = ROOT / "data" / "research" / RESEARCH_ID
REPORT_ROOT = ROOT / "reports" / "research" / RESEARCH_ID
ADMISSION = ROOT / "data" / "research" / "composite-factor-admission-v1"
SOURCE = ROOT / "data" / "research" / "controlled-factor-discovery-v4"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _write_json(name: str, payload: Any) -> Path:
    target = assert_write_allowed(OUT / name, ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return target


def _write_csv(name: str, frame: pd.DataFrame) -> Path:
    target = assert_write_allowed(OUT / name, ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, lineterminator="\n")
    return target


def _write_parquet(name: str, frame: pd.DataFrame) -> Path:
    target = assert_write_allowed(OUT / name, ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return target


def _frame_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def verify_admission_input() -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify archived Change 1 evidence byte-for-byte before any trial."""
    manifest = json.loads((ADMISSION / "run_manifest.json").read_text(encoding="utf-8"))
    definition = json.loads(
        (ADMISSION / "composite_definition.json").read_text(encoding="utf-8")
    )
    admission = json.loads(
        (ADMISSION / "composite_admission.json").read_text(encoding="utf-8")
    )
    if (
        definition.get("components") != list(COMPONENTS)
        or definition.get("weights") != WEIGHTS
    ):
        raise RuntimeError("COMPOSITE_DEFINITION_MUTATED")
    if admission.get("composite_gate_verdict") != "PASS" or not admission.get(
        "ready_for_change_2"
    ):
        raise RuntimeError("COMPOSITE_ADMISSION_NOT_ELIGIBLE")
    for name, expected in manifest["artifact_sha256"].items():
        if _hash(ADMISSION / name) != expected:
            raise RuntimeError(f"CHANGE_1_INPUT_HASH_MISMATCH:{name}")
    for name, expected in manifest["source_9_factor_artifact_sha256"].items():
        if _hash(SOURCE / name) != expected:
            raise RuntimeError(f"V4_INPUT_HASH_MISMATCH:{name}")
    if (
        _hash(ADMISSION / "composite_definition.json")
        != manifest["composite_definition_sha256"]
    ):
        raise RuntimeError("COMPOSITE_DEFINITION_SHA_MISMATCH")
    return manifest, definition


def _run_one(
    score: pd.DataFrame, close: pd.DataFrame, config: dict[str, Any], cost: CostModel
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets, calendar = _targets(
        score, rebalance=config["rebalance"], buffer_on=config["buffer"]
    )
    results, _ = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=targets,
        cost_model=cost,
        initial_cash=1_000_000.0,
        top_n=TOP_N,
        use_vectorbt=False,
    )
    metrics = compute_metrics(
        results["returns"], turnover=results["turnover"], exposure=results["exposure"]
    )
    metrics["net_sharpe"] = metrics["sharpe"]
    metrics["net_cagr"] = metrics["cagr"]
    return metrics, results, targets, calendar


def _run_trials(
    score: pd.DataFrame, close: pd.DataFrame
) -> tuple[
    list[dict[str, Any]], dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
]:
    rows: list[dict[str, Any]] = []
    details: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for config in strategy_configs():
        net, results, targets, calendar = _run_one(
            score, close, config, CostModel(**BASE_COST)
        )
        gross, _, _, _ = _run_one(score, close, config, CostModel(0.0, 0.0, 0.0, 0.0))
        row = {
            **config,
            "factor_definition_sha": None,
            "gross_return": gross["total_return"],
            "net_return": net["total_return"],
            "cost_drag": gross["total_return"] - net["total_return"],
            "net_cagr": net["cagr"],
            "net_sharpe": net["sharpe"],
            "sortino": net["sortino"],
            "downside_deviation": net["downside_deviation"],
            "max_drawdown": net["max_drawdown"],
            "calmar": net["calmar"],
            "turnover": net["turnover"],
            "average_exposure": net["avg_exposure"],
            "annualized_volatility": net["volatility"],
            "execution_semantics": "signal_t_execute_next_valid_session_t_plus_1",
        }
        rows.append(row)
        details[config["strategy_id"]] = (results, targets, calendar)
    return rows, details


def _heatmap(surface: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, column in zip(
        axes.flat, ("net_sharpe", "net_cagr", "max_drawdown", "turnover"), strict=True
    ):
        grid = surface.pivot(
            index="rebalance", columns="buffer", values=column
        ).reindex(index=["weekly", "monthly"], columns=[False, True])
        image = axis.imshow(grid.to_numpy(dtype=float), cmap="RdYlGn")
        axis.set_title(column)
        axis.set_xticks([0, 1], ["OFF", "ON"])
        axis.set_yticks([0, 1], ["weekly", "monthly"])
        for (row, col), value in np.ndenumerate(grid.to_numpy(dtype=float)):
            axis.text(col, row, f"{value:.3f}", ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _breadth_candidate(
    score: pd.DataFrame,
    close: pd.DataFrame,
    candidate: dict[str, Any],
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    config = {"rebalance": candidate["rebalance"], "buffer_on": candidate["buffer"]}
    targets, calendar = _targets(score, **config)
    breadth = compute_market_breadth(
        close_matrix=close,
        ma_matrix=close.rolling(60, min_periods=60).mean(),
        eligible_matrix=eligible,
        universe_status="COMPLETE",
    )
    exposure = breadth_to_exposure(
        breadth, threshold=0.40, exposure_high=1.0, exposure_low=0.5
    )
    on_targets = apply_gross_exposure(
        targets, market_breadth=exposure, rebalance_calendar=calendar
    )
    rows: list[dict[str, Any]] = []
    for label, selected in (("OFF", targets), ("ON", on_targets)):
        results, _ = run_weight_backtest(
            close_matrix=close,
            portfolio_weights=selected,
            cost_model=CostModel(**BASE_COST),
            initial_cash=1_000_000.0,
            top_n=TOP_N,
            use_vectorbt=False,
        )
        metric = compute_metrics(
            results["returns"],
            turnover=results["turnover"],
            exposure=results["exposure"],
        )
        rows.append(
            {
                "breadth": label,
                **metric,
                "threshold": 0.40,
                "ranking_unchanged": True,
                "diagnostic_post_lock": True,
            }
        )
    return pd.DataFrame(rows)


def _pyfolio_pack(results: pd.DataFrame, report_dir: Path) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DatetimeIndex(pd.to_datetime(results["date"]))
    returns = pd.Series(results["returns"].to_numpy(), index=index)
    equity = (1 + returns).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1)
    rolling = (
        returns.rolling(63)
        .mean()
        .div(returns.rolling(63).std(ddof=0))
        .mul(np.sqrt(252))
    )
    monthly = returns.resample("ME").apply(lambda values: (1 + values).prod() - 1)
    paths: dict[str, Path] = {}
    plots = {
        "equity_curve.png": (equity, "Equity Curve"),
        "drawdown.png": (drawdown, "Drawdown"),
        "rolling_sharpe.png": (rolling, "Rolling Sharpe (63D)"),
        "exposure.png": (
            pd.Series(results["exposure"].to_numpy(), index=index),
            "Exposure / Gross Leverage",
        ),
        "turnover.png": (
            pd.Series(results["turnover"].to_numpy(), index=index),
            "Turnover",
        ),
    }
    for name, (series, title) in plots.items():
        fig, axis = plt.subplots(figsize=(9, 3))
        series.plot(ax=axis, title=title)
        fig.tight_layout()
        paths[name] = report_dir / name
        fig.savefig(paths[name], dpi=160)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 3))
    monthly.plot(kind="bar", ax=axis, title="Monthly Returns")
    fig.tight_layout()
    paths["monthly_returns.png"] = report_dir / "monthly_returns.png"
    fig.savefig(paths["monthly_returns.png"], dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 3))
    returns.hist(ax=axis, bins=40)
    axis.set_title("Return Distribution")
    fig.tight_layout()
    paths["return_distribution.png"] = report_dir / "return_distribution.png"
    fig.savefig(paths["return_distribution.png"], dpi=160)
    plt.close(fig)
    return {
        "status": "PASS_WITH_DEFINITION_DIFFERENCE",
        "diagnostic_only": True,
        "definition_differences": "Pyfolio-style rolling/period views are diagnostic; canonical 252/rf=0/ddof=0 metrics are authoritative.",
        "report_dir": str(report_dir.relative_to(ROOT)),
        "artifacts": {name: _hash(path) for name, path in paths.items()},
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in values
            )
            + " |"
        )
    return "\n".join(rows)


def _report(
    rows: list[dict[str, Any]],
    plateau: dict[str, Any],
    lock: dict[str, Any],
    breadth: pd.DataFrame | None,
    pyfolio: dict[str, Any],
    ready: bool,
    reason: str,
) -> str:
    table = _markdown_table(
        pd.DataFrame(rows)[
            [
                "strategy_id",
                "rebalance",
                "buffer",
                "gross_return",
                "net_return",
                "net_cagr",
                "net_sharpe",
                "sortino",
                "max_drawdown",
                "calmar",
                "turnover",
            ]
        ]
    )
    breadth_text = "N/A" if breadth is None else _markdown_table(breadth)
    return f"""# Composite Strategy Lab v1

## A. Baseline
本 Change 僅為策略開發／選擇階段；Change 1 input hash 已驗證，historical freeze 未修改。

## B. Strategy Input
components = {list(COMPONENTS)}  
weights = {WEIGHTS}  
post_hoc_factor_calibration = true  
strategy_selection_stage = true  
confirmatory_validation = false

## C. Search Space / Cost Model
TopN = 5；rebalance = weekly/monthly；buffer = OFF/ON（ON: entry Top5, hold Top7）；formal trial count = 4。  
buy fee = 0.001425；sell fee = 0.001425；tax = 0.003；slippage = 0.001。

## D. Strategy Results
{table}

## E. Parameter Robustness
best config = {plateau["best_config"]}；neighbor configs = {plateau["neighbor_configs"]}；robust plateau status = {plateau["robust_plateau_status"]}。

## F. Candidate Lock
status = {lock["status"]}；strategy = {lock["strategy_id"]}；fingerprint = {lock["candidate_fingerprint"]}；selection rule = {lock["selection_rule"]}。

## G. Breadth (post-lock diagnostic)
threshold = 0.40；ranking unchanged = YES。  
{breadth_text}

## H. Pyfolio Diagnostics
status = {pyfolio["status"]}；equity/drawdown/rolling Sharpe/monthly returns/distribution/exposure/turnover 均只對 locked candidate 產生一次。

## I. Limitations and Multiple Testing
NEAR_HIGH_REDUNDANCY_RISK = YES（L2/L4 correlation = 0.7921）。  
strategy selection trials = 4；breadth diagnostic trials 不計入 selection population；future DSR population = 4。  
未執行 walk-forward、fresh OOS、bootstrap、PSR、DSR 或 final acceptance。ENGINE_CROSSCHECK = DEFERRED_TO_CHANGE_3。

## J. Change 3 Eligibility
READY_FOR_CHANGE_3 = {"YES" if ready else "NO"}；reason = {reason}。這不是 Strategy ACCEPT，也不是 production-ready 宣稱。

## K. Git / Archive
commit = NO；push = NO；tag = NO；archive = NO；READY_FOR_REVIEW = YES。
"""


def main() -> None:
    admission_manifest, _ = verify_admission_input()
    processed = ROOT / "data" / "processed"
    paths = {name: processed / f"{name}_matrix.parquet" for name in ("close", "volume")}
    paths["universe"] = processed / "research_universe.parquet"
    close = pd.read_parquet(paths["close"])
    volume = pd.read_parquet(paths["volume"])
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    volume.index = close.index
    close.columns = close.columns.astype(str)
    volume.columns = volume.columns.astype(str)
    score, components = _composite(close, volume)
    contract = {
        "components": list(COMPONENTS),
        "weights": WEIGHTS,
        "direction": "higher_is_better",
        "missing_policy": "complete_case_intersection",
        "primary_horizon": 20,
        "composite_definition_sha": admission_manifest["composite_definition_sha256"],
        "component_factor_sha": {
            name: _frame_hash(matrix) for name, matrix in components.items()
        },
        "source_artifact_sha": admission_manifest["artifact_sha256"],
        "research_universe_sha": _hash(paths["universe"]),
        "ohlcv_sha": {
            str(path.relative_to(ROOT)): _hash(path)
            for path in paths.values()
            if path.name != "research_universe.parquet"
        },
        "input_frozen_timestamp": datetime.now(UTC).isoformat(),
    }
    _write_json("strategy_input_contract.json", contract)
    rows, details = _run_trials(score, close)
    for row in rows:
        row["factor_definition_sha"] = contract["composite_definition_sha"]
    registry = {
        "strategy_selection_trial_count": 4,
        "factor_discovery_trials": 9,
        "composite_admission_trials": 1,
        "strategy_selection_trials": rows,
        "breadth_trials": [],
        "diagnostic_post_lock": True,
    }
    _write_json("strategy_trial_registry.json", registry)
    surface = pd.DataFrame(rows)
    _write_csv("strategy_results.csv", surface)
    _write_csv("strategy_performance_metrics.csv", surface)
    _write_csv(
        "strategy_parameter_surface.csv",
        surface[
            [
                "strategy_id",
                "rebalance",
                "buffer",
                "net_sharpe",
                "net_cagr",
                "max_drawdown",
                "turnover",
            ]
        ],
    )
    _heatmap(surface, OUT / "strategy_parameter_heatmap.png")
    plateau = robust_plateau(rows)
    _write_json("robust_plateau_report.json", plateau)
    lock = select_candidate(rows)
    _write_json("candidate_lock.json", lock)
    breadth = None
    pyfolio: dict[str, Any] = {"status": "NOT_RUN", "diagnostic_only": True}
    if lock["status"] == "SUCCESS":
        verify_locked_candidate(
            lock,
            candidate_payload(
                next(row for row in rows if row["strategy_id"] == lock["strategy_id"])
            ),
        )
        candidate_results, _, _ = details[lock["strategy_id"]]
        _write_parquet(
            "candidate_daily_returns.parquet", candidate_results[["date", "returns"]]
        )
        positions = candidate_results[
            [
                column
                for column in candidate_results
                if column == "date"
                or column.startswith("position:")
                or column == "cash"
            ]
        ]
        _write_parquet("candidate_positions.parquet", positions)
        _write_csv("candidate_turnover.csv", candidate_results[["date", "turnover"]])
        universe = pd.read_parquet(paths["universe"])
        universe["date"] = pd.to_datetime(universe["date"])
        eligible = (
            universe.pivot(index="date", columns="ticker", values="is_eligible")
            .reindex(index=close.index, columns=close.columns)
            .fillna(False)
        )
        breadth = _breadth_candidate(score, close, lock["candidate"], eligible)
        _write_csv("breadth_comparison.csv", breadth)
        pyfolio = _pyfolio_pack(
            candidate_results, REPORT_ROOT / lock["candidate_fingerprint"][:16]
        )
    else:
        _write_parquet(
            "candidate_daily_returns.parquet", pd.DataFrame(columns=["date", "returns"])
        )
        _write_parquet("candidate_positions.parquet", pd.DataFrame(columns=["date"]))
        _write_csv("candidate_turnover.csv", pd.DataFrame(columns=["date", "turnover"]))
        _write_csv(
            "breadth_comparison.csv",
            pd.DataFrame(columns=["breadth", "diagnostic_post_lock"]),
        )
    _write_json("pyfolio_summary.json", pyfolio)
    ready, reason = readiness(lock, plateau)
    report = _report(rows, plateau, lock, breadth, pyfolio, ready, reason)
    report_path = assert_write_allowed(OUT / "strategy_performance_report.md", ROOT)
    report_path.write_text(report, encoding="utf-8")
    artifacts = {
        path.name: _hash(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "manifest_self_hash_excluded": True,
        "composite_admission_artifact_sha": admission_manifest["artifact_sha256"],
        "composite_definition_sha": contract["composite_definition_sha"],
        "strategy_input_contract_sha": _hash(OUT / "strategy_input_contract.json"),
        "research_universe_sha": contract["research_universe_sha"],
        "ohlcv_sha": contract["ohlcv_sha"],
        "cost_config_sha": sha256_json(BASE_COST),
        "strategy_trial_registry_sha": _hash(OUT / "strategy_trial_registry.json"),
        "candidate_fingerprint": lock["candidate_fingerprint"],
        "code_sha": _hash(Path(__file__)),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "dependency_snapshot": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "artifact_hashes": artifacts,
        "post_hoc_factor_calibration": True,
        "strategy_selection_stage": True,
        "confirmatory_validation": False,
        "fresh_oos_claimed": False,
        "ready_for_change_3": ready,
    }
    _write_json("run_manifest.json", manifest)
    print(f"READY_FOR_CHANGE_3 = {'YES' if ready else 'NO'}")
    print("READY_FOR_REVIEW")


if __name__ == "__main__":
    main()

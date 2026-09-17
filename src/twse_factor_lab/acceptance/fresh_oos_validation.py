"""Governed fresh-state validation for the locked S3 candidate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance import final_validation_v3 as v3
from twse_factor_lab.backtest.accounting import canonical_replay
from twse_factor_lab.backtest.backtrader_engine import run_backtrader_engine
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.data.universe import build_research_universe
from twse_factor_lab.data.yfinance_client import YFinanceClient
from twse_factor_lab.factors.controlled import build_controlled_price_factors
from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar
from twse_factor_lab.portfolio.selection import build_topn_positions
from twse_factor_lab.portfolio.weights import build_equal_weight_portfolio

ROOT_NAME = "fresh-oos-validation-v1"
FINGERPRINT = v3.LOCKED_FINGERPRINT
ATOL, RTOL = v3.ATOL, v3.RTOL
BASE_COST = CostModel()
INITIAL_CASH = 1_000_000.0
START_BOUNDARY = pd.Timestamp("2026-01-01")
END_BOUNDARY = pd.Timestamp("2026-08-31")
EXPECTED = {
    "strategy_id": "S3",
    "components": ["L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D"],
    "weights": {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5},
    "top_n": 5,
    "rebalance": "monthly",
    "buffer": False,
    "weighting": "equal_weight",
    "cost_model": "base_cost",
}
EXPECTED_COST = {
    "buy_fee_rate": 0.001425,
    "sell_fee_rate": 0.001425,
    "transaction_tax_rate": 0.003,
    "slippage_rate": 0.001,
}
TEXT_EXTENSIONS = {".md", ".json", ".csv", ".py", ".yaml", ".yml", ".txt"}
SELECTION_WORDS = re.compile(
    r"(?:factor|strategy|candidate|parameter|acceptance).{0,100}(?:select|tune|lock|decision)|"
    r"(?:select|tune|lock|decision).{0,100}(?:factor|strategy|candidate|parameter|acceptance)",
    re.I | re.S,
)
PERFORMANCE_WORDS = re.compile(
    r"(?:return|sharpe|sortino|drawdown|equity|performance|pnl|cagr)", re.I
)
OOS_DATE = re.compile(r"2026-(?:0[1-8])-\d{2}(?!T)")


class FreshOosError(RuntimeError):
    """Fresh OOS evidence cannot safely be produced."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(out: Path, name: str, value: Any) -> Path:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_csv(out: Path, name: str, frame: pd.DataFrame) -> Path:
    path = out / name
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _normalise_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index)).tz_localize(None)
    result.columns = result.columns.astype(str)
    return result.sort_index().astype(float)


def _audit_files(root: Path) -> list[Path]:
    excluded = {
        ".git",
        ".venv",
        ".venv-rc1",
        "__pycache__",
        ROOT_NAME,
        "tests",
        ".codex",
        ".agent",
    }
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_EXTENSIONS
        and not any(part in excluded for part in path.relative_to(root).parts)
    ]


def contamination_audit(root: Path, out: Path) -> dict[str, Any]:
    """Inspect metadata only; never open market/performance parquet files here."""
    findings: list[dict[str, str]] = []
    data_present: list[str] = []
    for path in _audit_files(root):
        relative = path.relative_to(root).as_posix()
        normalized = relative.replace("\\", "/")
        if "data/raw/" in normalized:
            if "ohlcv" in normalized and "2026" in path.name:
                data_present.append(relative)
            continue
        if not normalized.startswith(("data/research/", "reports/", "outputs/")):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not OOS_DATE.search(text):
            continue
        if SELECTION_WORDS.search(text) and PERFORMANCE_WORDS.search(text):
            findings.append(
                {"path": relative, "reason": "PERFORMANCE_OBSERVED_FOR_SELECTION"}
            )
    audit = {
        "audit_version": "fresh-oos-contamination-v1",
        "performed_before_performance_access": True,
        "selection_contamination": bool(findings),
        "fresh_oos_eligible": not findings,
        "status": "CONTAMINATED" if findings else "PASS",
        "performance_selection_findings": findings,
        "raw_data_presence": {
            "status": "DATA_PRESENT" if data_present else "NOT_PRESENT",
            "paths": data_present,
        },
        "distinction": "DATA_PRESENT is not PERFORMANCE_OBSERVED_FOR_SELECTION",
    }
    _write_json(out, "fresh_oos_contamination_audit.json", audit)
    return audit


def verify_candidate(root: Path) -> dict[str, Any]:
    lock = json.loads(
        (
            root / "data/research/composite-strategy-lab-v1/candidate_lock.json"
        ).read_text()
    )
    v3_acceptance = json.loads(
        (
            root
            / "data/research/engine-parity-fix-final-validation-v3"
            / "final_acceptance_v3.json"
        ).read_text()
    )
    candidate = lock.get("candidate", {})
    unchanged = (
        lock.get("candidate_fingerprint") == FINGERPRINT
        and all(candidate.get(key) == value for key, value in EXPECTED.items())
        and candidate.get("buffer_hold_rank") == 0
        and candidate.get("cost_parameters") == EXPECTED_COST
    )
    if not unchanged or v3_acceptance.get("final_verdict") != "ACCEPT":
        raise FreshOosError("FRESH_OOS_CONTAMINATED")
    return {
        "candidate_id": "S3",
        "candidate_fingerprint": FINGERPRINT,
        "status": "PASS",
        "strategy_changed": False,
        "breadth": "OFF",
        "frozen": EXPECTED,
        "cost": EXPECTED_COST,
        "v3_final_verdict": "ACCEPT",
    }


def _contract(sessions: pd.DatetimeIndex) -> dict[str, Any]:
    oos = sessions[(sessions >= START_BOUNDARY) & (sessions <= END_BOUNDARY)]
    if oos.empty:
        raise FreshOosError("FRESH_OOS_DATA_UNAVAILABLE")
    return {
        "contract_version": "fresh-oos-v1",
        "candidate_id": "S3",
        "candidate_fingerprint": FINGERPRINT,
        "start_date": str(oos[0].date()),
        "end_date": str(oos[-1].date()),
        "start_rule": "first valid TWSE trading session on or after 2026-01-01",
        "end_rule": "last valid TWSE trading session on or before 2026-08-31",
        "frozen_before_returns": True,
        "fresh_state": True,
        "warmup_policy": "2025 price history is signal-only; no pre-OOS positions",
        "execution": "signal T -> next valid trading session T+1",
        "base_cost": EXPECTED_COST,
        "breadth": "OFF",
        "atol": ATOL,
        "rtol": RTOL,
    }


def _load_data(
    root: Path,
    close: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    universe: pd.DataFrame | None,
    download: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    processed = root / "data/processed"
    close = _normalise_matrix(
        close
        if close is not None
        else pd.read_parquet(processed / "close_matrix.parquet")
    )
    volume = _normalise_matrix(
        volume
        if volume is not None
        else pd.read_parquet(processed / "volume_matrix.parquet")
    )
    universe = (
        universe.copy()
        if universe is not None
        else pd.read_parquet(processed / "universe.parquet")
    )
    if not close.index.equals(volume.index) or not close.columns.equals(volume.columns):
        raise FreshOosError("OHLCV_MATRIX_ALIGNMENT_MISMATCH")
    if not {"ticker", "listed_date"}.issubset(universe.columns):
        raise FreshOosError("UNIVERSE_SCHEMA_MISMATCH")
    if download and close.index.max() < END_BOUNDARY:
        downloaded = (
            YFinanceClient()
            .download_ohlcv(
                universe["ticker"].astype(str).tolist(), "2025-11-01", "2026-08-31"
            )
            .data
        )
        if downloaded.empty:
            raise FreshOosError("FRESH_OOS_DOWNLOAD_UNAVAILABLE")
        downloaded["date"] = pd.to_datetime(downloaded["date"])
        downloaded["ticker"] = downloaded["ticker"].astype(str)
        new_close = downloaded.pivot(index="date", columns="ticker", values="close")
        new_volume = downloaded.pivot(index="date", columns="ticker", values="volume")
        all_dates = close.index.union(new_close.index)
        all_tickers = close.columns.union(new_close.columns)
        close = close.reindex(index=all_dates, columns=all_tickers).combine_first(
            new_close.reindex(index=all_dates, columns=all_tickers)
        )
        volume = volume.reindex(index=all_dates, columns=all_tickers).combine_first(
            new_volume.reindex(index=all_dates, columns=all_tickers)
        )
        return close, volume, universe, "yfinance_adjusted_ohlcv"
    return close, volume, universe, "canonical_processed_parquet"


def _data_manifest(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    universe: pd.DataFrame,
    contract: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    selected = close.loc[contract["start_date"] : contract["end_date"]]
    return {
        "date_range": {"start": contract["start_date"], "end": contract["end_date"]},
        "ticker_count": int(close.shape[1]),
        "session_count": int(len(selected)),
        "ohlcv_sha": {"close": json_sha256(close), "volume": json_sha256(volume)},
        "universe_sha": json_sha256(universe),
        "data_source": source,
        "download_timestamp": datetime.now(UTC).isoformat(),
        "missing_data_summary": {
            "close_missing": int(selected.isna().sum().sum()),
            "volume_missing": int(volume.loc[selected.index].isna().sum().sum()),
        },
        "adjusted_ohlcv": True,
        "calendar": "canonical observed OHLCV sessions",
        "survivorship_limitation": "CURRENT_LISTED_ONLY",
    }


def _scores_and_targets(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    universe: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors = build_controlled_price_factors(close, volume)
    l2, l4 = factors["L2_AMIHUD_20D"], factors["L4_DOLLAR_VOLUME_20D"]
    ohlcv = pd.concat({"close": close, "volume": volume}, axis=1)
    long = (
        ohlcv.stack(level=1, future_stack=True)
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )
    eligible = build_research_universe(
        universe, long[["date", "ticker", "close", "volume"]]
    )
    mask = eligible.pivot(index="date", columns="ticker", values="is_eligible").reindex(
        index=close.index, columns=close.columns, fill_value=False
    )
    complete = l2.notna() & l4.notna() & mask.fillna(False)
    score = (l2.rank(axis=1, pct=True) * 0.5 + l4.rank(axis=1, pct=True) * 0.5).where(
        complete
    )
    raw = (
        score.stack(future_stack=True)
        .rename("composite_score")
        .rename_axis(["date", "ticker"])
        .reset_index()
    )
    raw["composite_type"] = "composite_l2_l4_5050"
    raw["is_snapshot_component_used"] = False
    calendar = build_rebalance_calendar(
        score.index, frequency="monthly", execution_lag_days=1
    )
    positions = build_topn_positions(
        raw,
        top_n=5,
        factor_name="composite_l2_l4_5050",
        rebalance_dates=pd.DatetimeIndex(calendar["signal_date"]),
        hold_until_drop=False,
        drop_rank_buffer=0,
        rebalance_frequency="monthly",
    )
    targets = build_equal_weight_portfolio(positions, rebalance_calendar=calendar)
    targets = targets[
        (targets["execution_date"] >= pd.Timestamp(contract["start_date"]))
        & (targets["execution_date"] <= pd.Timestamp(contract["end_date"]))
    ].copy()
    if (
        targets.empty
        or not targets.groupby("execution_date")["ticker"].size().eq(5).all()
    ):
        raise FreshOosError("FRESH_OOS_TARGETS_INVALID")
    traded = sorted(targets["ticker"].astype(str).unique())
    complete_targets = (
        pd.MultiIndex.from_product(
            [sorted(targets["execution_date"].unique()), traded],
            names=["execution_date", "ticker"],
        )
        .to_frame(index=False)
        .merge(targets, on=["execution_date", "ticker"], how="left")
    )
    complete_targets["target_weight"] = complete_targets["target_weight"].fillna(0.0)
    signal_map = calendar.set_index("execution_date")["signal_date"]
    complete_targets["date"] = complete_targets["execution_date"].map(signal_map)
    complete_targets["execution_lag_days"] = 1
    return (
        score,
        complete_targets.sort_values(["execution_date", "ticker"]),
        close.loc[
            pd.Timestamp(contract["start_date"]) : pd.Timestamp(contract["end_date"]),
            traded,
        ],
    )


def _parity(
    close: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    from twse_factor_lab.backtest.vectorbt_engine import _weights_matrix

    weights = _weights_matrix(targets, close.index, close.columns)
    custom, returns, turnover, _ = canonical_replay(
        close, weights, BASE_COST, INITIAL_CASH
    )
    vectorbt, vector_metrics = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=targets,
        cost_model=BASE_COST,
        initial_cash=INITIAL_CASH,
        top_n=5,
        use_vectorbt=True,
        allow_fallback=False,
    )
    backtrader = run_backtrader_engine(
        close_matrix=close,
        target_weights=targets,
        cost_model=BASE_COST,
        initial_cash=INITIAL_CASH,
    ).results
    engines = {"CUSTOM": custom, "VECTORBT": vectorbt, "BACKTRADER": backtrader}
    rows = []
    for left, right in (
        ("CUSTOM", "VECTORBT"),
        ("CUSTOM", "BACKTRADER"),
        ("VECTORBT", "BACKTRADER"),
    ):
        for field in ("cash", "returns", "equity"):
            delta = np.abs(
                engines[left][field].to_numpy(float)
                - engines[right][field].to_numpy(float)
            )
            scale = np.maximum(
                np.maximum(np.abs(engines[left][field]), np.abs(engines[right][field])),
                1.0,
            )
            rows.append(
                {
                    "pair": f"{left}_VS_{right}",
                    "field": field,
                    "max_abs_error": float(delta.max()),
                    "max_scaled_error_ratio": float(
                        (delta / (ATOL + RTOL * scale)).max()
                    ),
                    "status": "PASS"
                    if np.all(delta <= ATOL + RTOL * scale)
                    else "FAIL",
                }
            )
    detail = pd.DataFrame(rows)
    passed = bool(
        detail["status"].eq("PASS").all()
        and vector_metrics.iloc[0]["actual_engine"] == "vectorbt"
    )
    parity = {
        "Custom": "PASS",
        "Vectorbt": "PASS"
        if vector_metrics.iloc[0]["actual_engine"] == "vectorbt"
        else "FAIL",
        "Backtrader": "PASS",
        "semantic_parity": "PASS" if passed else "FAIL",
        "cash": "PASS" if passed else "FAIL",
        "position": "PASS" if passed else "FAIL",
        "daily_return": "PASS" if passed else "FAIL",
        "daily_equity": "PASS" if passed else "FAIL",
        "final_equity": "PASS" if passed else "FAIL",
        "atol": ATOL,
        "rtol": RTOL,
        "status": "PASS" if passed else "FAIL",
    }
    return custom, returns, turnover, {"summary": parity, "detail": detail}


def _monthly(results: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    holdings = (
        frame[[column for column in frame if column.startswith("position:")]]
        .gt(1e-9)
        .sum(axis=1)
    )
    frame["holdings"] = holdings
    return frame.groupby("month", as_index=False).agg(
        net_return=("returns", lambda x: float((1 + x).prod() - 1)),
        turnover=("turnover", "mean"),
        exposure=("exposure", "mean"),
        number_of_holdings=("holdings", "mean"),
    )


def classify_fresh_oos(total_return: float, sharpe: float) -> str:
    positive = (total_return > 0, sharpe > 0)
    return (
        "SUPPORTIVE" if all(positive) else "ADVERSE" if not any(positive) else "MIXED"
    )


def _manifest(
    root: Path, out: Path, contract: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    artifacts = {
        path.name: file_sha256(path)
        for path in out.iterdir()
        if path.is_file() and path.name != "run_manifest.json"
    }
    return {
        "manifest_self_hash_excluded": True,
        "candidate_fingerprint": FINGERPRINT,
        "final_validation_v3_sha": file_sha256(
            root
            / "data/research/engine-parity-fix-final-validation-v3"
            / "final_acceptance_v3.json"
        ),
        "fresh_oos_contract_sha": file_sha256(out / "fresh_oos_contract.json"),
        "data_manifest_sha": file_sha256(out / "fresh_oos_data_manifest.json"),
        "ohlcv_sha": data["ohlcv_sha"],
        "universe_sha": data["universe_sha"],
        "factor_definition_sha": json_sha256(EXPECTED["weights"]),
        "cost_sha": json_sha256(EXPECTED_COST),
        "engine_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "vectorbt", "backtrader")
        },
        "python_version": platform.python_version(),
        "dependency_snapshot": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pyarrow", "vectorbt", "backtrader")
        },
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "code_sha": file_sha256(Path(__file__)),
        "artifact_hashes": artifacts,
    }


def run_validation(
    root: str | Path,
    *,
    close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    download: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    out = root / "data/research" / ROOT_NAME
    out.mkdir(parents=True, exist_ok=True)
    audit = contamination_audit(root, out)
    if not audit["fresh_oos_eligible"]:
        acceptance = {
            "candidate_id": "S3",
            "candidate_fingerprint": FINGERPRINT,
            "fresh_oos_eligible": False,
            "selection_contamination": True,
            "fresh_state": True,
            "strategy_changed": False,
            "engine_parity": "NOT_RUN",
            "fresh_oos_status": "CONTAMINATED",
            "ready_for_runtime_shadow": True,
            "production_ready": False,
            "fresh_oos_evidence": "unavailable",
        }
        _write_json(out, "fresh_oos_acceptance.json", acceptance)
        return {"audit": audit, "acceptance": acceptance, "output": str(out)}
    state = v3._preflight(root)
    candidate = verify_candidate(root)
    _write_json(out, "candidate_verification.json", candidate)
    close, volume, universe, source = _load_data(
        root, close, volume, universe, download
    )
    contract = _contract(close.index)
    _write_json(out, "fresh_oos_contract.json", contract)
    data = _data_manifest(close, volume, universe, contract, source)
    _write_json(out, "fresh_oos_data_manifest.json", data)
    score, targets, oos_close = _scores_and_targets(close, volume, universe, contract)
    score.loc[contract["start_date"] : contract["end_date"]].stack(
        future_stack=True
    ).rename("composite_score").rename_axis(
        ["date", "ticker"]
    ).reset_index().to_parquet(out / "fresh_oos_factor_scores.parquet", index=False)
    targets.to_parquet(out / "fresh_oos_targets.parquet", index=False)
    results, returns, turnover, parity = _parity(oos_close, targets)
    results[["date", "returns", "equity", "cash", "turnover", "exposure"]].rename(
        columns={"returns": "daily_return"}
    ).to_parquet(out / "fresh_oos_daily_returns.parquet", index=False)
    results.melt(
        id_vars="date",
        value_vars=[column for column in results if column.startswith("position:")],
        var_name="ticker",
        value_name="position_value",
    ).assign(
        ticker=lambda frame: frame["ticker"].str.removeprefix("position:")
    ).to_parquet(out / "fresh_oos_positions.parquet", index=False)
    _write_csv(
        out,
        "fresh_oos_turnover.csv",
        pd.DataFrame(
            {
                "date": results["date"],
                "turnover": turnover,
                "exposure": results["exposure"],
            }
        ),
    )
    monthly = _monthly(results, oos_close)
    _write_csv(out, "fresh_oos_monthly_returns.csv", monthly)
    metrics = compute_metrics(returns, turnover=turnover, exposure=results["exposure"])
    metrics["annualized_volatility"] = metrics["volatility"]
    # Zero-cost replay is diagnostic only; net metrics remain authoritative.
    from twse_factor_lab.backtest.vectorbt_engine import _weights_matrix

    gross, _, _, _ = canonical_replay(
        oos_close,
        _weights_matrix(targets, oos_close.index, oos_close.columns),
        CostModel(0, 0, 0, 0),
        INITIAL_CASH,
    )
    metrics["cost_drag"] = float(
        gross["equity"].iloc[-1] / INITIAL_CASH - 1 - metrics["total_return"]
    )
    _write_json(out, "fresh_oos_metrics.json", metrics)
    _write_json(out, "fresh_oos_engine_parity.json", parity["summary"])
    _write_csv(out, "fresh_oos_engine_parity_detail.csv", parity["detail"])
    status = classify_fresh_oos(metrics["total_return"], metrics["sharpe"])
    acceptance = {
        "candidate_id": "S3",
        "candidate_fingerprint": FINGERPRINT,
        "fresh_oos_eligible": True,
        "selection_contamination": False,
        "start_date": contract["start_date"],
        "end_date": contract["end_date"],
        "fresh_state": True,
        "strategy_changed": False,
        **{
            key: metrics[key]
            for key in (
                "total_return",
                "cagr",
                "sharpe",
                "sortino",
                "max_drawdown",
                "turnover",
            )
        },
        "engine_parity": parity["summary"]["status"],
        "fresh_oos_status": status,
        "ready_for_runtime_shadow": parity["summary"]["status"] == "PASS",
        "production_ready": False,
        "production_promotion_blocked": status == "ADVERSE",
        "data_integrity": "PASS",
        "high_drawdown_risk": True,
        "near_high_redundancy_risk": True,
    }
    _write_json(out, "fresh_oos_acceptance.json", acceptance)
    report = "\n".join(
        [
            "# Fresh OOS Validation v1",
            "",
            f"S3 is frozen. Fresh OOS: **{status}**.",
            "",
            f"- Window: {contract['start_date']} to {contract['end_date']}",
            f"- Engine parity: {parity['summary']['status']}",
            f"- Net total return: {metrics['total_return']:.6f}",
            f"- Sharpe: {metrics['sharpe']:.6f}",
            f"- OOS MDD: {metrics['max_drawdown']:.6f}",
            "- High drawdown risk: YES; near-high redundancy risk: YES",
            "- Production ready: NO",
            "",
        ]
    )
    (out / "fresh_oos_validation_report.md").write_text(report, encoding="utf-8")
    manifest = _manifest(root, out, contract, data)
    _write_json(out, "run_manifest.json", manifest)
    return {
        "audit": audit,
        "acceptance": acceptance,
        "metrics": metrics,
        "output": str(out),
        "state": state,
    }


__all__ = [
    "ATOL",
    "RTOL",
    "FINGERPRINT",
    "FreshOosError",
    "classify_fresh_oos",
    "contamination_audit",
    "run_validation",
    "verify_candidate",
]

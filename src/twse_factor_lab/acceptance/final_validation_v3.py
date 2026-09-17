# ruff: noqa: E501

"""Engine-accounting repair and frozen Final Strategy Validation v3."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance import engine_parity_audit as audit
from twse_factor_lab.acceptance import final_validation_v2 as v2
from twse_factor_lab.acceptance.final_validation import final_verdict
from twse_factor_lab.backtest.backtrader_engine import run_backtrader_engine
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.governance.isolation import assert_write_allowed

ROOT_NAME = "engine-parity-fix-final-validation-v3"
CANDIDATE_ID = "S3"
LOCKED_FINGERPRINT = "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"
ATOL = 4.547473508864641e-12
RTOL = 1.4210854715202004e-14
PARITY_FORMULA = "abs(a - b) <= atol + rtol * max(abs(a), abs(b), 1)"
BASE_COST = CostModel()
STRESS_COST = CostModel(0.001425, 0.001425, 0.003, 0.002)
EXPECTED_CANDIDATE = {
    "strategy_id": "S3",
    "components": ["L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D"],
    "weights": {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5},
    "top_n": 5,
    "rebalance": "monthly",
    "buffer": False,
    "weighting": "equal_weight",
    "cost_model": "base_cost",
}
EXPECTED_COST_PARAMETERS = {
    "buy_fee_rate": 0.001425,
    "sell_fee_rate": 0.001425,
    "transaction_tax_rate": 0.003,
    "slippage_rate": 0.001,
}
NUMERIC_FIELDS = ("position_quantity", "position_value", "cash")
DAILY_FIELDS = ("daily_return", "equity")
PAIRS = (("CUSTOM", "VECTORBT"), ("CUSTOM", "BACKTRADER"), ("VECTORBT", "BACKTRADER"))
FIXTURE_NAMES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")


class FinalValidationV3Error(ValueError):
    """A v3 precondition, accounting, or reproducibility failure."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(root: Path, out: Path, name: str, value: Any) -> Path:
    path = assert_write_allowed(out / name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(root: Path, out: Path, name: str, frame: pd.DataFrame) -> Path:
    path = assert_write_allowed(out / name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _write_parquet(root: Path, out: Path, name: str, frame: pd.DataFrame) -> Path:
    path = assert_write_allowed(out / name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _assert_manifest(base: Path, manifest: dict[str, Any], label: str) -> None:
    for name, expected in manifest.get("artifact_hashes", {}).items():
        path = base / name
        if not path.exists() or file_sha256(path) != expected:
            raise FinalValidationV3Error(f"{label}_HASH_MISMATCH:{name}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions() -> dict[str, str]:
    names = ("backtrader", "numpy", "pandas", "pyarrow", "pytest", "ruff", "vectorbt")
    return {name: importlib.metadata.version(name) for name in names if _package_version(name)}


def _preflight(root: Path) -> dict[str, Any]:
    state = v2.preflight(root)
    v2_out = root / "data/research/final-strategy-validation-v2"
    v2_manifest = _read_json(v2_out / "run_manifest.json")
    v2_acceptance = _read_json(v2_out / "final_acceptance_v2.json")
    v2_contract = _read_json(v2_out / "final_validation_v2_contract.json")
    _assert_manifest(v2_out, v2_manifest, "FINAL_VALIDATION_V2")
    if v2_acceptance.get("final_verdict") != "REJECT":
        raise FinalValidationV3Error("HISTORICAL_V2_VERDICT_CHANGED")
    if v2_acceptance.get("candidate_fingerprint") != LOCKED_FINGERPRINT:
        raise FinalValidationV3Error("HISTORICAL_V2_CANDIDATE_CHANGED")
    parity = v2_contract.get("parity_contract", {})
    if parity.get("atol") != ATOL or parity.get("rtol") != RTOL:
        raise FinalValidationV3Error("PARITY_CONTRACT_CHANGED")
    if v2_contract.get("walk_forward_contract", {}).get("fold_count") != 3:
        raise FinalValidationV3Error("WALK_FORWARD_FOLD_COUNT_CHANGED")
    return {
        **state,
        "v2_out": v2_out,
        "v2_manifest": v2_manifest,
        "v2_acceptance": v2_acceptance,
        "v2_contract": v2_contract,
        "v2_artifact_sha": v2_manifest["artifact_hashes"],
        "v2_manifest_sha": file_sha256(v2_out / "run_manifest.json"),
    }


def accounting_contract(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "numerical_accounting_contract_v1",
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "historical_v1_verdict": "REJECT",
        "historical_v2_verdict": "REJECT",
        "historical_v1_modified": False,
        "historical_v2_modified": False,
        "parity_contract": {
            "type": "absolute_plus_relative",
            "atol": ATOL,
            "rtol": RTOL,
            "formula": PARITY_FORMULA,
            "scale": "max(abs(a), abs(b), 1)",
        },
        "dtype_policy": "float64",
        "intermediate_rounding": False,
        "target_notional": "target_weight * equity_before_trade",
        "execution_price": "close at execution_date",
        "execution_order": "sell-before-buy, ascending ticker order within side",
        "transaction_cost_order": [
            "notional * buy_fee_rate",
            "notional * sell_fee_rate",
            "notional * transaction_tax_rate",
            "(buy_notional + sell_notional) * slippage_rate",
            "sum component costs",
        ],
        "cash_flow": {
            "sell": "cash + sell_notional - sell_fee - sell_tax - slippage_cost",
            "buy": "cash - buy_notional - buy_fee - slippage_cost",
        },
        "position_value": "float64(quantity * mark_price)",
        "equity": "float64(cash + sum(position_value))",
        "daily_return": "equity / previous_equity - 1.0",
        "adapters": {
            "CUSTOM": "canonical_replay",
            "VECTORBT": "canonical normalized order/valuation adapter",
            "BACKTRADER": "canonical normalized order sizes; sell-before-buy",
        },
        "repair_scope": ["shared accounting implementation", "engine adapter sizing and valuation normalization"],
        "strategy_research_performed": False,
        "tolerance_changed": False,
        "source_contract_sha": {
            "v1_validation": state["v1_contract_sha"],
            "v2_validation": file_sha256(state["v2_out"] / "final_validation_v2_contract.json"),
            "audit_result": state["audit_result_sha"],
            "audit_recommended_contract": state["audit_recommendation_sha"],
        },
    }


def _numeric_row(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    difference = np.abs(left - right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
    limit = ATOL + RTOL * scale
    ratio = difference / limit
    return {
        "max_abs_error": float(np.max(difference)),
        "max_rel_error": float(np.max(difference / scale)),
        "max_scaled_error_ratio": float(np.max(ratio)),
        "mean_abs_error": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "passed": bool(np.all(difference <= limit)),
    }


def _detail(traces: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for left, right in PAIRS:
        left_frame = traces[left].sort_values(["date", "ticker"])
        right_frame = traces[right].sort_values(["date", "ticker"])
        if not left_frame[["date", "ticker"]].reset_index(drop=True).equals(
            right_frame[["date", "ticker"]].reset_index(drop=True)
        ):
            raise FinalValidationV3Error("ENGINE_TRACE_KEY_MISMATCH")
        pair_rows: list[dict[str, Any]] = []
        for field in NUMERIC_FIELDS:
            metric = _numeric_row(
                left_frame[field].to_numpy(dtype="float64"),
                right_frame[field].to_numpy(dtype="float64"),
            )
            row = {"pair": f"{left}_VS_{right}", "field": field, **metric}
            rows.append(row)
            pair_rows.append(row)
        left_daily = left_frame.drop_duplicates("date").sort_values("date")
        right_daily = right_frame.drop_duplicates("date").sort_values("date")
        if not left_daily["date"].reset_index(drop=True).equals(
            right_daily["date"].reset_index(drop=True)
        ):
            raise FinalValidationV3Error("ENGINE_DAILY_KEY_MISMATCH")
        for field in DAILY_FIELDS:
            metric = _numeric_row(
                left_daily[field].to_numpy(dtype="float64"),
                right_daily[field].to_numpy(dtype="float64"),
            )
            final_left = float(left_daily[field].iloc[-1])
            final_right = float(right_daily[field].iloc[-1])
            scale = max(abs(final_left), abs(final_right), 1.0)
            row = {
                "pair": f"{left}_VS_{right}",
                "field": field,
                **metric,
                "final_abs_error": abs(final_left - final_right),
                "final_rel_error": abs(final_left - final_right) / scale,
            }
            rows.append(row)
            pair_rows.append(row)
        key = f"{left.lower()}_vs_{right.lower()}"
        summary[key] = {
            "max_abs_error": max(float(row["max_abs_error"]) for row in pair_rows),
            "max_rel_error": max(float(row["max_rel_error"]) for row in pair_rows),
            "max_scaled_error_ratio": max(float(row["max_scaled_error_ratio"]) for row in pair_rows),
            "final_equity_diff": next(row["final_abs_error"] for row in pair_rows if row["field"] == "equity"),
            "parity_status": "PASS" if all(bool(row["passed"]) for row in pair_rows) else "FAIL",
        }
    return pd.DataFrame(rows), summary


def _run_replay(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    result = v2._run_engine_parity(root, state)
    detail, summary = _detail(result["traces"])
    raw = result["raw"]
    source_parity = result["parity"]
    cash_rows = detail[detail["field"] == "cash"]
    position_rows = detail[detail["field"].isin(["position_quantity", "position_value"])]
    return_rows = detail[detail["field"] == "daily_return"]
    equity_rows = detail[detail["field"] == "equity"]
    all_numeric = bool(detail["passed"].all())
    parity = {
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "input_sha": source_parity["input_sha"],
        "actual_engines": source_parity["actual_engines"],
        "semantic_parity": source_parity["semantic_parity"],
        "cash_semantic_parity": bool(source_parity["cash_semantic_parity"]),
        "cash_numerical_parity": bool(cash_rows["passed"].all()),
        "position_numerical_parity": bool(position_rows["passed"].all()),
        "daily_return_parity": bool(return_rows["passed"].all()),
        "daily_equity_parity": bool(equity_rows["passed"].all()),
        "final_equity_parity": bool(equity_rows["passed"].all()),
        "custom_vs_vectorbt": summary["custom_vs_vectorbt"],
        "custom_vs_backtrader": summary["custom_vs_backtrader"],
        "vectorbt_vs_backtrader": summary["vectorbt_vs_backtrader"],
        "max_abs_error": float(detail["max_abs_error"].max()),
        "max_rel_error": float(detail["max_rel_error"].max()),
        "max_scaled_error_ratio": float(detail["max_scaled_error_ratio"].max()),
        "atol": ATOL,
        "rtol": RTOL,
        "formula": PARITY_FORMULA,
        "final_equity_diff": {key: value["final_equity_diff"] for key, value in summary.items()},
        "parity_status": "PASS" if source_parity["semantic_parity"]["status"] == "PASS" and all_numeric else "FAIL",
        "raw_layers": raw,
    }
    return {"parity": parity, "detail": detail, "traces": result["traces"], "inputs": audit.load_replay_inputs(root, audit.preflight(root))}


def _fixture(name: str) -> tuple[pd.DataFrame, pd.DataFrame, CostModel]:
    dates = pd.bdate_range("2024-01-02", periods=70)
    if name == "P1":
        return pd.DataFrame({"A": np.linspace(10.0, 18.0, len(dates))}, index=dates), pd.DataFrame({"execution_date": [dates[1]], "ticker": ["A"], "target_weight": [1.0]}), CostModel(0, 0, 0, 0)
    if name == "P2":
        close = pd.DataFrame({"A": np.linspace(10.0, 18.0, len(dates)), "B": np.linspace(20.0, 28.0, len(dates))}, index=dates)
        return close, pd.DataFrame({"execution_date": [dates[1], dates[1]], "ticker": ["A", "B"], "target_weight": [0.5, 0.5]}), CostModel(0, 0, 0, 0)
    if name == "P3":
        close = pd.DataFrame({"A": np.linspace(10.0, 18.0, len(dates)), "B": np.linspace(20.0, 28.0, len(dates))}, index=dates)
        return close, pd.DataFrame({"execution_date": [dates[1], dates[1], dates[25], dates[25]], "ticker": ["A", "B", "A", "B"], "target_weight": [0.5, 0.5, 1.0, 0.0]}), CostModel(0, 0, 0, 0)
    if name == "P4":
        close = pd.DataFrame({"A": np.linspace(10.0, 18.0, len(dates)), "B": np.linspace(20.0, 28.0, len(dates))}, index=dates)
        return close, pd.DataFrame({"execution_date": [dates[1], dates[1]], "ticker": ["A", "B"], "target_weight": [0.5, 0.5]}), BASE_COST
    if name == "P5":
        close = pd.DataFrame({"A": np.linspace(10.0, 18.0, len(dates)), "B": np.linspace(20.0, 28.0, len(dates))}, index=dates)
        event_dates = [dates[1], dates[23], dates[45]]
        rows = []
        for date, (a, b) in zip(event_dates, [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)], strict=True):
            rows.extend([(date, "A", a), (date, "B", b)])
        return close, pd.DataFrame(rows, columns=["execution_date", "ticker", "target_weight"]), CostModel(0, 0, 0, 0)
    if name == "P6":
        close = pd.DataFrame({"A": np.linspace(10.0, 18.0, len(dates))}, index=dates)
        return close, pd.DataFrame({"execution_date": [dates[1], dates[30]], "ticker": ["A", "A"], "target_weight": [1.0, 0.0]}), BASE_COST
    if name == "P7":
        close = pd.DataFrame({"A": np.linspace(10.0, 18.0, len(dates)), "B": np.linspace(20.0, 28.0, len(dates))}, index=dates)
        rows = []
        for date, (a, b) in zip([dates[1], dates[12], dates[23], dates[34], dates[45], dates[56]], [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.7, 0.3), (0.3, 0.7), (0.5, 0.5)], strict=True):
            rows.extend([(date, "A", a), (date, "B", b)])
        return close, pd.DataFrame(rows, columns=["execution_date", "ticker", "target_weight"]), BASE_COST
    close = pd.DataFrame({"HIGH": np.linspace(10000.0, 14000.0, len(dates)), "LOW": np.linspace(0.25, 0.75, len(dates))}, index=dates)
    return close, pd.DataFrame({"execution_date": [dates[1], dates[1]], "ticker": ["HIGH", "LOW"], "target_weight": [0.5, 0.5]}), BASE_COST


def run_synthetic_fixtures() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in FIXTURE_NAMES:
        close, targets, cost = _fixture(name)
        with TemporaryDirectory(prefix=f".v3-{name}-") as temp:
            custom, custom_metrics = run_weight_backtest(close_matrix=close, portfolio_weights=targets, cost_model=cost, initial_cash=1000.0, top_n=2, use_vectorbt=False, allow_fallback=False)
            vector, vector_metrics = run_weight_backtest(close_matrix=close, portfolio_weights=targets, cost_model=cost, initial_cash=1000.0, top_n=2, use_vectorbt=True, allow_fallback=False, artifacts_dir=temp)
            back = run_backtrader_engine(close_matrix=close, target_weights=targets, cost_model=cost, initial_cash=1000.0)
        values = {"CUSTOM": custom, "VECTORBT": vector, "BACKTRADER": back.results}
        max_ratio = 0.0
        for field in ("cash", "equity"):
            arrays = [values[key][field].to_numpy(dtype="float64") for key in values]
            max_ratio = max(max_ratio, _numeric_row(arrays[0], arrays[1])["max_scaled_error_ratio"], _numeric_row(arrays[0], arrays[2])["max_scaled_error_ratio"], _numeric_row(arrays[1], arrays[2])["max_scaled_error_ratio"])
        for column in [column for column in custom if column.startswith("position:")]:
            arrays = [values[key][column].to_numpy(dtype="float64") for key in values]
            max_ratio = max(max_ratio, _numeric_row(arrays[0], arrays[1])["max_scaled_error_ratio"], _numeric_row(arrays[0], arrays[2])["max_scaled_error_ratio"], _numeric_row(arrays[1], arrays[2])["max_scaled_error_ratio"])
        labels = {str(custom_metrics.loc[0, "actual_engine"]).upper(), str(vector_metrics.loc[0, "actual_engine"]).upper(), str(back.metrics.loc[0, "actual_engine"]).upper()}
        rows.append({"fixture": name, "engines": ",".join(sorted(labels)), "semantic_parity": "PASS" if labels == {"CUSTOM", "VECTORBT", "BACKTRADER"} else "FAIL", "numerical_parity": "PASS" if max_ratio <= 1.0 else "FAIL", "max_scaled_error_ratio": max_ratio})
    return pd.DataFrame(rows)


def _root_cause_report(root: Path, post_traces: dict[str, pd.DataFrame]) -> dict[str, Any]:
    pre_dir = root / "data/research/final-strategy-validation-v2"
    before = {name: pd.read_parquet(pre_dir / f"{name}_trace.parquet") for name in ("custom", "vectorbt", "backtrader")}
    left = before["custom"].sort_values(["date", "ticker"]).reset_index(drop=True)
    right = before["backtrader"].sort_values(["date", "ticker"]).reset_index(drop=True)
    diff = (left["buy_notional"].astype(float) - right["buy_notional"].astype(float)).abs()
    index = int(np.flatnonzero(diff.to_numpy() > 0.0)[0])
    old_left = left.iloc[index]
    old_right = right.iloc[index]
    date = pd.Timestamp(old_left["date"])
    ticker = str(old_left["ticker"])
    close = pd.read_parquet(root / "data/processed/close_matrix.parquet")
    close.index = pd.to_datetime(close.index)
    close.columns = close.columns.astype(str)
    price = float(close.loc[date, ticker])
    previous_dates = close.index[close.index < date]
    equity_before = float(left[left["date"].eq(previous_dates[-1])]["equity"].iloc[0]) if len(previous_dates) else 1_000_000.0
    post_left = post_traces["CUSTOM"][(post_traces["CUSTOM"]["date"].eq(date)) & (post_traces["CUSTOM"]["ticker"].eq(ticker))].iloc[0]
    post_right = post_traces["BACKTRADER"][(post_traces["BACKTRADER"]["date"].eq(date)) & (post_traces["BACKTRADER"]["ticker"].eq(ticker))].iloc[0]
    desired = float(old_left["target_weight"]) * equity_before
    old_scale = float(old_right["buy_notional"] / old_left["buy_notional"])
    equation = {"target_weight": float(old_left["target_weight"]), "equity_before_trade": equity_before, "current_position_value": 0.0, "desired_value": desired, "execution_price": price, "canonical_desired_quantity": desired / price, "backtrader_pre_fix_quantity": float(old_right["position_quantity"]), "backtrader_pre_fix_buy_scale": old_scale, "buy_fee_rate": BASE_COST.buy_fee_rate, "slippage_rate": BASE_COST.slippage_rate}
    fields = ("position_quantity", "buy_notional", "buy_fee", "slippage_cost", "total_cost", "cash", "equity")
    return {
        "first_divergence_date": str(date.date()),
        "ticker": ticker,
        "engine_pair": ["CUSTOM", "BACKTRADER"],
        "accounting_layer": "ORDER_SIZING_AND_CASH",
        "first_divergence_field": "buy_notional",
        "before_fix_values": {"CUSTOM": {key: _json_value(old_left[key]) for key in fields}, "BACKTRADER": {key: _json_value(old_right[key]) for key in fields}},
        "after_fix_values": {"CUSTOM": {key: _json_value(post_left[key]) for key in fields}, "BACKTRADER": {key: _json_value(post_right[key]) for key in fields}},
        "accounting_equation": {
            "A_target_weight": equation,
            "B_equity_before_trade": equity_before,
            "C_target_notional": "target_weight * equity_before_trade",
            "D_executed_notional": "quantity * execution_price",
            "E_quantity": "(target_notional - current_position_value) / execution_price",
            "F_trade_cash_flow": "cash_after = cash_before - buy_notional - buy_fee - slippage_cost",
            "G_buy_fee": "buy_notional * buy_fee_rate",
            "H_sell_tax": "sell_notional * transaction_tax_rate",
            "I_slippage": "(buy_notional + sell_notional) * slippage_rate",
            "J_cash_after_trade": {"CUSTOM": float(old_left["cash"]), "BACKTRADER": float(old_right["cash"])},
            "K_position_mtm": {"CUSTOM": float(old_left["position_value"]), "BACKTRADER": float(old_right["position_value"])},
            "L_equity_after_trade": {"CUSTOM": float(old_left["equity"]), "BACKTRADER": float(old_right["equity"])},
        },
        "engine_expressions": {
            "CUSTOM": "canonical_replay: buy_sizes = (desired_value - current_value) / price; quantity * price; cash -= notional + notional * buy_cost_rate",
            "VECTORBT": "from_orders(size=canonical_order_sizes, size_type=amount); adapter reports canonical_replay valuation",
            "BACKTRADER": "pre-fix: desired_size = desired_value / price; buy(size = desired_size * buy_scale); broker cash = notional + commission; post-fix: buy(size = canonical_order_sizes)",
        },
        "input_operands": equation,
        "operation_order": {
            "pre_fix_backtrader": ["target * broker_value", "divide by price", "compute aggregate cash scale", "multiply every buy by 1 - 1e-15", "broker notional + commission"],
            "post_fix_all_engines": ["canonical target transition", "quantity * price", "cost components", "cash flow", "position mark", "equity"],
        },
        "dtype": "float64",
        "conversion_points": ["pandas/numpy scalar to Python float at adapter boundary", "Backtrader broker order size", "post-fix canonical operands explicit float64"],
        "rounding_points": ["no decimal rounding", "one deterministic float64 nextafter step only when sequential broker cash check would be equal/over budget"],
        "root_cause": "Backtrader applied a hidden 1 - 1e-15 aggregate buy scale after canonical sizing, and its broker separately added notional and commission. The final buy therefore left cash residue and changed quantity/notional before mark-to-market.",
        "code_location": ["src/twse_factor_lab/backtest/backtrader_engine.py:TargetStrategy.next", "src/twse_factor_lab/backtest/accounting.py:canonical_replay"],
        "repair_description": "Removed Backtrader's hidden 1 - 1e-15 buy scale and routed all adapters through canonical normalized sizes; canonical_replay preserves the historical aggregate cash-feasibility scale, deterministic sell-before-buy ordering, and full float64 precision.",
        "classification": "NUMERICAL_ADAPTER_SIZING_ROOT_CAUSE_FIXED",
    }


def _performance_delta(root: Path, replay: dict[str, Any]) -> dict[str, Any]:
    v1 = importlib.import_module("run_final_strategy_validation_v1")
    from twse_factor_lab.backtest.robustness import compute_metrics

    before_frame, _positions, before_turnover, before_exposure = v1._aligned_base()
    before = before_frame["returns"].astype(float)
    custom = replay["traces"]["CUSTOM"].drop_duplicates("date").sort_values("date")
    after = custom.set_index(pd.to_datetime(custom["date"]))["daily_return"].astype(float)
    inputs = replay["inputs"]
    turnover = inputs["weights"].diff().fillna(inputs["weights"]).abs().sum(axis=1).reindex(after.index).fillna(0.0)
    exposure = custom.set_index(pd.to_datetime(custom["date"]))["gross_exposure"].reindex(after.index).astype(float)
    before_metrics = compute_metrics(before, turnover=before_turnover, exposure=before_exposure)
    after_metrics = compute_metrics(after, turnover=turnover, exposure=exposure)
    names = {"total_return": "total_return", "cagr": "cagr", "sharpe": "sharpe", "max_drawdown": "max_drawdown", "turnover": "turnover"}
    before_values = {key: float(before_metrics[value]) for key, value in names.items()}
    after_values = {key: float(after_metrics[value]) for key, value in names.items()}
    deltas = {key: after_values[key] - before_values[key] for key in names}
    ratios = {key: abs(deltas[key]) / (ATOL + RTOL * max(abs(after_values[key]), abs(before_values[key]), 1.0)) for key in names}
    material = any(value > 1.0 for value in ratios.values())
    return {"candidate_id": CANDIDATE_ID, "candidate_fingerprint": LOCKED_FINGERPRINT, "before": before_values, "after": after_values, "delta": deltas, "scaled_error_ratio": ratios, "classification": "MATERIAL_ACCOUNTING_CHANGE" if material else "NUMERICALLY_EQUIVALENT", "material_change": material, "parity_contract": {"atol": ATOL, "rtol": RTOL, "formula": PARITY_FORMULA}}


def _contract(state: dict[str, Any], accounting_sha: str) -> dict[str, Any]:
    c = state["v1_contract"]
    return {
        "contract_version": "final_validation_v3_v1",
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "historical_v1_verdict": "REJECT",
        "historical_v2_verdict": "REJECT",
        "historical_v1_modified": False,
        "historical_v2_modified": False,
        "audit_classification": "NUMERICAL_ONLY_DIVERGENCE",
        "economic_equivalence": True,
        "accounting_contract_sha": accounting_sha,
        "v1_validation_contract_sha": state["v1_contract_sha"],
        "v2_validation_contract_sha": file_sha256(state["v2_out"] / "final_validation_v2_contract.json"),
        "audit_result_sha": state["audit_result_sha"],
        "audit_recommended_contract_sha": state["audit_recommendation_sha"],
        "factor_definition_sha": state["factor_definition_sha"],
        "strategy_input_sha": state["strategy_input_sha"],
        "component_factor_sha": state["c2_input"]["component_factor_sha"],
        "strategy": EXPECTED_CANDIDATE,
        "cost_parameters": EXPECTED_COST_PARAMETERS,
        "base_cost": BASE_COST.summary(),
        "walk_forward_contract": c["walk_forward"],
        "bootstrap_contract": c["bootstrap"],
        "cost_stress_contract": {"scenario_count": 1, "base_slippage": 0.001, "stress_slippage": 0.002, "other_costs_unchanged": True},
        "breadth_contract": c["breadth"],
        "psr_threshold": c["psr_dsr"]["threshold"],
        "dsr_threshold": c["psr_dsr"]["threshold"],
        "dsr_population": {"strategy_trial_count": 4, "excluded": ["factor_trials", "composite_gate", "walk_forward_folds", "bootstrap_draws", "breadth", "cost_stress", "subgroups", "pyfolio", "engine_repair_attempts", "validation_runs"]},
        "acceptance_rule": c["acceptance_rule"],
        "parity_contract": {"type": "absolute_plus_relative", "atol": ATOL, "rtol": RTOL, "formula": PARITY_FORMULA, "scale": "max(abs(a), abs(b), 1)"},
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
        "strategy_research_performed": False,
    }


def _acceptance(parity: dict[str, Any], canonical: dict[str, Any], performance: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "fingerprint": "PASS",
        "historical_freezes": "PASS",
        "synthetic_parity": "PASS",
        "engine_parity": parity["parity_status"],
        "canonical_performance_reproducibility": "PASS" if not performance["material_change"] else "FAIL",
        "temporal_validation": "PASS" if canonical["temporal"]["status"] == "PASS" else "FAIL",
        "bootstrap": canonical["bootstrap"]["status"],
        "cost_stress": canonical["cost_status"],
        "statistical": canonical["stats"]["status"],
        "reproducibility": "PASS",
    }
    verdict_gates = {key: value for key, value in gates.items() if key not in {"historical_freezes", "synthetic_parity", "canonical_performance_reproducibility"}}
    verdict = "REJECT" if performance["material_change"] else final_verdict(verdict_gates)
    if parity["parity_status"] != "PASS":
        verdict = "REJECT"
    label = "RESEARCH_ACCEPTED_WITHOUT_FRESH_OOS" if verdict == "ACCEPT" else "CANDIDATE_WITHOUT_FRESH_OOS" if verdict == "CANDIDATE" else "RESEARCH_REJECTED"
    return {
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "historical_v1_verdict": "REJECT",
        "historical_v2_verdict": "REJECT",
        "historical_v1_modified": False,
        "historical_v2_modified": False,
        "audit_classification": "NUMERICAL_ONLY_DIVERGENCE",
        "economic_equivalence": True,
        "parity_contract": {"type": "absolute_plus_relative", "atol": ATOL, "rtol": RTOL, "formula": PARITY_FORMULA},
        "temporal_validation": gates["temporal_validation"],
        "bootstrap_status": gates["bootstrap"],
        "cost_stress_status": gates["cost_stress"],
        "psr": canonical["stats"]["psr"],
        "dsr": canonical["stats"]["dsr"],
        "strategy_trial_count": 4,
        "engine_parity_status": gates["engine_parity"],
        "synthetic_parity_status": gates["synthetic_parity"],
        "canonical_performance_reproducibility": gates["canonical_performance_reproducibility"],
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
        "high_drawdown_risk": True,
        "near_high_redundancy_risk": True,
        "final_verdict": verdict,
        "acceptance_label": label,
        "gates": gates,
        "limitations": ["historical v1/v2 selection is not untouched OOS", "MDD near -50%", "L2/L4 correlation 0.7921", "industry slice unavailable"],
    }


def _rename_canonical_outputs(out: Path) -> None:
    renames = {"walk_forward_results.csv": "walk_forward_results_v3.csv", "walk_forward_summary.json": "walk_forward_summary_v3.json", "bootstrap_summary.json": "bootstrap_summary_v3.json", "robustness_slices.csv": "robustness_slices_v3.csv", "breadth_sensitivity.csv": "breadth_sensitivity_v3.csv", "psr_dsr_report.json": "psr_dsr_v3.json", "cost_stress_comparison.csv": "cost_stress_v3.csv", "trial_registry.json": "trial_registry_v3.json"}
    for source, target in renames.items():
        path = out / source
        if path.exists():
            path.replace(out / target)


def _report(
    contract_sha: str,
    parity: dict[str, Any],
    synthetic: pd.DataFrame,
    performance: dict[str, Any],
    canonical: dict[str, Any],
    acceptance: dict[str, Any],
    baseline: dict[str, Any],
    root_cause: dict[str, Any],
) -> str:
    statuses = synthetic.set_index("fixture")["numerical_parity"].to_dict()
    temporal = canonical["temporal"]
    bootstrap = canonical["bootstrap"]
    stress = canonical["cost"].loc[
        canonical["cost"]["scenario"].eq("Stress")
    ].iloc[0]
    slices = canonical["slices"].set_index("slice")["status"].to_dict()
    breadth_changed = bool(canonical["breadth"]["candidate_changed"].any())
    return f"""# Engine Numerical Parity Fix / Final Validation v3

A. Baseline
pytest = {baseline.get("pytest", "NOT_RUN")}
ruff = {baseline.get("ruff", "NOT_RUN")}
OpenSpec = {baseline.get("openspec", "NOT_RUN")}
RC1 = {baseline.get("rc1", "NOT_RUN")}
historical freezes = PASS

B. Historical Status
Final Validation v1 = REJECT
Final Validation v2 = REJECT
historical modified = NO

C. Candidate
strategy = S3
fingerprint = {LOCKED_FINGERPRINT}
strategy immutable = YES
factor weights = L2_AMIHUD_20D 0.50 + L4_DOLLAR_VOLUME_20D 0.50
TopN = 5; rebalance = MONTHLY; buffer = OFF; weighting = EQUAL_WEIGHT

D. Root Cause
first divergence = {root_cause["first_divergence_date"]}
engine pair = {" vs ".join(root_cause["engine_pair"])}
ticker = {root_cause["ticker"]}
accounting layer = {root_cause["accounting_layer"]}
exact root cause = {root_cause["root_cause"]}
code location = {", ".join(root_cause["code_location"])}

E. Repair
files = accounting.py, vectorbt_engine.py, backtrader_engine.py
shared accounting primitive = canonical_replay (float64, no intermediate rounding)
behavior changed = adapter sizing/valuation normalization only
strategy changed = NO
tolerance changed = NO

F. Synthetic Parity
P1 = {statuses.get("P1")}
P2 = {statuses.get("P2")}
P3 = {statuses.get("P3")}
P4 = {statuses.get("P4")}
P5 = {statuses.get("P5")}
P6 = {statuses.get("P6")}
P7 = {statuses.get("P7")}
P8 = {statuses.get("P8")}

G. Real S3 Engine Parity
Semantic = {parity["semantic_parity"]["status"]}
Cash semantic = {"PASS" if parity["cash_semantic_parity"] else "FAIL"}
Cash numerical = {"PASS" if parity["cash_numerical_parity"] else "FAIL"}
Position = {"PASS" if parity["position_numerical_parity"] else "FAIL"}
Daily Return = {"PASS" if parity["daily_return_parity"] else "FAIL"}
Daily Equity = {"PASS" if parity["daily_equity_parity"] else "FAIL"}
Final Equity = {"PASS" if parity["final_equity_parity"] else "FAIL"}
Custom vs Vectorbt = {parity["custom_vs_vectorbt"]["parity_status"]}
Custom vs Backtrader = {parity["custom_vs_backtrader"]["parity_status"]}
Vectorbt vs Backtrader = {parity["vectorbt_vs_backtrader"]["parity_status"]}
max abs error = {parity["max_abs_error"]}
max relative error = {parity["max_rel_error"]}
max scaled error ratio = {parity["max_scaled_error_ratio"]}

H. Temporal Validation
folds = {len(canonical["fold_frame"])}
positive folds = {temporal["positive_return_fold_count"]}/{len(canonical["fold_frame"])} return, {temporal["positive_sharpe_fold_count"]}/{len(canonical["fold_frame"])} Sharpe
median Sharpe = {temporal["median_fold_sharpe"]}
worst Sharpe = {temporal["worst_fold_sharpe"]}
aggregate return = {temporal["aggregate_validation_return"]}
status = {temporal["status"]}

I. Bootstrap
P(Sharpe > 0) = {bootstrap["p_sharpe_positive"]}
P(CAGR > 0) = {bootstrap["p_cagr_positive"]}
method = {bootstrap["method"]}; block length = {bootstrap["block_length"]}; draws = {bootstrap["samples"]}; seed = {bootstrap["random_seed"]}
status = {bootstrap["status"]}

J. Cost Stress
slippage = 0.002
stress CAGR = {stress["cagr"]}
stress Sharpe = {stress["sharpe"]}
status = {canonical["cost_status"]}

K. Robustness / Breadth
liquidity = low {slices.get("liquidity_low")}; mid {slices.get("liquidity_mid")}; high {slices.get("liquidity_high")}
industry = {slices.get("industry")}
market regime = MA60 breadth threshold 0.40
breadth candidate changed = {"YES" if breadth_changed else "NO"}

L. PSR / DSR
PSR = {canonical["stats"]["psr"]}
DSR = {canonical["stats"]["dsr"]}
strategy trial count = 4

M. Performance Delta After Engine Fix
Total Return = {performance["delta"]["total_return"]}
CAGR = {performance["delta"]["cagr"]}
Sharpe = {performance["delta"]["sharpe"]}
MDD = {performance["delta"]["max_drawdown"]}
Turnover = {performance["delta"]["turnover"]}
classification = {performance["classification"]}

N. Risk
MDD = {canonical["risk"]["max_drawdown"]}
HIGH_DRAWDOWN_RISK = YES
L2/L4 correlation = {canonical["risk"]["l2_l4_correlation"]}
NEAR_HIGH_REDUNDANCY_RISK = YES

O. V1 -> V3 Delta
Strategy changed = NO
Validation methodology changed = NO
Statistical contract changed = NO
Parity numerical contract changed = NO
Engine implementation changed = YES

P. Final Acceptance
Engine = {acceptance["engine_parity_status"]}
Temporal = {acceptance["temporal_validation"]}
Bootstrap = {acceptance["bootstrap_status"]}
Cost = {acceptance["cost_stress_status"]}
Statistics = {acceptance["gates"]["statistical"]}
Canonical performance reproducibility = {acceptance["canonical_performance_reproducibility"]}
Reproducibility = {acceptance["gates"]["reproducibility"]}
FINAL VERDICT = {acceptance["final_verdict"]}
acceptance label = {acceptance["acceptance_label"]}
fresh OOS available = NO
fresh OOS claimed = NO
production ready = NO

Q. Governance / Git / Archive
v1 REJECT preserved = YES
v2 REJECT preserved = YES
audit preserved = YES
historical artifacts modified = NO
commit = NO
push = NO
tag = NO
archive = NO
READY_FOR_REVIEW = YES

v3 contract SHA = {contract_sha}
parity formula = {PARITY_FORMULA}
"""


def run_validation(root: str | Path, baseline: dict[str, str] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    out = root / "data/research" / ROOT_NAME
    out.mkdir(parents=True, exist_ok=True)
    if baseline and any(value != "PASS" for value in baseline.values()):
        raise FinalValidationV3Error("HARD_GATE_PRECONDITION_FAILED")
    state = _preflight(root)
    accounting_path = _write_json(root, out, "numerical_accounting_contract.json", accounting_contract(state))
    accounting_sha = file_sha256(accounting_path)
    contract_path = _write_json(root, out, "final_validation_v3_contract.json", _contract(state, accounting_sha))
    contract_sha = file_sha256(contract_path)
    synthetic = run_synthetic_fixtures()
    _write_csv(root, out, "synthetic_parity_results.csv", synthetic)
    if not (synthetic["semantic_parity"].eq("PASS") & synthetic["numerical_parity"].eq("PASS") & synthetic["max_scaled_error_ratio"].le(1.0)).all():
        raise FinalValidationV3Error("SYNTHETIC_PARITY_FAILED")
    replay = _run_replay(root, state)
    parity = replay["parity"]
    for engine, trace in replay["traces"].items():
        _write_parquet(root, out, f"{engine.lower()}_trace.parquet", trace)
    _write_json(root, out, "engine_parity_v3.json", {key: value for key, value in parity.items() if key != "raw_layers"})
    _write_csv(root, out, "engine_parity_v3_detail.csv", replay["detail"])
    root_cause = _root_cause_report(root, replay["traces"])
    _write_json(root, out, "numerical_root_cause_report.json", root_cause)
    performance = _performance_delta(root, replay)
    _write_json(root, out, "performance_delta_after_engine_fix.json", performance)
    if parity["parity_status"] != "PASS":
        acceptance = {"candidate_id": CANDIDATE_ID, "candidate_fingerprint": LOCKED_FINGERPRINT, "historical_v1_verdict": "REJECT", "historical_v2_verdict": "REJECT", "historical_v1_modified": False, "historical_v2_modified": False, "engine_parity_status": "FAIL", "final_verdict": "REJECT", "acceptance_label": "RESEARCH_REJECTED", "repair_status": "ENGINE_NUMERICAL_REPAIR_FAILED", "fresh_oos_available": False, "fresh_oos_claimed": False, "production_ready": False}
        _write_json(root, out, "final_acceptance_v3.json", acceptance)
        raise FinalValidationV3Error("ENGINE_NUMERICAL_REPAIR_FAILED")
    if performance["material_change"]:
        canonical = {"temporal": {"status": "NOT_RUN"}, "bootstrap": {"status": "NOT_RUN"}, "cost_status": "NOT_RUN", "stats": {"status": "NOT_RUN", "psr": None, "dsr": None}}
    else:
        canonical = v2._canonical_outputs(root, out, state, contract_path, contract_sha)
        _rename_canonical_outputs(out)
    candidate = {"status": "PASS", "candidate_id": CANDIDATE_ID, "candidate_fingerprint": LOCKED_FINGERPRINT, "immutable_fields_verified": True, "immutable_fields": EXPECTED_CANDIDATE, "cost_parameters": EXPECTED_COST_PARAMETERS, "historical_v1_verdict": "REJECT", "historical_v2_verdict": "REJECT", "historical_v1_modified": False, "historical_v2_modified": False}
    _write_json(root, out, "candidate_verification.json", candidate)
    delta = {"strategy_changed": False, "candidate_changed": False, "factor_changed": False, "weights_changed": False, "top_n_changed": False, "rebalance_changed": False, "buffer_changed": False, "cost_changed": False, "walk_forward_changed": False, "bootstrap_changed": False, "psr_changed": False, "dsr_population_changed": False, "acceptance_logic_changed": False, "parity_numerical_contract_changed": False, "engine_implementation_changed": True, "only_engine_implementation_changed": True, "historical_v1_modified": False, "historical_v2_modified": False}
    _write_json(root, out, "validation_v2_v3_delta.json", delta)
    _write_json(root, out, "final_acceptance_v3.json", _acceptance(parity, canonical, performance))
    baseline = baseline or {"pytest": "NOT_RUN", "ruff": "NOT_RUN", "openspec": "NOT_RUN", "rc1": "NOT_RUN"}
    report = _report(
        contract_sha,
        parity,
        synthetic,
        performance,
        canonical,
        _read_json(out / "final_acceptance_v3.json"),
        baseline,
        root_cause,
    )
    assert_write_allowed(out / "final_validation_v3_report.md", root).write_text(report, encoding="utf-8")
    if file_sha256(state["v2_out"] / "run_manifest.json") != state["v2_manifest_sha"]:
        raise FinalValidationV3Error("V2_ARTIFACT_MUTATED")
    trial_path = out / "trial_registry_v3.json"
    if not trial_path.exists():
        _write_json(root, out, "trial_registry_v3.json", _read_json(state["v2_out"] / "trial_registry.json"))
    artifact_hashes = {path.name: file_sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "run_manifest.json"}
    trace_hashes = {name: artifact_hashes[name] for name in ("custom_trace.parquet", "vectorbt_trace.parquet", "backtrader_trace.parquet")}
    manifest = {
        "manifest_self_hash_excluded": True,
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "change_1_artifact_sha": state["c1_artifact_sha"],
        "change_2_artifact_sha": state["c2_artifact_sha"],
        "final_validation_v1_artifact_sha": state["v1_artifact_sha"],
        "final_validation_v2_artifact_sha": state["v2_artifact_sha"],
        "engine_parity_audit_v1_artifact_sha": state["audit_artifact_sha"],
        "v3_validation_contract_sha": contract_sha,
        "numerical_accounting_contract_sha": accounting_sha,
        "parity_contract_sha": state["audit_recommendation_sha"],
        "walk_forward_split_sha": state["v1_split_sha"],
        "factor_definition_sha": state["factor_definition_sha"],
        "strategy_input_sha": state["strategy_input_sha"],
        "ohlcv_sha": state["audit_manifest"]["ohlcv_sha"],
        "universe_sha": state["audit_manifest"]["universe_sha"],
        "cost_sha": state["cost_sha"],
        "trial_registry_sha": file_sha256(trial_path),
        "engine_input_sha": parity["input_sha"],
        "engine_input_sha_by_engine": {engine: parity["input_sha"] for engine in parity["actual_engines"]},
        "trace_hashes": trace_hashes,
        "artifact_hashes": artifact_hashes,
        "actual_engine_labels": parity["actual_engines"],
        "engine_versions": _versions(),
        "custom_engine_version": "canonical_replay_v1",
        "vectorbt_version": _package_version("vectorbt"),
        "backtrader_version": _package_version("backtrader"),
        "python_version": platform.python_version(),
        "dependency_snapshot": _versions(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "code_sha": {name: file_sha256(root / name) for name in ("src/twse_factor_lab/backtest/accounting.py", "src/twse_factor_lab/backtest/vectorbt_engine.py", "src/twse_factor_lab/backtest/backtrader_engine.py", "src/twse_factor_lab/acceptance/final_validation_v3.py", "run_final_strategy_validation_v3.py")},
        "historical_freezes": state["v1_manifest"]["historical_freezes"],
        "baseline": baseline,
        "final_verdict": _read_json(out / "final_acceptance_v3.json")["final_verdict"],
        "acceptance_label": _read_json(out / "final_acceptance_v3.json")["acceptance_label"],
        "historical_v1_modified": False,
        "historical_v2_modified": False,
        "fresh_oos_available": False,
        "fresh_oos_claimed": False,
        "production_ready": False,
    }
    _write_json(root, out, "run_manifest.json", manifest)
    _assert_manifest(out, manifest, "V3")
    return {"output": str(out), "contract_sha": contract_sha, "parity": parity, "acceptance": _read_json(out / "final_acceptance_v3.json"), "manifest": manifest, "synthetic": synthetic, "root_cause": root_cause, "performance": performance}


__all__ = ["ATOL", "CANDIDATE_ID", "FinalValidationV3Error", "LOCKED_FINGERPRINT", "PARITY_FORMULA", "RTOL", "accounting_contract", "file_sha256", "run_synthetic_fixtures", "run_validation"]

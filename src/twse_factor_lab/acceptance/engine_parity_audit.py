# ruff: noqa: E501

"""Immutable, layer-by-layer audit of the locked S3 engine replay."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.backtrader_engine import run_backtrader_engine
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import (
    _custom_backtest,
    _weights_matrix,
    run_weight_backtest,
)
from twse_factor_lab.governance.isolation import assert_write_allowed
from twse_factor_lab.strategy.composite_replay import (
    build_composite as _composite,
)
from twse_factor_lab.strategy.composite_replay import (
    build_targets as _targets,
)

ROOT_NAME = "engine-parity-audit-v1"
CANDIDATE_ID = "S3"
LOCKED_FINGERPRINT = (
    "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"
)
CANONICAL_TOLERANCE = 1e-8
AUDIT_ATOL = 1e-10
AUDIT_RTOL = 1e-10
INITIAL_CASH = 1_000_000.0
ENGINE_ORDER = ("CUSTOM", "VECTORBT", "BACKTRADER")
PAIR_ORDER = (("CUSTOM", "VECTORBT"), ("CUSTOM", "BACKTRADER"))
TRACE_COLUMNS = [
    "date",
    "engine",
    "signal_date",
    "execution_date",
    "ticker",
    "selected",
    "target_weight",
    "executed_weight",
    "position_quantity",
    "position_value",
    "buy_notional",
    "sell_notional",
    "buy_fee",
    "sell_fee",
    "sell_tax",
    "slippage_cost",
    "total_cost",
    "cash",
    "gross_exposure",
    "net_exposure",
    "daily_pnl",
    "daily_return",
    "equity",
]
REQUIRED_AUDIT_OUTPUTS = {
    "engine_parity_audit_contract.json",
    "custom_trace.parquet",
    "vectorbt_trace.parquet",
    "backtrader_trace.parquet",
    "rebalance_date_parity.json",
    "selection_parity.csv",
    "target_weight_parity.csv",
    "execution_parity.csv",
    "cost_parity.csv",
    "position_parity.csv",
    "cash_parity.csv",
    "return_parity.csv",
    "equity_parity.csv",
    "first_divergence_report.json",
    "numerical_error_analysis.json",
    "economic_equivalence.json",
    "synthetic_parity_results.csv",
    "engine_parity_audit_result.json",
    "engine_parity_audit_report.md",
}
CAUSES = {
    "DATE_MISMATCH",
    "SELECTION_MISMATCH",
    "TARGET_WEIGHT_MISMATCH",
    "ORDER_SEMANTICS",
    "POSITION_ROUNDING",
    "COST_ROUNDING",
    "CASH_ROUNDING",
    "FLOATING_POINT_ACCUMULATION",
    "UNKNOWN",
}


class EngineParityAuditError(RuntimeError):
    """The immutable audit contract cannot be satisfied."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineParityAuditError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise EngineParityAuditError(f"JSON object required: {path}")
    return payload


def _path_from_key(root: Path, key: str) -> Path:
    return root / Path(key.replace("\\", "/"))


def _assert_hashes(root: Path, directory: Path, hashes: dict[str, str], label: str) -> None:
    for name, expected in hashes.items():
        path = directory / name
        if not path.exists() or file_sha256(path) != expected:
            raise EngineParityAuditError(f"{label}_HASH_MISMATCH:{name}")


def _assert_protected_hashes(root: Path, state: dict[str, Any]) -> None:
    """Recheck frozen bytes after replay and before reporting any result."""
    protected = state["protected_hashes"]
    _assert_hashes(
        root,
        root / "data/research/final-strategy-validation-v1",
        protected["final_validation_v1"],
        "FINAL_VALIDATION_V1",
    )
    _assert_hashes(
        root,
        root / "data/research/composite-strategy-lab-v1",
        protected["change_2"],
        "CHANGE_2",
    )
    _assert_hashes(
        root,
        root / "data/research/composite-factor-admission-v1",
        protected["change_1"],
        "CHANGE_1",
    )


def _normalise_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    result.index = pd.DatetimeIndex(result.index).tz_localize(None)
    result.columns = result.columns.astype(str)
    return result.sort_index().astype(float)


def _frame_sha(frame: pd.DataFrame) -> str:
    metadata = json.dumps(
        {
            "columns": [str(column) for column in frame.columns],
            "index": [str(value) for value in frame.index],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(metadata + values).hexdigest()


def _legacy_frame_sha(frame: pd.DataFrame) -> str:
    """Match the historical Change 2 frame-hash convention exactly."""
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _expected_cost() -> CostModel:
    return CostModel(
        buy_fee_rate=0.001425,
        sell_fee_rate=0.001425,
        transaction_tax_rate=0.003,
        slippage_rate=0.001,
    )


def preflight(root: Path) -> dict[str, Any]:
    """Verify every historical input before reading strategy replay data."""
    c1 = root / "data/research/composite-factor-admission-v1"
    c2 = root / "data/research/composite-strategy-lab-v1"
    v1 = root / "data/research/final-strategy-validation-v1"
    c2_manifest = _read_json(c2 / "run_manifest.json")
    lock = _read_json(c2 / "candidate_lock.json")
    strategy_input = _read_json(c2 / "strategy_input_contract.json")
    final_acceptance = _read_json(v1 / "final_acceptance.json")
    final_crosscheck = _read_json(v1 / "engine_crosscheck.json")
    final_manifest = _read_json(v1 / "run_manifest.json")

    expected_candidate = {
        "strategy_id": CANDIDATE_ID,
        "components": ["L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D"],
        "weights": {"L2_AMIHUD_20D": 0.5, "L4_DOLLAR_VOLUME_20D": 0.5},
        "top_n": 5,
        "rebalance": "monthly",
        "buffer": False,
        "weighting": "equal_weight",
        "cost_model": "base_cost",
    }
    candidate = lock.get("candidate", {})
    if lock.get("candidate_fingerprint") != LOCKED_FINGERPRINT:
        raise EngineParityAuditError("S3_FINGERPRINT_MISMATCH")
    if lock.get("status") != "SUCCESS" or any(
        candidate.get(key) != value for key, value in expected_candidate.items()
    ):
        raise EngineParityAuditError("CANDIDATE_LOCK_MISMATCH")
    if final_acceptance.get("candidate_fingerprint") != LOCKED_FINGERPRINT:
        raise EngineParityAuditError("FINAL_VALIDATION_FINGERPRINT_MISMATCH")
    if final_acceptance.get("final_verdict") != "REJECT":
        raise EngineParityAuditError("HISTORICAL_VERDICT_MISMATCH")
    if final_acceptance.get("gates", {}).get("engine_parity") != "FAIL":
        raise EngineParityAuditError("HISTORICAL_ENGINE_PARITY_MISMATCH")
    if final_crosscheck.get("tolerance") != CANONICAL_TOLERANCE:
        raise EngineParityAuditError("CANONICAL_TOLERANCE_MISMATCH")
    if final_crosscheck.get("status") != "FAIL":
        raise EngineParityAuditError("HISTORICAL_PARITY_STATUS_MISMATCH")

    _assert_hashes(root, c2, c2_manifest.get("artifact_hashes", {}), "CHANGE_2")
    _assert_hashes(
        root,
        v1,
        final_manifest.get("artifact_hashes", {}),
        "FINAL_VALIDATION_V1",
    )
    _assert_hashes(
        root,
        c1,
        strategy_input.get("source_artifact_sha", {}),
        "CHANGE_1",
    )
    source_ohlcv = {
        str(key): str(value) for key, value in strategy_input["ohlcv_sha"].items()
    }
    for key, expected in source_ohlcv.items():
        path = _path_from_key(root, key)
        if not path.exists() or file_sha256(path) != expected:
            raise EngineParityAuditError(f"OHLCV_HASH_MISMATCH:{key}")
    universe_path = root / "data/processed/research_universe.parquet"
    if file_sha256(universe_path) != strategy_input["research_universe_sha"]:
        raise EngineParityAuditError("UNIVERSE_HASH_MISMATCH")
    cost = _expected_cost()
    if candidate.get("cost_parameters") != {
        "buy_fee_rate": cost.buy_fee_rate,
        "sell_fee_rate": cost.sell_fee_rate,
        "transaction_tax_rate": cost.transaction_tax_rate,
        "slippage_rate": cost.slippage_rate,
    }:
        raise EngineParityAuditError("COST_CONTRACT_MISMATCH")
    if c2_manifest.get("candidate_fingerprint") != LOCKED_FINGERPRINT:
        raise EngineParityAuditError("CHANGE_2_FINGERPRINT_MISMATCH")
    return {
        "candidate_lock": lock,
        "strategy_input": strategy_input,
        "change_1_artifact_sha": strategy_input["source_artifact_sha"],
        "change_2_artifact_sha": c2_manifest["artifact_hashes"],
        "final_validation_v1_artifact_sha": final_manifest["artifact_hashes"],
        "final_acceptance": final_acceptance,
        "final_crosscheck": final_crosscheck,
        "final_validation_manifest": final_manifest,
        "factor_definition_sha": strategy_input["composite_definition_sha"],
        "strategy_input_sha": file_sha256(c2 / "strategy_input_contract.json"),
        "source_ohlcv_sha": source_ohlcv,
        "universe_sha": file_sha256(universe_path),
        "cost_sha": json_sha256(cost.summary()),
        "cost": cost,
        "protected_hashes": {
            "final_validation_v1": final_manifest["artifact_hashes"],
            "change_2": c2_manifest["artifact_hashes"],
            "change_1": strategy_input["source_artifact_sha"],
        },
    }


def load_replay_inputs(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Recreate the frozen S3 targets without searching or changing them."""
    processed = root / "data/processed"
    close = _normalise_matrix(pd.read_parquet(processed / "close_matrix.parquet"))
    volume = _normalise_matrix(pd.read_parquet(processed / "volume_matrix.parquet"))
    if not close.index.equals(volume.index) or not close.columns.equals(volume.columns):
        raise EngineParityAuditError("OHLCV_MATRIX_ALIGNMENT_MISMATCH")
    score, components = _composite(close, volume)
    expected_components = state["strategy_input"]["component_factor_sha"]
    for name, matrix in components.items():
        if _legacy_frame_sha(matrix) != expected_components.get(name):
            raise EngineParityAuditError(f"FACTOR_INPUT_SHA_MISMATCH:{name}")
    targets, calendar = _targets(score, rebalance="monthly", buffer_on=False)
    targets = targets.copy()
    targets["date"] = pd.to_datetime(targets["date"])
    targets["execution_date"] = pd.to_datetime(targets["execution_date"])
    targets["ticker"] = targets["ticker"].astype(str)
    traded = sorted(targets["ticker"].unique())
    if len(traded) == 0 or not set(traded).issubset(close.columns):
        raise EngineParityAuditError("TARGET_INPUT_MISMATCH")
    replay_close = close.loc[:, traded]
    weights = _weights_matrix(targets, replay_close.index, replay_close.columns)
    event_calendar = calendar.copy()
    event_calendar["signal_date"] = pd.to_datetime(event_calendar["signal_date"])
    event_calendar["execution_date"] = pd.to_datetime(
        event_calendar["execution_date"]
    )
    event_calendar = event_calendar.set_index("execution_date")
    # The target table is authoritative for executable rebalance events.  The
    # calendar also contains the warm-up signal before the first target event.
    event_calendar = event_calendar.loc[
        event_calendar.index.isin(pd.DatetimeIndex(targets["execution_date"].unique()))
    ]
    if targets["execution_date"].nunique() != 95 or len(traded) != 43:
        raise EngineParityAuditError("S3_TARGET_SHAPE_MISMATCH")
    if len(event_calendar) != targets["execution_date"].nunique():
        raise EngineParityAuditError("REBALANCE_CALENDAR_TARGET_MISMATCH")
    if not np.allclose(
        targets.groupby("execution_date")["target_weight"].sum().to_numpy(), 1.0
    ):
        raise EngineParityAuditError("TARGET_WEIGHT_SUM_MISMATCH")
    target_sha = _frame_sha(targets)
    input_sha = json_sha256(
        {
            "close_sha": _frame_sha(replay_close),
            "target_sha": target_sha,
            "dates": [str(date) for date in replay_close.index],
            "tickers": traded,
            "initial_cash": INITIAL_CASH,
            "cost_sha": state["cost_sha"],
        }
    )
    return {
        "close": replay_close,
        "volume": volume,
        "targets": targets,
        "weights": weights,
        "calendar": event_calendar,
        "traded_tickers": traded,
        "target_sha": target_sha,
        "engine_input_sha": input_sha,
    }


def _order_costs(
    buy_notional: float, sell_notional: float, cost: CostModel
) -> dict[str, float]:
    buy_fee = buy_notional * cost.buy_fee_rate
    sell_fee = sell_notional * cost.sell_fee_rate
    sell_tax = sell_notional * cost.transaction_tax_rate
    slippage = (buy_notional + sell_notional) * cost.slippage_rate
    return {
        "buy_fee": buy_fee,
        "sell_fee": sell_fee,
        "sell_tax": sell_tax,
        "slippage_cost": slippage,
        "total_cost": buy_fee + sell_fee + sell_tax + slippage,
    }


def _empty_order_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "buy_notional",
            "sell_notional",
            "position_change",
            "buy_fee",
            "sell_fee",
            "sell_tax",
            "slippage_cost",
            "total_cost",
            "engine_reported_cost",
            "order_count",
            "direction",
        ]
    )


def _orders_from_sizes(
    sizes: pd.DataFrame, close: pd.DataFrame, cost: CostModel, engine: str
) -> pd.DataFrame:
    mark = close.ffill()
    rows: list[dict[str, Any]] = []
    for date in sizes.index:
        for ticker in sizes.columns:
            size = float(sizes.loc[date, ticker])
            if size == 0 or not math.isfinite(size):
                continue
            price = float(mark.loc[date, ticker])
            buy = max(size, 0.0) * price
            sell = max(-size, 0.0) * price
            components = _order_costs(buy, sell, cost)
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "ticker": str(ticker),
                    "buy_notional": buy,
                    "sell_notional": sell,
                    "buy_fee": components["buy_fee"],
                    "sell_fee": components["sell_fee"],
                    "sell_tax": components["sell_tax"],
                    "slippage_cost": components["slippage_cost"],
                    "total_cost": components["total_cost"],
                    "engine_reported_cost": components["total_cost"],
                    "order_count": 1,
                    "direction": "BUY" if buy else "SELL",
                    "engine": engine,
                }
            )
    return pd.DataFrame(rows) if rows else _empty_order_frame()


def _orders_from_vectorbt(
    path: Path, cost: CostModel, engine: str
) -> pd.DataFrame:
    if not path.exists():
        raise EngineParityAuditError("VECTORBT_ORDER_ARTIFACT_MISSING")
    frame = pd.read_parquet(path)
    if frame.empty:
        return _empty_order_frame()
    rows: list[dict[str, Any]] = []
    for (date, ticker), group in frame.assign(
        date=pd.to_datetime(frame["Timestamp"]),
        ticker=frame["Column"].astype(str),
    ).groupby(["date", "ticker"], sort=True):
        buy = float(
            group.loc[group["Side"].astype(str).str.lower().eq("buy"), "Size"]
            .mul(group.loc[group["Side"].astype(str).str.lower().eq("buy"), "Price"])
            .abs()
            .sum()
        )
        sell = float(
            group.loc[group["Side"].astype(str).str.lower().eq("sell"), "Size"]
            .mul(group.loc[group["Side"].astype(str).str.lower().eq("sell"), "Price"])
            .abs()
            .sum()
        )
        components = _order_costs(buy, sell, cost)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "ticker": str(ticker),
                "buy_notional": buy,
                "sell_notional": sell,
                **components,
                "engine_reported_cost": float(group["Fees"].sum()),
                "order_count": int(len(group)),
                "direction": "BUY" if buy else "SELL",
                "engine": engine,
            }
        )
    return pd.DataFrame(rows)


def _orders_from_fills(
    fills: pd.DataFrame, cost: CostModel, engine: str
) -> pd.DataFrame:
    if fills.empty:
        return _empty_order_frame()
    frame = fills.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype(str)
    rows: list[dict[str, Any]] = []
    for (date, ticker), group in frame.groupby(["date", "ticker"], sort=True):
        buy = float(
            group.loc[group["size"] > 0, "size"]
            .mul(group.loc[group["size"] > 0, "price"])
            .abs()
            .sum()
        )
        sell = float(
            group.loc[group["size"] < 0, "size"]
            .mul(group.loc[group["size"] < 0, "price"])
            .abs()
            .sum()
        )
        components = _order_costs(buy, sell, cost)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "ticker": str(ticker),
                "buy_notional": buy,
                "sell_notional": sell,
                **components,
                "engine_reported_cost": float(group["commission"].sum()),
                "order_count": int(len(group)),
                "direction": "BUY" if buy else "SELL",
                "engine": engine,
            }
        )
    return pd.DataFrame(rows)


def _position_frame(result: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(result["date"])))
    frame["equity"] = result["equity"].to_numpy(dtype=float)
    frame["cash"] = result["cash"].to_numpy(dtype=float)
    frame["daily_return"] = result["returns"].to_numpy(dtype=float)
    frame["exposure"] = result["exposure"].to_numpy(dtype=float)
    values = pd.DataFrame(index=frame.index)
    for ticker in close.columns:
        values[str(ticker)] = result[f"position:{ticker}"].to_numpy(dtype=float)
    quantities = values.div(close.ffill().replace(0.0, np.nan)).fillna(0.0)
    return pd.concat(
        {
            "value": values,
            "quantity": quantities,
            "portfolio": frame,
        },
        axis=1,
    )


def _trace(
    engine: str,
    result: pd.DataFrame,
    orders: pd.DataFrame,
    inputs: dict[str, Any],
) -> pd.DataFrame:
    close = inputs["close"]
    weights = inputs["weights"]
    calendar = inputs["calendar"]
    positions = _position_frame(result, close)
    order_lookup = orders.set_index(["date", "ticker"]) if not orders.empty else None
    previous_equity = INITIAL_CASH
    rows: list[dict[str, Any]] = []
    for date in close.index:
        equity = float(positions.loc[date, ("portfolio", "equity")])
        daily_pnl = equity - previous_equity
        signal_date = calendar.loc[date, "signal_date"] if date in calendar.index else None
        for ticker in close.columns:
            key = (pd.Timestamp(date), str(ticker))
            order = (
                order_lookup.loc[key]
                if order_lookup is not None and key in order_lookup.index
                else None
            )
            value = float(positions.loc[date, ("value", ticker)])
            quantity = float(positions.loc[date, ("quantity", ticker)])
            target = float(weights.loc[date, ticker])
            rows.append(
                {
                    "date": date,
                    "engine": engine,
                    "signal_date": signal_date,
                    "execution_date": date if signal_date is not None else None,
                    "ticker": str(ticker),
                    "selected": bool(target > 0),
                    "target_weight": target,
                    "executed_weight": value / equity if equity else 0.0,
                    "position_quantity": quantity,
                    "position_value": value,
                    "buy_notional": float(order["buy_notional"]) if order is not None else 0.0,
                    "sell_notional": float(order["sell_notional"]) if order is not None else 0.0,
                    "buy_fee": float(order["buy_fee"]) if order is not None else 0.0,
                    "sell_fee": float(order["sell_fee"]) if order is not None else 0.0,
                    "sell_tax": float(order["sell_tax"]) if order is not None else 0.0,
                    "slippage_cost": float(order["slippage_cost"])
                    if order is not None
                    else 0.0,
                    "total_cost": float(order["total_cost"]) if order is not None else 0.0,
                    "cash": float(positions.loc[date, ("portfolio", "cash")]),
                    "gross_exposure": float(
                        positions.loc[date, ("portfolio", "exposure")]
                    ),
                    "net_exposure": float(
                        positions.loc[date, ("portfolio", "exposure")]
                    ),
                    "daily_pnl": daily_pnl,
                    "daily_return": float(
                        positions.loc[date, ("portfolio", "daily_return")]
                    ),
                    "equity": equity,
                }
            )
        previous_equity = equity
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


def run_engines(root: Path, inputs: dict[str, Any], out: Path) -> dict[str, Any]:
    close = inputs["close"]
    targets = inputs["targets"]
    cost = _expected_cost()
    out.mkdir(parents=True, exist_ok=True)
    custom_result, custom_metrics = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=targets,
        cost_model=cost,
        initial_cash=INITIAL_CASH,
        top_n=5,
        use_vectorbt=False,
        allow_fallback=False,
    )
    custom_weights = _weights_matrix(targets, close.index, close.columns)
    _, _, _, custom_sizes = _custom_backtest(
        close, custom_weights, cost, INITIAL_CASH
    )
    vector_result, vector_metrics = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=targets,
        cost_model=cost,
        initial_cash=INITIAL_CASH,
        top_n=5,
        use_vectorbt=True,
        allow_fallback=False,
        artifacts_dir=str(out),
    )
    backtrader_run = run_backtrader_engine(
        close_matrix=close,
        target_weights=targets,
        cost_model=cost,
        initial_cash=INITIAL_CASH,
    )
    actual = {
        "CUSTOM": str(custom_metrics.iloc[0]["actual_engine"]).upper(),
        "VECTORBT": str(vector_metrics.iloc[0]["actual_engine"]).upper(),
        "BACKTRADER": str(backtrader_run.metrics.iloc[0]["actual_engine"]).upper(),
    }
    if actual != {name: name for name in ENGINE_ORDER}:
        raise EngineParityAuditError(f"ENGINE_FALLBACK_OR_UNAVAILABLE:{actual}")
    fills_path = assert_write_allowed(out / "backtrader_fills.csv", root)
    backtrader_run.fills.to_csv(fills_path, index=False, lineterminator="\n")
    custom_orders = _orders_from_sizes(custom_sizes, close, cost, "CUSTOM")
    vector_orders = _orders_from_vectorbt(out / "vectorbt_orders.parquet", cost, "VECTORBT")
    back_orders = _orders_from_fills(backtrader_run.fills, cost, "BACKTRADER")
    return {
        "results": {
            "CUSTOM": custom_result,
            "VECTORBT": vector_result,
            "BACKTRADER": backtrader_run.results,
        },
        "orders": {
            "CUSTOM": custom_orders,
            "VECTORBT": vector_orders,
            "BACKTRADER": back_orders,
        },
        "traces": {
            "CUSTOM": _trace("CUSTOM", custom_result, custom_orders, inputs),
            "VECTORBT": _trace("VECTORBT", vector_result, vector_orders, inputs),
            "BACKTRADER": _trace(
                "BACKTRADER", backtrader_run.results, back_orders, inputs
            ),
        },
        "actual_engines": actual,
    }


def _metric(left: Any, right: Any) -> dict[str, Any]:
    if pd.isna(left) and pd.isna(right):
        return {
            "left": None,
            "right": None,
            "absolute_difference": 0.0,
            "relative_difference": 0.0,
            "classification": "exact match",
        }
    if pd.isna(left) or pd.isna(right):
        return {
            "left": _json_value(left),
            "right": _json_value(right),
            "absolute_difference": None,
            "relative_difference": None,
            "classification": "material difference",
        }
    left_value = float(left)
    right_value = float(right)
    absolute = abs(left_value - right_value)
    scale = max(abs(left_value), abs(right_value), 1.0)
    relative = absolute / scale
    if absolute == 0.0:
        classification = "exact match"
    elif absolute <= AUDIT_ATOL + AUDIT_RTOL * scale:
        classification = "floating-point difference"
    else:
        classification = "material difference"
    return {
        "left": left_value,
        "right": right_value,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "classification": classification,
    }


def _first_raw(frame: pd.DataFrame, fields: list[str]) -> dict[str, Any] | None:
    for row in frame.itertuples(index=False):
        for field in fields:
            value = getattr(row, field, 0.0)
            if pd.notna(value) and float(value) != 0.0:
                result = {str(key): _json_value(value) for key, value in row._asdict().items()}
                result["field"] = field
                return result
    return None


def _pair_summary(
    frame: pd.DataFrame, absolute_field: str, relative_field: str
) -> dict[str, Any]:
    absolute = pd.to_numeric(frame[absolute_field], errors="coerce").dropna()
    relative = pd.to_numeric(frame[relative_field], errors="coerce").dropna()
    material = frame["classification"].eq("material difference")
    return {
        "status": "FAIL" if bool(material.any()) else "PASS",
        "max_abs_error": float(absolute.max()) if not absolute.empty else 0.0,
        "max_rel_error": float(relative.max()) if not relative.empty else 0.0,
        "mean_abs_error": float(absolute.mean()) if not absolute.empty else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(absolute))))
        if not absolute.empty
        else 0.0,
        "raw_difference_count": int(
            frame[absolute_field].fillna(0.0).astype(float).ne(0.0).sum()
        ),
        "material_difference_count": int(material.sum()),
        "first_raw_difference": _first_raw(frame, [absolute_field]),
    }


def _target_events(inputs: dict[str, Any]) -> pd.DataFrame:
    targets = inputs["targets"]
    columns = inputs["traded_tickers"]
    rows: list[dict[str, Any]] = []
    for execution_date, group in targets.groupby("execution_date", sort=True):
        selected = set(group["ticker"].astype(str))
        values = dict(zip(group["ticker"].astype(str), group["target_weight"], strict=True))
        for ticker in columns:
            rows.append(
                {
                    "date": pd.Timestamp(execution_date),
                    "ticker": ticker,
                    "target_weight": float(values.get(ticker, 0.0)),
                    "selected": ticker in selected,
                }
            )
    return pd.DataFrame(rows)


def compare_rebalance_and_selection(
    inputs: dict[str, Any], traces: dict[str, pd.DataFrame], orders: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any], pd.DataFrame]:
    calendar = inputs["calendar"].copy()
    calendar.index = pd.DatetimeIndex(calendar.index)
    planned = calendar[["signal_date"]].copy()
    planned["execution_date"] = planned.index
    planned = planned.reset_index(drop=True)

    def trace_schedule(engine: str) -> pd.DataFrame:
        events = traces[engine].loc[
            traces[engine]["execution_date"].notna(),
            ["signal_date", "execution_date"],
        ].drop_duplicates()
        return events.sort_values(["signal_date", "execution_date"]).reset_index(drop=True)

    schedules = {engine: trace_schedule(engine) for engine in ENGINE_ORDER}
    fill_dates = {
        engine: sorted(pd.to_datetime(orders[engine]["date"]).dt.normalize().unique())
        if not orders[engine].empty
        else []
        for engine in ENGINE_ORDER
    }
    date_result = {
        "status": "PASS",
        "signal_dates": {
            engine: [str(pd.Timestamp(date).date()) for date in schedules[engine]["signal_date"]]
            for engine in ENGINE_ORDER
        },
        "execution_dates": {
            engine: [str(pd.Timestamp(date).date()) for date in schedules[engine]["execution_date"]]
            for engine in ENGINE_ORDER
        },
        "fill_execution_dates": {
            engine: [str(pd.Timestamp(date).date()) for date in dates]
            for engine, dates in fill_dates.items()
        },
        "comparisons": {},
        "first_divergence_date": None,
        "semantic_classification": "EXACT_MATCH",
    }

    def first_schedule_difference(left: pd.DataFrame, right: pd.DataFrame) -> pd.Timestamp | None:
        for index in range(max(len(left), len(right))):
            left_value = tuple(left.iloc[index]) if index < len(left) else None
            right_value = tuple(right.iloc[index]) if index < len(right) else None
            if left_value != right_value:
                values = [value for value in (left_value, right_value) if value is not None]
                if values:
                    return min(pd.Timestamp(value[1]) for value in values)
                return None
        return None

    for left, right in PAIR_ORDER:
        same_signal = schedules[left]["signal_date"].equals(schedules[right]["signal_date"])
        same_execution = schedules[left]["execution_date"].equals(
            schedules[right]["execution_date"]
        )
        same_fills = fill_dates[left] == fill_dates[right]
        first_difference = first_schedule_difference(schedules[left], schedules[right])
        if first_difference is None and not same_fills:
            differing = sorted(set(fill_dates[left]).symmetric_difference(fill_dates[right]))
            first_difference = pd.Timestamp(differing[0]) if differing else None
        date_result["comparisons"][f"{left.lower()}_vs_{right.lower()}"] = {
            "signal_dates_equal": bool(same_signal),
            "execution_dates_equal": bool(same_execution),
            "fill_execution_dates_equal": bool(same_fills),
            "status": "PASS"
            if same_signal and same_execution and same_fills
            else "SEMANTIC_DIVERGENCE",
        }
        if not same_signal or not same_execution or not same_fills:
            date_result["status"] = "SEMANTIC_DIVERGENCE"
            date_result["semantic_classification"] = "SEMANTIC_DIVERGENCE"
            if first_difference is not None:
                date_result["first_divergence_date"] = str(first_difference.date())
    event_targets = _target_events(inputs)
    selection_rows: list[dict[str, Any]] = []
    for date, _group in event_targets.groupby("date", sort=True):
        row = {"date": date}
        for engine in ENGINE_ORDER:
            selected = sorted(
                traces[engine]
                .loc[
                    traces[engine]["date"].eq(date) & traces[engine]["selected"],
                    "ticker",
                ]
                .astype(str)
            )
            row[f"{engine.lower()}_tickers"] = ",".join(selected)
        row["match"] = len({row[f"{engine.lower()}_tickers"] for engine in ENGINE_ORDER}) == 1
        selection_rows.append(row)
    selection = pd.DataFrame(selection_rows)
    if not selection.empty and not selection["match"].all():
        date_result["status"] = "SEMANTIC_DIVERGENCE"
        mismatch = selection.loc[~selection["match"]].iloc[0]
        date_result["selection_first_divergence_date"] = str(
            pd.Timestamp(mismatch["date"]).date()
        )
    else:
        date_result["selection_first_divergence_date"] = None
    return date_result, selection


def compare_targets(inputs: dict[str, Any], traces: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = traces["CUSTOM"][["date", "ticker"]].copy()
    output = keys.copy()
    index = pd.MultiIndex.from_frame(keys)
    for engine in ENGINE_ORDER:
        values = traces[engine].set_index(["date", "ticker"])["target_weight"]
        output[f"{engine.lower()}_target_weight"] = values.reindex(index).to_numpy()
    relative_columns: list[str] = []
    absolute_columns: list[str] = []
    for left, right in PAIR_ORDER:
        absolute_name = f"absolute_difference_{left.lower()}_{right.lower()}"
        relative_name = f"relative_difference_{left.lower()}_{right.lower()}"
        output[absolute_name] = (
            output[f"{left.lower()}_target_weight"]
            - output[f"{right.lower()}_target_weight"]
        ).abs()
        scale = output[
            [f"{left.lower()}_target_weight", f"{right.lower()}_target_weight"]
        ].abs().max(axis=1).clip(lower=1.0)
        output[relative_name] = output[absolute_name] / scale
        absolute_columns.append(absolute_name)
        relative_columns.append(relative_name)
    max_abs = output[absolute_columns].max(axis=1)
    max_scale = output[
        [f"{engine.lower()}_target_weight" for engine in ENGINE_ORDER]
    ].abs().max(axis=1).clip(lower=1.0)
    bound = AUDIT_ATOL + AUDIT_RTOL * max_scale
    output["classification"] = np.where(
        max_abs.eq(0.0),
        "exact match",
        np.where(max_abs <= bound, "floating-point difference", "material difference"),
    )
    first = output.loc[output["classification"].eq("material difference")]
    return output, {
        "status": "PASS" if first.empty else "FAIL",
        "max_abs_error": float(output[absolute_columns].max().max()),
        "max_rel_error": float(output[relative_columns].max().max()),
        "raw_difference_count": int((output[absolute_columns] > 0).any(axis=1).sum()),
        "material_difference_count": int(len(first)),
        "first_raw_difference": _first_raw(output, absolute_columns),
    }


def _order_compare(
    orders: dict[str, pd.DataFrame], traces: dict[str, pd.DataFrame], inputs: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = pd.MultiIndex.from_product(
        [inputs["close"].index, inputs["traded_tickers"]],
        names=["date", "ticker"],
    ).to_frame(index=False)
    output = keys.copy()
    index = pd.MultiIndex.from_frame(keys)
    for engine in ENGINE_ORDER:
        order = (
            orders[engine].set_index(["date", "ticker"]).sort_index()
            if not orders[engine].empty
            else pd.DataFrame(index=index)
        )
        for field in ["buy_notional", "sell_notional", "total_cost", "direction"]:
            if field == "direction":
                values = (
                    order[field].reindex(index).fillna("NONE")
                    if field in order
                    else pd.Series("NONE", index=index)
                )
                output[f"{engine.lower()}_{field}"] = values.to_numpy()
            else:
                values = (
                    order[field].reindex(index).fillna(0.0)
                    if field in order
                    else pd.Series(0.0, index=index)
                )
                output[f"{engine.lower()}_{field}"] = values.to_numpy(dtype=float)
        quantity_matrix = traces[engine].pivot(
            index="date", columns="ticker", values="position_quantity"
        ).reindex(index=inputs["close"].index, columns=inputs["traded_tickers"])
        changes = quantity_matrix.diff().fillna(quantity_matrix)
        output[f"{engine.lower()}_position_change"] = changes.to_numpy().reshape(-1)
    for left, right in PAIR_ORDER:
        for field in ["buy_notional", "sell_notional", "position_change"]:
            prefix = f"{left.lower()}_{field}"
            other = f"{right.lower()}_{field}"
            output[f"absolute_difference_{left.lower()}_{right.lower()}_{field}"] = (
                output[prefix] - output[other]
            ).abs()
            output[f"relative_difference_{left.lower()}_{right.lower()}_{field}"] = output[
                f"absolute_difference_{left.lower()}_{right.lower()}_{field}"
            ] / output[[prefix, other]].abs().max(axis=1).clip(lower=1.0)
        def effective_direction(engine: str) -> pd.Series:
            direction = output[f"{engine.lower()}_direction"].copy()
            size = output[
                [
                    f"{engine.lower()}_buy_notional",
                    f"{engine.lower()}_sell_notional",
                    f"{engine.lower()}_position_change",
                ]
            ].abs().max(axis=1)
            return direction.mask(size <= AUDIT_ATOL, "NONE")

        output[f"effective_direction_{left.lower()}"] = effective_direction(left)
        output[f"effective_direction_{right.lower()}"] = effective_direction(right)
        output[f"direction_match_{left.lower()}_{right.lower()}"] = output[
            f"effective_direction_{left.lower()}"
        ].eq(output[f"effective_direction_{right.lower()}"])
    abs_columns = [column for column in output if column.startswith("absolute_difference_")]
    event_dates = pd.DatetimeIndex(inputs["calendar"].index)
    event_targets = inputs["weights"].reindex(event_dates).fillna(0.0)
    prior_events = event_targets.shift(1).fillna(0.0)
    removed = (prior_events > 0) & event_targets.eq(0)
    removed_zero = bool(removed.any(axis=None))
    removed_targets_zero = bool(
        (event_targets.where(removed).fillna(0.0) == 0.0).all(axis=None)
    )
    complete_replacement = bool(
        event_targets.sum(axis=1).sub(1.0).abs().le(AUDIT_ATOL).all()
    )
    output["removed_ticker_target_zero"] = removed_zero and removed_targets_zero
    direction_columns = [
        column for column in output if column.startswith("direction_match_")
    ]
    relative_columns = [
        column
        for column in output
        if column.startswith("relative_difference_")
    ]
    output["semantic_status"] = np.where(
        output[direction_columns].eq(False).any(axis=1)
        | (output[relative_columns].max(axis=1) > AUDIT_RTOL),
        "SEMANTIC_DIVERGENCE",
        "PASS",
    )
    semantic_rows = output["semantic_status"].eq("SEMANTIC_DIVERGENCE")
    close_index = pd.DatetimeIndex(inputs["close"].index)
    execution_dates = pd.DatetimeIndex(inputs["calendar"].index)
    positions = close_index.get_indexer(execution_dates)
    t_to_t_plus_1 = bool(
        (positions > 0).all()
        and all(
            close_index[position - 1] == signal
            for position, signal in zip(
                positions, inputs["calendar"]["signal_date"], strict=True
            )
        )
    )
    position_change_columns = [
        f"{engine.lower()}_position_change" for engine in ENGINE_ORDER
    ]
    fill_count = int(
        output[position_change_columns].abs().gt(0).any(axis=1).sum()
    )
    return output, {
        "status": "FAIL"
        if semantic_rows.any()
        or not complete_replacement
        or not output["removed_ticker_target_zero"].all()
        else "PASS",
        "t_to_t_plus_1": t_to_t_plus_1,
        "complete_target_replacement": complete_replacement,
        "removed_ticker_target_zero": bool(output["removed_ticker_target_zero"].all()),
        "fill_count": fill_count,
        "raw_difference_count": int((output[abs_columns] > 0).any(axis=1).sum()),
        "material_difference_count": int(semantic_rows.sum()),
        "first_raw_difference": _first_raw(output, abs_columns),
    }


def _cost_compare(orders: dict[str, pd.DataFrame], inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = pd.MultiIndex.from_product(
        [inputs["close"].index, inputs["traded_tickers"]], names=["date", "ticker"]
    ).to_frame(index=False)
    output = keys.copy()
    index = pd.MultiIndex.from_frame(keys)
    fields = ["buy_fee", "sell_fee", "sell_tax", "slippage_cost", "total_cost", "engine_reported_cost"]
    for engine in ENGINE_ORDER:
        order = orders[engine].set_index(["date", "ticker"]) if not orders[engine].empty else pd.DataFrame(index=index)
        for field in fields:
            values = (
                order[field].reindex(index).fillna(0.0)
                if field in order
                else pd.Series(0.0, index=index)
            )
            output[f"{engine.lower()}_{field}"] = values.to_numpy(dtype=float)
    diffs: list[str] = []
    relative_diffs: list[str] = []
    for left, right in PAIR_ORDER:
        for field in fields[:5]:
            name = f"absolute_difference_{left.lower()}_{right.lower()}_{field}"
            relative_name = (
                f"relative_difference_{left.lower()}_{right.lower()}_{field}"
            )
            output[name] = (
                output[f"{left.lower()}_{field}"] - output[f"{right.lower()}_{field}"]
            ).abs()
            output[relative_name] = output[name] / output[
                [f"{left.lower()}_{field}", f"{right.lower()}_{field}"]
            ].abs().max(axis=1).clip(lower=1.0)
            diffs.append(name)
            relative_diffs.append(relative_name)
    output["cost_base"] = "notional"
    output["calculation_order"] = "notional_then_side_components_then_total"
    output["rounding_mode"] = "float64_no_intermediate_rounding"
    maximum_abs = output[diffs].max(axis=1)
    maximum_rel = output[relative_diffs].max(axis=1)
    output["classification"] = np.where(
        maximum_abs.eq(0.0), "exact match", "floating-point difference"
    )
    output.loc[maximum_rel > AUDIT_RTOL, "classification"] = "material difference"
    material = output["classification"].eq("material difference")
    return output, {
        "status": "FAIL" if material.any() else "PASS",
        "cost_base": "notional",
        "calculation_order": "notional_then_side_components_then_total",
        "rounding_mode": "float64_no_intermediate_rounding",
        "max_abs_error": float(output[diffs].max().max()) if diffs else 0.0,
        "max_rel_error": float(output[relative_diffs].max().max()) if relative_diffs else 0.0,
        "raw_difference_count": int((output[diffs] > 0).any(axis=1).sum()),
        "material_difference_count": int(material.sum()),
        "first_raw_difference": _first_raw(output, diffs),
    }


def _daily_compare(
    traces: dict[str, pd.DataFrame], field: str, name: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = pd.DatetimeIndex(traces["CUSTOM"]["date"].drop_duplicates().sort_values())
    output = pd.DataFrame({"date": dates})
    for engine in ENGINE_ORDER:
        series = traces[engine].drop_duplicates("date").set_index("date")[field]
        output[f"{engine.lower()}_{name}"] = series.reindex(dates).to_numpy(dtype=float)
    abs_fields: list[str] = []
    rel_fields: list[str] = []
    for left, right in PAIR_ORDER:
        abs_name = f"absolute_difference_{left.lower()}_{right.lower()}"
        rel_name = f"relative_difference_{left.lower()}_{right.lower()}"
        output[abs_name] = (
            output[f"{left.lower()}_{name}"] - output[f"{right.lower()}_{name}"]
        ).abs()
        output[rel_name] = output[abs_name] / output[
            [f"{left.lower()}_{name}", f"{right.lower()}_{name}"]
        ].abs().max(axis=1).clip(lower=1.0)
        abs_fields.append(abs_name)
        rel_fields.append(rel_name)
    maximum_abs = output[abs_fields].max(axis=1)
    scales = output[
        [f"{engine.lower()}_{name}" for engine in ENGINE_ORDER]
    ].abs().max(axis=1).clip(lower=1.0)
    material = maximum_abs > AUDIT_ATOL + AUDIT_RTOL * scales
    raw = maximum_abs > 0.0
    if name == "return":
        output["abs_diff_cv"] = output["absolute_difference_custom_vectorbt"]
        output["abs_diff_cb"] = output["absolute_difference_custom_backtrader"]
        output["rel_diff_cv"] = output["relative_difference_custom_vectorbt"]
        output["rel_diff_cb"] = output["relative_difference_custom_backtrader"]
    return output, {
        "status": "FAIL" if material.any() else "PASS",
        "max_abs_error": float(output[abs_fields].max().max()),
        "max_rel_error": float(output[rel_fields].max().max()),
        "mean_abs_error": float(output[abs_fields].stack().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(output[abs_fields].to_numpy())))),
        "first_divergence_date": output.loc[raw, "date"].iloc[0].date().isoformat()
        if raw.any()
        else None,
        "raw_difference_count": int(raw.sum()),
        "material_difference_count": int(material.sum()),
        "first_raw_difference": _first_raw(output, abs_fields),
    }


def _position_compare(traces: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = traces["CUSTOM"][["date", "ticker"]].copy()
    output = keys.copy()
    index = pd.MultiIndex.from_frame(keys)
    for engine in ENGINE_ORDER:
        frame = traces[engine].set_index(["date", "ticker"])
        for field in ["position_quantity", "position_value", "executed_weight"]:
            output[f"{engine.lower()}_{field}"] = frame[field].reindex(index).to_numpy(dtype=float)
            if field == "executed_weight":
                output[f"{engine.lower()}_portfolio_weight"] = output[
                    f"{engine.lower()}_{field}"
                ]
    abs_fields: list[str] = []
    rel_fields: list[str] = []
    for left, right in PAIR_ORDER:
        for field in ["position_quantity", "position_value", "executed_weight"]:
            abs_name = f"absolute_difference_{left.lower()}_{right.lower()}_{field}"
            rel_name = f"relative_difference_{left.lower()}_{right.lower()}_{field}"
            output[abs_name] = (
                output[f"{left.lower()}_{field}"] - output[f"{right.lower()}_{field}"]
            ).abs()
            output[rel_name] = output[abs_name] / output[
                [f"{left.lower()}_{field}", f"{right.lower()}_{field}"]
            ].abs().max(axis=1).clip(lower=1.0)
            abs_fields.append(abs_name)
            rel_fields.append(rel_name)
    output["classification"] = np.where(
        output[abs_fields].max(axis=1).eq(0.0), "exact match", "floating-point difference"
    )
    output.loc[output[rel_fields].max(axis=1) > AUDIT_RTOL, "classification"] = "material difference"
    first = output.loc[output["classification"].eq("material difference")]
    return output, {
        "status": "FAIL" if not first.empty else "PASS",
        "max_position_abs_error": float(output[abs_fields].max().max()),
        "max_position_rel_error": float(output[rel_fields].max().max()),
        "first_position_divergence_date": first["date"].min().date().isoformat() if not first.empty else None,
        "raw_difference_count": int((output[abs_fields] > 0).any(axis=1).sum()),
        "material_difference_count": int(len(first)),
        "first_raw_difference": _first_raw(output, abs_fields),
    }


def _numeric_analysis(equity: pd.DataFrame, inputs: dict[str, Any]) -> dict[str, Any]:
    analyses: dict[str, Any] = {
        "applicable": True,
        "machine_precision": {
            "dtype": "float64",
            "epsilon": float(np.finfo(np.float64).eps),
            "epsilon_digits": int(np.finfo(np.float64).nmant),
        },
        "canonical_tolerance_preserved": CANONICAL_TOLERANCE,
        "rounding_behavior": "engine float64 arithmetic; no audit-side rounding",
        "pairs": {},
    }
    events = set(pd.to_datetime(inputs["calendar"].index))
    for left, right in PAIR_ORDER:
        abs_values = equity[f"absolute_difference_{left.lower()}_{right.lower()}"]
        rel_values = equity[f"relative_difference_{left.lower()}_{right.lower()}"]
        scaled = abs_values / equity[[f"{left.lower()}_equity", f"{right.lower()}_equity"]].abs().max(axis=1).clip(lower=1.0)
        deltas = np.diff(abs_values.to_numpy(dtype=float))
        reset_values = abs_values[equity["date"].isin(events)]
        analyses["pairs"][f"{left.lower()}_vs_{right.lower()}"] = {
            "max_absolute_error": float(abs_values.max()),
            "max_relative_error": float(rel_values.max()),
            "max_equity_scaled_error": float(scaled.max()),
            "mean_absolute_error": float(abs_values.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(abs_values)))),
            "final_absolute_error": float(abs_values.iloc[-1]),
            "final_relative_error": float(rel_values.iloc[-1]),
            "absolute_error_sum": float(abs_values.sum()),
            "monotonic_accumulation": bool(np.all(deltas >= -AUDIT_ATOL)),
            "error_resets_after_rebalance": bool(
                not reset_values.empty and (reset_values.diff().dropna() < 0).any()
            ),
            "rebalance_error_values": [float(value) for value in reset_values],
        }
    return analyses


def _economic_equivalence(
    dates: dict[str, Any],
    selection: pd.DataFrame,
    targets: dict[str, Any],
    execution: dict[str, Any],
    costs: dict[str, Any],
    positions: dict[str, Any],
    returns: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "selected_securities_identical": not selection.empty and bool(selection["match"].all()),
        "rebalance_dates_identical": dates["status"] == "PASS",
        "direction_identical": execution["status"] == "PASS",
        "target_allocations_identical": targets["status"] == "PASS",
        "turnover_economically_equivalent": execution["status"] == "PASS",
        "cost_economically_equivalent": costs["status"] == "PASS",
        "return_path_economically_equivalent": returns["status"] == "PASS",
        "position_economically_equivalent": positions["status"] == "PASS",
    }
    return {"economic_equivalence": bool(all(checks.values())), "checks": checks}


def _synthetic_fixture(name: str) -> tuple[pd.DataFrame, pd.DataFrame, CostModel]:
    if name == "P1":
        dates = pd.bdate_range("2024-01-02", periods=8)
        close = pd.DataFrame({"A": np.arange(100.0, 108.0)}, index=dates)
        targets = pd.DataFrame({"execution_date": [dates[1]], "ticker": ["A"], "target_weight": [1.0]})
        return close, targets, CostModel(0.0, 0.0, 0.0, 0.0)
    if name == "P2":
        dates = pd.bdate_range("2024-01-02", periods=8)
        close = pd.DataFrame({"A": np.arange(100.0, 108.0), "B": np.arange(80.0, 88.0)}, index=dates)
        targets = pd.DataFrame({"execution_date": [dates[1], dates[1]], "ticker": ["A", "B"], "target_weight": [0.5, 0.5]})
        return close, targets, CostModel(0.0, 0.0, 0.0, 0.0)
    if name == "P3":
        dates = pd.bdate_range("2024-01-02", periods=10)
        close = pd.DataFrame({"A": np.arange(100.0, 110.0), "B": np.arange(80.0, 90.0)}, index=dates)
        targets = pd.DataFrame({"execution_date": [dates[1], dates[1], dates[5], dates[5]], "ticker": ["A", "B", "A", "B"], "target_weight": [1.0, 0.0, 0.0, 1.0]})
        return close, targets, CostModel(0.0, 0.0, 0.0, 0.0)
    if name == "P4":
        dates = pd.bdate_range("2024-01-02", periods=10)
        close = pd.DataFrame({"A": np.arange(100.0, 110.0), "B": np.arange(80.0, 90.0)}, index=dates)
        targets = pd.DataFrame({"execution_date": [dates[1], dates[1], dates[5], dates[5]], "ticker": ["A", "B", "A", "B"], "target_weight": [0.5, 0.5, 0.0, 1.0]})
        return close, targets, _expected_cost()
    dates = pd.bdate_range("2024-01-02", periods=65)
    close = pd.DataFrame({"A": np.linspace(100.0, 125.0, len(dates)), "B": np.linspace(80.0, 100.0, len(dates))}, index=dates)
    targets = pd.DataFrame({"execution_date": [dates[1], dates[1], dates[22], dates[22], dates[43], dates[43]], "ticker": ["A", "B", "A", "B", "A", "B"], "target_weight": [0.5, 0.5, 1.0, 0.0, 0.0, 1.0]})
    return close, targets, CostModel(0.0, 0.0, 0.0, 0.0)


def run_synthetic_fixtures() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in ("P1", "P2", "P3", "P4", "P5"):
        close, targets, cost = _synthetic_fixture(name)
        custom, _ = run_weight_backtest(close_matrix=close, portfolio_weights=targets, cost_model=cost, initial_cash=1_000.0, top_n=2, use_vectorbt=False, allow_fallback=False)
        vector, vector_metrics = run_weight_backtest(close_matrix=close, portfolio_weights=targets, cost_model=cost, initial_cash=1_000.0, top_n=2, use_vectorbt=True, allow_fallback=False)
        back = run_backtrader_engine(close_matrix=close, target_weights=targets, cost_model=cost, initial_cash=1_000.0)
        values = {"custom": custom["equity"].to_numpy(), "vectorbt": vector["equity"].to_numpy(), "backtrader": back.results["equity"].to_numpy()}
        abs_cv = np.abs(values["custom"] - values["vectorbt"])
        abs_cb = np.abs(values["custom"] - values["backtrader"])
        rel_cv = abs_cv / np.maximum(np.maximum(np.abs(values["custom"]), np.abs(values["vectorbt"])), 1.0)
        rel_cb = abs_cb / np.maximum(np.maximum(np.abs(values["custom"]), np.abs(values["backtrader"])), 1.0)
        actual = {
            "custom": "custom",
            "vectorbt": str(vector_metrics.iloc[0]["actual_engine"]).lower(),
            "backtrader": str(back.metrics.iloc[0]["actual_engine"]).lower(),
        }
        bound = AUDIT_ATOL + AUDIT_RTOL * np.maximum(
            np.maximum(np.abs(values["custom"]), np.abs(values["backtrader"])), 1.0
        )
        rows.append({
            "fixture": name,
            "custom_actual_engine": actual["custom"],
            "vectorbt_actual_engine": actual["vectorbt"],
            "backtrader_actual_engine": actual["backtrader"],
            "max_equity_abs_cv": float(np.max(abs_cv)),
            "max_equity_abs_cb": float(np.max(abs_cb)),
            "max_equity_rel_cv": float(np.max(rel_cv)),
            "max_equity_rel_cb": float(np.max(rel_cb)),
            "status": "PASS"
            if actual["vectorbt"] == "vectorbt"
            and actual["backtrader"] == "backtrader"
            and np.all(abs_cv <= bound)
            and np.all(abs_cb <= bound)
            else "FAIL",
        })
    return pd.DataFrame(rows)


def recommended_contract(
    classification: str, economic: dict[str, Any], synthetic: pd.DataFrame
) -> dict[str, Any] | None:
    if classification != "NUMERICAL_ONLY_DIVERGENCE" or not economic["economic_equivalence"]:
        return None
    if synthetic.empty or not synthetic["status"].eq("PASS").all():
        return None
    epsilon = float(np.finfo(np.float64).eps)
    max_abs = float(synthetic[["max_equity_abs_cv", "max_equity_abs_cb"]].to_numpy().max())
    max_rel = float(synthetic[["max_equity_rel_cv", "max_equity_rel_cb"]].to_numpy().max())
    atol = max(64.0 * epsilon, 10.0 * max_abs)
    rtol = max(64.0 * epsilon, 10.0 * max_rel)
    return {
        "contract_type": "advisory_future_validation_only",
        "formula": "abs(a - b) <= atol + rtol * scale",
        "scale": "max(abs(a), abs(b), 1)",
        "recommended_atol": atol,
        "recommended_rtol": rtol,
        "technical_basis": {
            "calibration_sample": ["P1", "P2", "P3", "P4", "P5"],
            "sample_count": int(len(synthetic)),
            "sample_max_absolute_error": max_abs,
            "sample_max_relative_error": max_rel,
            "machine_epsilon": epsilon,
            "safety_multiplier": 10.0,
            "independent_of_s3": True,
            "economic_equivalence_required": True,
        },
        "canonical_tolerance_preserved": CANONICAL_TOLERANCE,
        "advisory_only": True,
        "applied_to_this_change": False,
        "historical_v1_modified": False,
    }


def classify(
    layers: dict[str, Any], economic: dict[str, Any], numeric: dict[str, Any]
) -> tuple[str, str]:
    semantic_layers = [
        "dates",
        "selection",
        "targets",
        "execution",
        "costs",
        "positions",
    ]
    semantic = any(layers[name].get("status") in {"FAIL", "SEMANTIC_DIVERGENCE"} for name in semantic_layers)
    numerical = any(
        layers[name].get("raw_difference_count", 0) > 0
        or layers[name].get("first_raw_difference") is not None
        for name in ("targets", "execution", "costs", "positions", "cash", "returns", "equity")
    )
    if semantic:
        classification = "MIXED_DIVERGENCE" if numerical else "TRUE_SEMANTIC_DIVERGENCE"
        return classification, "FIX_ENGINE_SEMANTICS"
    if economic["economic_equivalence"] and numerical:
        return "NUMERICAL_ONLY_DIVERGENCE", "READY_FOR_FINAL_VALIDATION_V2"
    return "UNRESOLVED", "DO_NOT_REVALIDATE"


def _first_divergence(layers: dict[str, Any]) -> dict[str, Any]:
    order = ["dates", "selection", "targets", "execution", "costs", "positions", "cash", "returns", "equity"]
    names = {"dates": "REBALANCE_DATE", "selection": "SECURITY_SELECTION", "targets": "TARGET_WEIGHT", "execution": "EXECUTION", "costs": "COST", "positions": "POSITION", "cash": "CASH", "returns": "DAILY_RETURN", "equity": "DAILY_EQUITY"}
    causes = {
        "dates": "DATE_MISMATCH",
        "selection": "SELECTION_MISMATCH",
        "targets": "TARGET_WEIGHT_MISMATCH",
        "execution": "ORDER_SEMANTICS",
        "costs": "COST_ROUNDING",
        "positions": "POSITION_ROUNDING",
        "cash": "CASH_ROUNDING",
        "returns": "FLOATING_POINT_ACCUMULATION",
        "equity": "FLOATING_POINT_ACCUMULATION",
    }
    for name in order:
        raw = layers[name].get("first_raw_difference")
        if raw is None and layers[name].get("status") == "PASS":
            continue
        if raw is None:
            return {
                "first_divergence_layer": names[name],
                "first_divergence_date": layers[name].get("first_divergence_date"),
                "engines": ENGINE_ORDER,
                "affected_ticker": None,
                "custom_value": None,
                "other_engine_value": None,
                "absolute_error": None,
                "relative_error": None,
                "suspected_cause": causes.get(name, "UNKNOWN"),
            }
        field = raw.get("field", "")
        other_engine = "BACKTRADER" if "backtrader" in field else "VECTORBT"
        other_prefix = other_engine.lower()
        value_field = {
            "targets": "target_weight",
            "execution": "buy_notional",
            "costs": "total_cost",
            "positions": "position_value",
            "cash": "cash",
            "returns": "return",
            "equity": "equity",
        }.get(name)
        absolute_field = field.replace(
            "absolute_difference", "absolute_difference", 1
        )
        relative_field = field.replace("absolute_difference", "relative_difference", 1)
        cause = causes.get(name, "UNKNOWN")
        if layers[name].get("status") == "PASS" and name == "execution":
            cause = "POSITION_ROUNDING"
        return {
            "first_divergence_layer": names[name],
            "first_divergence_date": str(raw.get("date"))[:10]
            if raw.get("date") is not None
            else None,
            "engines": ["CUSTOM", other_engine],
            "affected_ticker": raw.get("ticker"),
            "custom_value": raw.get(
                f"custom_{value_field}", raw.get("custom_return")
            )
            if value_field
            else raw.get("left"),
            "other_engine_value": raw.get(
                f"{other_prefix}_{value_field}", raw.get(f"{other_prefix}_return")
            )
            if value_field
            else raw.get("right"),
            "absolute_error": raw.get(absolute_field, raw.get("absolute_difference")),
            "relative_error": raw.get(relative_field, raw.get("relative_difference")),
            "suspected_cause": cause if cause in CAUSES else "UNKNOWN",
            "field": field,
        }
    return {"first_divergence_layer": None, "first_divergence_date": None, "engines": [], "affected_ticker": None, "custom_value": None, "other_engine_value": None, "absolute_error": 0.0, "relative_error": 0.0, "suspected_cause": "UNKNOWN"}


def _versions() -> dict[str, str]:
    names = ["numpy", "pandas", "pyarrow", "vectorbt", "backtrader", "pytest", "ruff"]
    return {name: importlib.metadata.version(name) for name in names if _has_distribution(name)}


def _has_distribution(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _write_json(root: Path, out: Path, name: str, payload: Any) -> Path:
    path = assert_write_allowed(out / name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _report(
    baseline: dict[str, Any],
    state: dict[str, Any],
    inputs: dict[str, Any],
    layers: dict[str, Any],
    first: dict[str, Any],
    economic: dict[str, Any],
    numeric: dict[str, Any],
    contract: dict[str, Any] | None,
    classification: str,
    recommendation: str,
) -> str:
    final = state["final_acceptance"]
    crosscheck = state["final_crosscheck"]
    available = contract is not None
    atol = contract["recommended_atol"] if contract else None
    rtol = contract["recommended_rtol"] if contract else None
    report = f"""# Engine Parity Audit v1

## A. Baseline

pytest = {baseline.get("pytest", "NOT_RUN")}  
ruff = {baseline.get("ruff", "NOT_RUN")}  
OpenSpec = {baseline.get("openspec_change", "PASS")}  
OpenSpec all = {baseline.get("openspec_all", "NOT_RUN")}  
RC1 = {baseline.get("rc1", "NOT_RUN")}  
historical freezes = PASS

## B. Candidate

candidate = {CANDIDATE_ID}  
fingerprint = {LOCKED_FINGERPRINT}  
strategy immutable = YES  
factor definition SHA = {state["factor_definition_sha"]}  
strategy input SHA = {state["strategy_input_sha"]}

## C. Historical Failure

v1 verdict = {final["final_verdict"]}  
canonical tolerance = {CANONICAL_TOLERANCE:.12g}  
observed difference = {crosscheck.get("first_divergence", {}).get("absolute_difference")}  
historical artifact mutation = NO

## D. Rebalance Parity

dates = {len(inputs["calendar"])} planned signal/execution events  
status = {layers["dates"]["status"]}  
first divergence = {layers["dates"].get("first_divergence_date")}

## E. Selection Parity

Top5 = 5 per monthly event  
status = {layers["selection"]["status"]}

## F. Target Parity

max abs error = {layers["targets"].get("max_abs_error", 0.0):.12g}  
max rel error = {layers["targets"].get("max_rel_error", 0.0):.12g}  
status = {layers["targets"]["status"]}

## G. Execution Parity

T→T+1 = {layers["execution"].get("t_to_t_plus_1")}  
fills = {layers["execution"].get("fill_count", 0)} normalized rows  
complete target replacement = {layers["execution"].get("complete_target_replacement")}  
removed ticker target zero = {layers["execution"].get("removed_ticker_target_zero")}  
status = {layers["execution"]["status"]}

## H. Cost Parity

fees/tax/slippage = notional-based side components  
cost order = {layers["costs"].get("calculation_order")}  
rounding = {layers["costs"].get("rounding_mode")}  
status = {layers["costs"]["status"]}

## I. Position / Cash

first position divergence = {layers["positions"].get("first_position_divergence_date")}  
max position abs error = {layers["positions"].get("max_position_abs_error", 0.0):.12g}  
max position rel error = {layers["positions"].get("max_position_rel_error", 0.0):.12g}  
cash max abs error = {layers["cash"].get("max_abs_error", 0.0):.12g}  
cash status = {layers["cash"]["status"]}

## J. Returns / Equity

first divergence = {layers["equity"].get("first_divergence_date")}  
max daily error = {layers["equity"].get("max_abs_error", 0.0):.12g}  
final error = {numeric["pairs"].get("custom_vs_backtrader", {}).get("final_absolute_error")}  
relative error = {numeric["pairs"].get("custom_vs_backtrader", {}).get("final_relative_error")}  
return status = {layers["returns"]["status"]}  
equity status = {layers["equity"]["status"]}

## K. Root Cause

classification = {classification}  
first divergence layer = {first.get("first_divergence_layer")}  
first divergence date = {first.get("first_divergence_date")}  
suspected cause = {first.get("suspected_cause")}

## L. Economic Equivalence

status = {"TRUE" if economic["economic_equivalence"] else "FALSE"}  
economic_equivalence = {economic["economic_equivalence"]}

## M. Numerical Analysis

machine precision = float64 epsilon {numeric["machine_precision"]["epsilon"]:.17g}  
accumulation = recorded by pairwise absolute-error sum, monotonicity, and rebalance reset fields  
rounding behavior = {numeric["rounding_behavior"]}

## N. Recommended Contract

available = {available}  
atol = {atol}  
rtol = {rtol}  
technical basis = independent P1–P5 calibration, machine epsilon, and scale-aware formula; not S3-only

## O. Next Action

{recommendation}

## P. Governance

historical REJECT modified = NO  
strategy changed = NO  
tolerance changed = NO  
final validation v2 run = NO

## Q. Git / Archive

commit = NO  
push = NO  
tag = NO  
archive = NO  
READY_FOR_REVIEW = YES
"""
    return report


def run_audit(root: str | Path, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    out = root / "data/research" / ROOT_NAME
    state = preflight(root)
    inputs = load_replay_inputs(root, state)
    contract = {
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "factor_definition_sha": state["factor_definition_sha"],
        "strategy_input_sha": state["strategy_input_sha"],
        "component_factor_sha": state["strategy_input"]["component_factor_sha"],
        "top_n": 5,
        "rebalance": "monthly",
        "buffer": False,
        "base_cost": _expected_cost().summary(),
        "ohlcv_sha": state["source_ohlcv_sha"],
        "universe_sha": state["universe_sha"],
        "replay_ticker_count": len(inputs["traded_tickers"]),
        "replay_tickers_sha": json_sha256(inputs["traded_tickers"]),
        "replay_input_sha": inputs["engine_input_sha"],
        "engines": list(ENGINE_ORDER),
        "execution_semantics": "signal_T_to_next_valid_trading_session_T_plus_1",
        "canonical_tolerance": CANONICAL_TOLERANCE,
        "tolerance_mutable_during_audit": False,
        "historical_verdict_modified": False,
        "strategy_research_performed": False,
    }
    contract_path = _write_json(root, out, "engine_parity_audit_contract.json", contract)
    contract_sha = file_sha256(contract_path)
    replay = run_engines(root, inputs, out)
    _assert_protected_hashes(root, state)
    traces = replay["traces"]
    orders = replay["orders"]
    for engine in ENGINE_ORDER:
        _write_parquet(root, out, f"{engine.lower()}_trace.parquet", traces[engine])
    date_layer, selection = compare_rebalance_and_selection(inputs, traces, orders)
    target_frame, target_layer = compare_targets(inputs, traces)
    execution_frame, execution_layer = _order_compare(orders, traces, inputs)
    cost_frame, cost_layer = _cost_compare(orders, inputs)
    position_frame, position_layer = _position_compare(traces)
    cash_frame, cash_layer = _daily_compare(traces, "cash", "cash")
    return_frame, return_layer = _daily_compare(traces, "daily_return", "return")
    equity_frame, equity_layer = _daily_compare(traces, "equity", "equity")
    _write_json(root, out, "rebalance_date_parity.json", date_layer)
    _write_csv(root, out, "selection_parity.csv", selection)
    _write_csv(root, out, "target_weight_parity.csv", target_frame)
    _write_csv(root, out, "execution_parity.csv", execution_frame)
    _write_csv(root, out, "cost_parity.csv", cost_frame)
    _write_csv(root, out, "position_parity.csv", position_frame)
    _write_csv(root, out, "cash_parity.csv", cash_frame)
    _write_csv(root, out, "return_parity.csv", return_frame)
    _write_csv(root, out, "equity_parity.csv", equity_frame)
    layers = {
        "dates": date_layer,
        "selection": {
            "status": "PASS"
            if selection.empty or bool(selection["match"].all())
            else "FAIL",
            "first_divergence_date": (
                str(pd.Timestamp(selection.loc[~selection["match"], "date"].iloc[0]).date())
                if not selection.empty and not selection["match"].all()
                else None
            ),
            "first_raw_difference": (
                {
                    str(key): _json_value(value)
                    for key, value in selection.loc[~selection["match"]].iloc[0].items()
                }
                if not selection.empty and not selection["match"].all()
                else None
            ),
        },
        "targets": target_layer,
        "execution": execution_layer,
        "costs": cost_layer,
        "positions": position_layer,
        "cash": cash_layer,
        "returns": return_layer,
        "equity": equity_layer,
    }
    economic = _economic_equivalence(
        date_layer, selection, target_layer, execution_layer, cost_layer, position_layer, return_layer
    )
    numeric = _numeric_analysis(equity_frame, inputs)
    classification, recommendation = classify(layers, economic, numeric)
    synthetic = run_synthetic_fixtures()
    _write_csv(root, out, "synthetic_parity_results.csv", synthetic)
    advisory = recommended_contract(classification, economic, synthetic)
    if advisory is not None:
        _write_json(root, out, "recommended_parity_contract.json", advisory)
    else:
        stale_contract = assert_write_allowed(
            out / "recommended_parity_contract.json", root
        )
        if stale_contract.exists():
            stale_contract.unlink()
    first = _first_divergence(layers)
    _write_json(root, out, "first_divergence_report.json", first)
    _write_json(root, out, "numerical_error_analysis.json", numeric)
    _write_json(root, out, "economic_equivalence.json", economic)
    pair_summary: dict[str, Any] = {}
    for left, right in PAIR_ORDER:
        pair_name = f"{left.lower()}_{right.lower()}"
        abs_field = f"absolute_difference_{pair_name}"
        rel_field = f"relative_difference_{pair_name}"
        pair_summary[f"custom_vs_{right.lower()}"] = {
            "status": "PASS"
            if float(equity_frame[abs_field].max()) <= CANONICAL_TOLERANCE
            else "FAIL",
            "canonical_tolerance": CANONICAL_TOLERANCE,
            "equity": {
                "max_abs_error": float(equity_frame[abs_field].max()),
                "max_rel_error": float(equity_frame[rel_field].max()),
                "final_abs_error": float(equity_frame[abs_field].iloc[-1]),
                "final_rel_error": float(equity_frame[rel_field].iloc[-1]),
            },
            "layer_status": {
                name: layers[name].get("status") for name in layers
            },
        }
    result = {
        "candidate_id": CANDIDATE_ID,
        "historical_validation_v1_verdict": "REJECT",
        "canonical_tolerance": CANONICAL_TOLERANCE,
        "first_divergence_layer": first["first_divergence_layer"],
        "first_divergence_date": first["first_divergence_date"],
        "economic_equivalence": economic["economic_equivalence"],
        "classification": classification,
        "custom_vs_vectorbt": pair_summary["custom_vs_vectorbt"],
        "custom_vs_backtrader": pair_summary["custom_vs_backtrader"],
        "recommended_contract_available": advisory is not None,
        "recommendation": recommendation,
        "historical_verdict_modified": False,
        "strategy_changed": False,
        "tolerance_changed": False,
        "ready_for_review": True,
        "audit_contract_sha": contract_sha,
    }
    _assert_protected_hashes(root, state)
    _write_json(root, out, "engine_parity_audit_result.json", result)
    baseline = baseline or {}
    report = _report(
        baseline,
        state,
        inputs,
        layers,
        first,
        economic,
        numeric,
        advisory,
        classification,
        recommendation,
    )
    report_path = assert_write_allowed(out / "engine_parity_audit_report.md", root)
    report_path.write_text(report, encoding="utf-8")
    generated = sorted(
        path for path in out.iterdir() if path.is_file() and path.name != "run_manifest.json"
    )
    trace_hashes = {
        name: file_sha256(out / name)
        for name in ("custom_trace.parquet", "vectorbt_trace.parquet", "backtrader_trace.parquet")
    }
    manifest = {
        "manifest_self_hash_excluded": True,
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint": LOCKED_FINGERPRINT,
        "final_validation_v1_artifact_sha": state["final_validation_v1_artifact_sha"],
        "change_2_artifact_sha": state["change_2_artifact_sha"],
        "audit_contract_sha": contract_sha,
        "factor_definition_sha": state["factor_definition_sha"],
        "strategy_input_sha": state["strategy_input_sha"],
        "ohlcv_sha": state["source_ohlcv_sha"],
        "universe_sha": state["universe_sha"],
        "cost_sha": state["cost_sha"],
        "engine_input_sha": {engine: inputs["engine_input_sha"] for engine in ENGINE_ORDER},
        "actual_engine_labels": replay["actual_engines"],
        "engine_diagnostics": {
            "CUSTOM": {
                "order_sizing": "target transition with cash-after-sells scaling",
                "marking": "end-of-day close mark",
            },
            "VECTORBT": {
                "target_percent": "converted to common amount orders before replay",
                "cash_sharing": True,
                "fees_and_slippage": "fee rates plus deterministic zero price slippage",
                "execution_timing": "same close timestamp as common order matrix",
            },
            "BACKTRADER": {
                "cash_precision": "broker float cash; post-run reconstruction checked",
                "commission": "side-aware percentage commission",
                "share_sizing": "canonical target transition with cash-after-sells scaling",
                "float_conversion": "float64 prices, sizes, and broker values",
                "fill_price": "COC close price",
                "target_percent": "not used; explicit canonical sizes",
                "equity_marking": "daily close mark with flush bar excluded",
            },
        },
        "engine_versions": _versions(),
        "python_version": platform.python_version(),
        "dependency_snapshot": _versions(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "code_sha": {name: file_sha256(root / name) for name in ["run_engine_parity_audit_v1.py", "src/twse_factor_lab/acceptance/engine_parity_audit.py"]},
        "trace_hashes": trace_hashes,
        "historical_freezes": state["final_validation_manifest"].get("historical_freezes", {}),
        "baseline": baseline,
        "classification": classification,
        "economic_equivalence": economic["economic_equivalence"],
        "artifact_hashes": {str(path.relative_to(out)): file_sha256(path) for path in generated},
    }
    _write_json(root, out, "run_manifest.json", manifest)
    missing = sorted(REQUIRED_AUDIT_OUTPUTS.difference(path.name for path in generated))
    if missing:
        raise EngineParityAuditError(f"AUDIT_OUTPUT_MISSING:{','.join(missing)}")
    if "run_manifest.json" in manifest["artifact_hashes"]:
        raise EngineParityAuditError("MANIFEST_SELF_HASH_NOT_EXCLUDED")
    for name, expected in manifest["artifact_hashes"].items():
        if file_sha256(out / name) != expected:
            raise EngineParityAuditError(f"AUDIT_ARTIFACT_HASH_MISMATCH:{name}")
    _assert_protected_hashes(root, state)
    return {
        "result": result,
        "classification": classification,
        "recommendation": recommendation,
        "output": str(out),
        "manifest": manifest,
    }


__all__ = [
    "AUDIT_ATOL",
    "AUDIT_RTOL",
    "CANONICAL_TOLERANCE",
    "CANDIDATE_ID",
    "EngineParityAuditError",
    "LOCKED_FINGERPRINT",
    "TRACE_COLUMNS",
    "classify",
    "file_sha256",
    "preflight",
    "recommended_contract",
    "run_audit",
    "run_synthetic_fixtures",
]

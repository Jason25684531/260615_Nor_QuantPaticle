"""Validation and loading for the Day 4 research-only handoff."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.governance.isolation import RESEARCH_NAMESPACE


class HandoffValidationError(ValueError):
    """The Day 4 handoff is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class StrategyHandoff:
    """Validated Day 4 artifacts consumed by Day 5 diagnostics."""

    path: Path
    metadata: dict[str, Any]
    returns: pd.Series
    nav: pd.Series
    positions: pd.DataFrame
    target_weights: pd.DataFrame

    @property
    def research_id(self) -> str:
        return str(self.metadata["research_id"])

    @property
    def strategy_id(self) -> str:
        return str(self.metadata["strategy_id"])

    @property
    def experiment_id(self) -> str:
        return str(self.metadata["experiment_id"])

    @property
    def dataset_version(self) -> str:
        return str(self.metadata["dataset_version"])

    @property
    def transactions_available(self) -> bool:
        return bool(self.metadata.get("transactions", {}).get("available", False))


def _path_under_research(root: Path, handoff_dir: str | Path) -> tuple[Path, str]:
    root = root.resolve()
    path = Path(handoff_dir)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise HandoffValidationError(
            "handoff must be under the repository root"
        ) from exc
    namespace = RESEARCH_NAMESPACE.parts
    if relative.parts[: len(namespace)] != namespace:
        raise HandoffValidationError(
            "Day 4 handoff must be under data/research/<research_id>/"
        )
    if len(relative.parts) < 4 or relative.parts[3] != "strategy_handoffs":
        raise HandoffValidationError("path is not a Day 4 strategy handoff")
    return path, relative.as_posix()


def _read_frame(path: Path, required_columns: set[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise HandoffValidationError(f"missing {name}: {path.name}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise HandoffValidationError(f"cannot read {name}: {path.name}") from exc
    missing = required_columns.difference(frame.columns)
    if missing:
        raise HandoffValidationError(
            f"{name} missing columns: {', '.join(sorted(missing))}"
        )
    if "date" not in frame:
        raise HandoffValidationError(f"{name} must contain a date column")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
    ):
        raise HandoffValidationError(f"{name} dates must be unique and monotonic")
    frame = frame.copy()
    frame["date"] = dates
    return frame


def _finite_column(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise HandoffValidationError(f"{name}.{column} contains non-finite values")
    return values.astype(float)


def _target_records(metadata: dict[str, Any], dates: pd.DatetimeIndex) -> pd.DataFrame:
    records = metadata.get("target_weights", [])
    if records is None:
        records = []
    if not isinstance(records, list):
        raise HandoffValidationError("metadata.target_weights must be a list")
    columns = [
        "signal_date",
        "execution_date",
        "ticker",
        "target_weight",
        "execution_lag_days",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not {
            "signal_date",
            "execution_date",
            "ticker",
            "target_weight",
        }.issubset(record):
            raise HandoffValidationError("invalid target-weight record")
        signal = pd.to_datetime(record["signal_date"], errors="coerce")
        execution = pd.to_datetime(record["execution_date"], errors="coerce")
        weight = pd.to_numeric(record["target_weight"], errors="coerce")
        if pd.isna(signal) or pd.isna(execution) or pd.isna(weight):
            raise HandoffValidationError("target-weight dates and values are required")
        if signal >= execution or execution not in dates:
            raise HandoffValidationError("target event must execute after signal date")
        execution_position = dates.searchsorted(execution)
        if execution_position == 0 or dates[execution_position - 1] != signal:
            raise HandoffValidationError(
                "target event must use the next valid trading session"
            )
        if not math.isfinite(float(weight)) or float(weight) < 0:
            raise HandoffValidationError(
                "target_weight must be finite and non-negative"
            )
        ticker = record["ticker"]
        if not isinstance(ticker, str) or not ticker:
            raise HandoffValidationError("target ticker must be non-empty")
        lag = record.get("execution_lag_days", 1)
        if (
            isinstance(lag, bool)
            or not isinstance(lag, (int, float))
            or not float(lag).is_integer()
            or int(lag) < 1
        ):
            raise HandoffValidationError("execution_lag_days must be positive")
        rows.append(
            {
                "signal_date": signal,
                "execution_date": execution,
                "ticker": ticker,
                "target_weight": float(weight),
                "execution_lag_days": int(lag),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    if result.duplicated(["execution_date", "ticker"]).any():
        raise HandoffValidationError("duplicate target event for ticker/date")
    return result.sort_values(["execution_date", "ticker"]).reset_index(drop=True)


def load_strategy_handoff(
    root: str | Path,
    handoff_dir: str | Path,
    *,
    tolerance: float = 1e-8,
    expected_research_id: str | None = None,
    expected_dataset_version: str | None = None,
) -> StrategyHandoff:
    """Load and validate a Day 4 handoff before any Day 5 calculation."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise HandoffValidationError("tolerance must be finite and non-negative")
    path, relative = _path_under_research(Path(root), handoff_dir)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise HandoffValidationError("missing metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffValidationError("invalid metadata.json") from exc
    if not isinstance(metadata, dict):
        raise HandoffValidationError("metadata.json must contain an object")

    required_metadata = {
        "research_id",
        "strategy_id",
        "experiment_id",
        "dataset_version",
        "requested_engine",
        "actual_engine",
        "cost_model",
        "execution_semantics_version",
        "period",
        "transactions",
    }
    if not required_metadata.issubset(metadata):
        missing = required_metadata.difference(metadata)
        raise HandoffValidationError(
            f"metadata missing fields: {', '.join(sorted(missing))}"
        )
    relative_parts = Path(relative).parts
    if metadata["research_id"] != relative_parts[2]:
        raise HandoffValidationError("metadata research_id does not match handoff path")
    directory_identity = relative_parts[4]
    path_strategy, separator, path_experiment = directory_identity.rpartition("__")
    if (
        not separator
        or metadata["strategy_id"] != path_strategy
        or metadata["experiment_id"] != path_experiment
    ):
        raise HandoffValidationError("metadata strategy/experiment identity mismatch")
    if expected_research_id and metadata["research_id"] != expected_research_id:
        raise HandoffValidationError("metadata research_id mismatch")
    if (
        expected_dataset_version
        and metadata["dataset_version"] != expected_dataset_version
    ):
        raise HandoffValidationError("metadata dataset_version mismatch")
    if not isinstance(metadata["cost_model"], dict):
        raise HandoffValidationError("metadata cost_model must be an object")
    if not isinstance(metadata["period"], dict):
        raise HandoffValidationError("metadata period must be an object")
    metadata_handoff = metadata.get("handoff_dir")
    if metadata_handoff is not None:
        metadata_handoff = str(metadata_handoff).replace("\\", "/")
    if metadata_handoff not in {None, relative}:
        raise HandoffValidationError("metadata handoff_dir mismatch")
    transactions = metadata["transactions"]
    if not isinstance(transactions, dict) or not isinstance(
        transactions.get("available"), bool
    ):
        raise HandoffValidationError("metadata transactions availability is invalid")
    if not transactions["available"] and (path / "transactions.csv").exists():
        raise HandoffValidationError("unavailable transactions must not be fabricated")

    returns_frame = _read_frame(path / "returns.csv", {"returns"}, "returns")
    nav_frame = _read_frame(path / "nav.csv", {"nav"}, "NAV")
    positions = _read_frame(path / "positions.csv", set(), "positions")
    returns_values = _finite_column(returns_frame, "returns", "returns")
    nav_values = _finite_column(nav_frame, "nav", "NAV")
    if (nav_values <= 0).any():
        raise HandoffValidationError("NAV must remain positive")
    return_dates = pd.DatetimeIndex(returns_frame["date"])
    nav_dates = pd.DatetimeIndex(nav_frame["date"])
    position_dates = pd.DatetimeIndex(positions["date"])
    if not return_dates.equals(nav_dates) or not return_dates.equals(position_dates):
        raise HandoffValidationError("returns, NAV, and positions dates must align")
    position_columns = [column for column in positions.columns if column != "date"]
    if not position_columns:
        raise HandoffValidationError("positions must contain at least one value column")
    if "cash" not in position_columns:
        raise HandoffValidationError("positions must contain the cash column")
    for column in position_columns:
        _finite_column(positions, column, "positions")
    if len(nav_values) > 1:
        implied = nav_values.iloc[1:].to_numpy() / nav_values.iloc[:-1].to_numpy() - 1.0
        actual = returns_values.iloc[1:].to_numpy()
        if not np.allclose(implied, actual, rtol=0.0, atol=tolerance):
            raise HandoffValidationError("NAV and returns are internally inconsistent")
    period = metadata["period"]
    expected_period = {
        "start_date": return_dates.min().date().isoformat(),
        "end_date": return_dates.max().date().isoformat(),
    }
    for key, value in expected_period.items():
        if period.get(key) not in {None, value}:
            raise HandoffValidationError(f"metadata period {key} does not match files")
    target_weights = _target_records(metadata, return_dates)
    return StrategyHandoff(
        path=path,
        metadata=metadata,
        returns=pd.Series(
            returns_values.to_numpy(), index=return_dates, name="returns"
        ),
        nav=pd.Series(nav_values.to_numpy(), index=return_dates, name="nav"),
        positions=positions.set_index("date"),
        target_weights=target_weights,
    )

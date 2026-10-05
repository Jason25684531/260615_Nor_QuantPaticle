"""Validated, identity-bound inputs for comprehensive performance reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.production.fundamental import FundamentalStrategySpec

PERIODS = ("BACKTEST", "HISTORICAL_OOS", "FRESH_OOS")
TIMEZONE_POLICY = "trading-date labels normalized to UTC midnight; no return fill"
POSITION_COLUMNS = (
    "date",
    "ticker",
    "market_value",
    "cash",
    "portfolio_value",
    "weight",
)
TRANSACTION_COLUMNS = (
    "date",
    "ticker",
    "amount",
    "price",
    "value",
    "commission",
    "tax",
    "slippage",
    "total_cost",
)


class PerformanceDataError(ValueError):
    """Raised when a claimed canonical performance source is invalid."""


class FrozenStrategyChanged(PerformanceDataError):
    """Raised when formal report identity no longer matches the frozen contract."""


@dataclass(frozen=True)
class FrozenIdentity:
    strategy_id: str
    strategy_fingerprint: str
    config: dict[str, Any]
    digest: str


@dataclass
class PerformanceData:
    """Validated inputs for exactly one evidence period."""

    period: str
    status: str
    reason: str | None = None
    returns: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    positions: pd.DataFrame | None = None
    transactions: pd.DataFrame | None = None
    benchmark: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    timezone_policy: str = TIMEZONE_POLICY

    @property
    def available(self) -> bool:
        return self.status == "AVAILABLE" and self.returns is not None

    @property
    def transaction_status(self) -> dict[str, Any]:
        if self.transactions is None or self.transactions.empty:
            return {
                "status": "UNAVAILABLE",
                "reason": self.source.get(
                    "transactions_reason", "NO_TRUE_TRANSACTION_STREAM"
                ),
            }
        return {"status": "AVAILABLE", "reason": None}

    def to_pyfolio_inputs(
        self,
    ) -> tuple[pd.Series, pd.DataFrame | None, pd.DataFrame | None]:
        """Return Pyfolio-compatible views without recomputing strategy data."""

        if not self.available or self.returns is None:
            raise PerformanceDataError(f"{self.period} returns are unavailable")
        positions = None
        if self.positions is not None:
            positions = self.positions.pivot(
                index="date", columns="ticker", values="market_value"
            ).fillna(0.0)
            positions["cash"] = self.positions.groupby("date")["cash"].first()
            positions = positions.reindex(self.returns.index)
            positions.columns.name = None
        transactions = None
        if self.transactions is not None:
            transactions = self.transactions.rename(columns={"ticker": "symbol"})[
                ["date", "symbol", "amount", "price"]
            ].set_index("date")
        return self.returns, positions, transactions

    @classmethod
    def unavailable(cls, period: str, reason: str) -> PerformanceData:
        return cls(_period(period), "INSUFFICIENT_DATA", reason=reason)

    @classmethod
    def from_frames(
        cls,
        period: str,
        returns: pd.Series | pd.DataFrame,
        *,
        benchmark_returns: pd.Series | pd.DataFrame | None = None,
        positions: pd.DataFrame | None = None,
        transactions: pd.DataFrame | None = None,
        benchmark_id: str | None = None,
        benchmark_source: str | None = None,
        source: dict[str, Any] | None = None,
    ) -> PerformanceData:
        name = _period(period)
        strategy = _return_series(returns, "returns")
        benchmark_series = (
            _return_series(benchmark_returns, "benchmark_returns")
            if benchmark_returns is not None
            else None
        )
        position_frame = _positions(positions)
        transaction_frame = _transactions(transactions)
        return_dates = set(strategy.index)
        if (
            position_frame is not None
            and set(position_frame["date"]) != return_dates
        ):
            raise PerformanceDataError(
                "positions dates must align exactly with returns"
            )
        if transaction_frame is not None and not set(
            transaction_frame["date"]
        ).issubset(return_dates):
            raise PerformanceDataError("transaction dates must align with returns")
        benchmark = {
            "benchmark_id": benchmark_id,
            "benchmark_source": benchmark_source,
            "benchmark_start": (
                benchmark_series.index.min().date().isoformat()
                if benchmark_series is not None and not benchmark_series.empty
                else None
            ),
            "benchmark_end": (
                benchmark_series.index.max().date().isoformat()
                if benchmark_series is not None and not benchmark_series.empty
                else None
            ),
        }
        return cls(
            period=name,
            status="AVAILABLE",
            returns=strategy,
            benchmark_returns=benchmark_series,
            positions=position_frame,
            transactions=transaction_frame,
            benchmark=benchmark,
            source=dict(source or {}),
        )


def _period(value: str) -> str:
    result = str(value).upper().replace("-", "_")
    if result not in PERIODS:
        raise PerformanceDataError(f"unsupported performance period: {value}")
    return result


def _date_index(
    values: Any, label: str, *, unique: bool = True
) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if parsed.isna().any():
        raise PerformanceDataError(f"{label} contains invalid dates")
    if parsed.tz is not None:
        parsed = parsed.tz_convert("Asia/Taipei").tz_localize(None)
    parsed = parsed.normalize().tz_localize("UTC")
    if unique and parsed.has_duplicates:
        raise PerformanceDataError(f"{label} contains duplicate dates")
    if not parsed.is_monotonic_increasing:
        raise PerformanceDataError(f"{label} dates must be ascending")
    return parsed


def _return_series(
    values: pd.Series | pd.DataFrame, preferred: str
) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        if "date" in frame:
            frame = frame.set_index("date")
        candidates = [preferred, "daily_return", "returns", "return"]
        column = next((item for item in candidates if item in frame), None)
        if column is None:
            if len(frame.columns) != 1:
                raise PerformanceDataError(f"{preferred} value column is ambiguous")
            column = str(frame.columns[0])
        result = frame[column]
    elif isinstance(values, pd.Series):
        result = values.copy()
    else:
        raise PerformanceDataError(f"{preferred} must be a Series or DataFrame")
    if result.empty:
        raise PerformanceDataError(f"{preferred} is empty")
    result.index = _date_index(result.index, preferred)
    result = pd.to_numeric(result, errors="coerce").astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise PerformanceDataError(f"{preferred} contains NaN or non-finite values")
    result.name = preferred
    return result


def _positions(values: pd.DataFrame | None) -> pd.DataFrame | None:
    if values is None or values.empty:
        return None
    missing = set(POSITION_COLUMNS) - set(values.columns)
    if missing:
        raise PerformanceDataError(
            f"positions missing columns: {', '.join(sorted(missing))}"
        )
    frame = values.loc[:, POSITION_COLUMNS].copy()
    frame["date"] = _date_index(frame["date"], "positions", unique=False)
    frame["ticker"] = frame["ticker"].astype(str)
    if frame.duplicated(["date", "ticker"]).any():
        raise PerformanceDataError("positions contain duplicate date/ticker rows")
    for column in POSITION_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame.loc[:, POSITION_COLUMNS[2:]].to_numpy()).all():
        raise PerformanceDataError("positions contain NaN or non-finite values")
    return frame.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def _transactions(values: pd.DataFrame | None) -> pd.DataFrame | None:
    if values is None or values.empty:
        return None
    missing = set(TRANSACTION_COLUMNS) - set(values.columns)
    if missing:
        raise PerformanceDataError(
            f"transactions missing columns: {', '.join(sorted(missing))}"
        )
    frame = values.loc[:, TRANSACTION_COLUMNS].copy()
    frame["date"] = _date_index(frame["date"], "transactions", unique=False)
    frame["ticker"] = frame["ticker"].astype(str)
    for column in TRANSACTION_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame.loc[:, TRANSACTION_COLUMNS[2:]].to_numpy()).all():
        raise PerformanceDataError("transactions contain NaN or non-finite values")
    return frame.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def load_frozen_identity(root: str | Path) -> FrozenIdentity:
    root = Path(root).resolve()
    spec = FundamentalStrategySpec()
    spec.validate()
    config = spec.as_dict()
    contract_path = (
        root
        / "data/research/fundamental-production-final-v1"
        / "production_runtime_contract.json"
    )
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected = {
            "strategy_id": spec.strategy_id,
            "strategy_fingerprint": spec.fingerprint,
            "factors": list(spec.factors),
            "factor_weighting": spec.factor_weighting,
            "top_n": spec.top_n,
            "rebalance": spec.rebalance,
            "portfolio_weighting": spec.portfolio_weighting,
            "research_knowledge_cutoff": spec.research_knowledge_cutoff,
        }
        if any(contract.get(key) != value for key, value in expected.items()):
            raise FrozenStrategyChanged("FROZEN_STRATEGY_CHANGED")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return FrozenIdentity(
        spec.strategy_id,
        spec.fingerprint,
        config,
        hashlib.sha256(encoded).hexdigest(),
    )


def _load_frame(root: Path, value: str | None) -> pd.DataFrame | None:
    if not value:
        return None
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.exists():
        raise PerformanceDataError(f"canonical source does not exist: {value}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise PerformanceDataError(f"unsupported canonical source format: {value}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_canonical_manifest(
    root: Path, source: dict[str, Any], identity: FrozenIdentity
) -> None:
    value = source.get("canonical_manifest")
    if not value:
        return
    path = (root / str(value)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PerformanceDataError("CANONICAL_BACKTEST_REJECTED:MANIFEST_MISSING")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("strategy_id") != identity.strategy_id
        or manifest.get("strategy_fingerprint") != identity.strategy_fingerprint
        or manifest.get("strategy_config_hash") != identity.digest
    ):
        raise PerformanceDataError(
            "CANONICAL_BACKTEST_REJECTED:FROZEN_STRATEGY_CHANGED"
        )
    if manifest.get("pit_audit_status") != "PASS":
        raise PerformanceDataError("CANONICAL_BACKTEST_REJECTED:BLOCKED_PIT")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise PerformanceDataError("CANONICAL_BACKTEST_REJECTED:HASHES_MISSING")
    for name, expected in hashes.items():
        artifact = (path.parent / str(name)).resolve()
        if (
            not artifact.is_relative_to(path.parent)
            or not artifact.is_file()
            or _file_sha256(artifact) != expected
        ):
            raise PerformanceDataError(
                f"CANONICAL_BACKTEST_REJECTED:ARTIFACT_HASH_MISMATCH:{name}"
            )


def load_repository_performance_data(
    root: str | Path,
    identity: FrozenIdentity | None = None,
) -> dict[str, PerformanceData]:
    """Load only explicitly identity-bound sources; never guess from old artifacts."""

    root = Path(root).resolve()
    identity = identity or load_frozen_identity(root)
    final_dir = root / "data/research/fundamental-production-final-v1"
    eligibility_path = final_dir / "fresh_oos_eligibility_audit.json"
    eligibility = (
        json.loads(eligibility_path.read_text(encoding="utf-8"))
        if eligibility_path.exists()
        else {}
    )
    manifest_path = final_dir / "performance_source_manifest.json"
    if not manifest_path.exists():
        fresh_reason = eligibility.get(
            "reason", "NO_IDENTITY_BOUND_CANONICAL_FRESH_OOS_SOURCE"
        )
        return {
            "BACKTEST": PerformanceData.unavailable(
                "BACKTEST", "NO_IDENTITY_BOUND_CANONICAL_BACKTEST_SOURCE"
            ),
            "HISTORICAL_OOS": PerformanceData.unavailable(
                "HISTORICAL_OOS",
                "NO_IDENTITY_BOUND_CANONICAL_HISTORICAL_OOS_SOURCE",
            ),
            "FRESH_OOS": PerformanceData.unavailable("FRESH_OOS", fresh_reason),
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("strategy_id") != identity.strategy_id
        or manifest.get("strategy_fingerprint") != identity.strategy_fingerprint
    ):
        raise PerformanceDataError("PERFORMANCE_SOURCE_IDENTITY_MISMATCH")
    period_sources = manifest.get("periods")
    if not isinstance(period_sources, dict):
        raise PerformanceDataError("performance source manifest periods are invalid")
    result: dict[str, PerformanceData] = {}
    for period in PERIODS:
        source = period_sources.get(period)
        if period == "FRESH_OOS" and not eligibility.get("fresh_oos_available", False):
            result[period] = PerformanceData.unavailable(
                period,
                eligibility.get(
                    "reason", "NO_LEGAL_UNTOUCHED_WINDOW_WITH_REQUIRED_HISTORY"
                ),
            )
            continue
        if not isinstance(source, dict) or not source.get("returns"):
            result[period] = PerformanceData.unavailable(
                period,
                source.get("reason", f"NO_IDENTITY_BOUND_CANONICAL_{period}_SOURCE")
                if isinstance(source, dict)
                else f"NO_IDENTITY_BOUND_CANONICAL_{period}_SOURCE",
            )
            continue
        _validate_canonical_manifest(root, source, identity)
        result[period] = PerformanceData.from_frames(
            period,
            _load_frame(root, source["returns"]),
            benchmark_returns=_load_frame(root, source.get("benchmark_returns")),
            positions=_load_frame(root, source.get("positions")),
            transactions=_load_frame(root, source.get("transactions")),
            benchmark_id=source.get("benchmark_id"),
            benchmark_source=source.get("benchmark_source"),
            source=source,
        )
    return result


__all__ = [
    "PERIODS",
    "POSITION_COLUMNS",
    "TRANSACTION_COLUMNS",
    "FrozenIdentity",
    "FrozenStrategyChanged",
    "PerformanceData",
    "PerformanceDataError",
    "load_frozen_identity",
    "load_repository_performance_data",
]

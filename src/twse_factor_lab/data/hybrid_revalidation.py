# ruff: noqa: E501
"""One-shot TWSE/yfinance canonical-data revalidation for frozen S3.

This module owns the experiment boundary.  It deliberately does not alter the
processed store, frozen Fresh OOS evidence, or runtime ledgers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.accounting import canonical_replay
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.robustness import compute_metrics
from twse_factor_lab.factors.controlled import build_controlled_price_factors
from twse_factor_lab.strategy.composite_replay import build_composite, build_targets

FINGERPRINT = "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"
WINDOW_START = pd.Timestamp("2026-01-02")
WINDOW_END = pd.Timestamp("2026-08-31")
ATOL = 4.547473508864641e-12
RTOL = 1.4210854715202004e-14
INITIAL_CASH = 1_000_000.0
OUTPUT_NAMESPACE = "canonical-hybrid-revalidation-v1"

SOURCE_ROLES = {
    "RAW_PRICE_AUTHORITY": "TWSE_OFFICIAL",
    "VOLUME_AUTHORITY": "TWSE_OFFICIAL",
    "TRADE_VALUE_AUTHORITY": "TWSE_OFFICIAL",
    "TRADING_CALENDAR_AUTHORITY": "TWSE_OFFICIAL",
    "SECURITY_STATUS_AUTHORITY": "TWSE_OFFICIAL",
    "ADJUSTMENT_FACTOR_PROVIDER": "YFINANCE",
    "SECONDARY_GAP_FILL_PROVIDER": "YFINANCE",
    "RECONCILIATION_PROVIDER": "YFINANCE",
}
LEGITIMATE_NO_DATA = frozenset(
    {"NOT_YET_LISTED", "DELISTED", "SUSPENDED", "LEGITIMATE_NO_TRADE"}
)
GAP_STATUSES = frozenset({"API_MISSING", "OFFICIAL_SOURCE_GAP"})
GAP_COLUMNS = [
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


class HybridRevalidationError(RuntimeError):
    """The fixed hybrid experiment cannot safely continue."""


class SourceAdapter(Protocol):
    source_name: str
    role: str

    def read(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class DataFrameSource:
    """Deterministic test/runner adapter; it never selects another source."""

    frame: pd.DataFrame
    source_name: str
    role: str

    def read(self) -> pd.DataFrame:
        return self.frame.copy(deep=True)


@dataclass(frozen=True)
class ExperimentContract:
    """The complete predeclared experiment contract."""

    source_roles: dict[str, str] = field(
        default_factory=lambda: dict(SOURCE_ROLES)
    )
    candidate_fingerprint: str = FINGERPRINT
    start_date: str = str(WINDOW_START.date())
    end_date: str = str(WINDOW_END.date())
    factors: dict[str, float] = field(
        default_factory=lambda: {
            "L2_AMIHUD_20D": 0.5,
            "L4_DOLLAR_VOLUME_20D": 0.5,
        }
    )
    top_n: int = 5
    weighting: str = "equal_weight"
    rebalance: str = "monthly"
    buffer: bool = False
    execution: str = "signal T -> next valid trading session T+1"
    costs: dict[str, float] = field(
        default_factory=lambda: CostModel().summary()
    )
    atol: float = ATOL
    rtol: float = RTOL
    volume_semantics: str = "TWSE reported shares unchanged"
    trade_value_semantics: str = "TWSE reported trade value unchanged"

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": "canonical-hybrid-revalidation-v1",
            "source_roles": dict(self.source_roles),
            "candidate_fingerprint": self.candidate_fingerprint,
            "fresh_oos_window": {
                "start": self.start_date,
                "end": self.end_date,
            },
            "strategy": {
                "id": "S3",
                "factors": dict(self.factors),
                "top_n": self.top_n,
                "weighting": self.weighting,
                "rebalance": self.rebalance,
                "buffer": self.buffer,
                "execution": self.execution,
            },
            "costs": dict(self.costs),
            "numerical_contract": {"atol": self.atol, "rtol": self.rtol},
            "volume_semantics": self.volume_semantics,
            "trade_value_semantics": self.trade_value_semantics,
            "single_experiment": True,
            "outcome_directed_source_switching": False,
        }

    def validate(self) -> None:
        if self.source_roles != SOURCE_ROLES:
            raise HybridRevalidationError("SOURCE_ROLE_CONTRACT_MISMATCH")
        if self.candidate_fingerprint != FINGERPRINT:
            raise HybridRevalidationError("S3_FINGERPRINT_MISMATCH")
        if (self.start_date, self.end_date) != (
            str(WINDOW_START.date()),
            str(WINDOW_END.date()),
        ):
            raise HybridRevalidationError("FRESH_OOS_WINDOW_IMMUTABLE")
        if self.factors != {
            "L2_AMIHUD_20D": 0.5,
            "L4_DOLLAR_VOLUME_20D": 0.5,
        }:
            raise HybridRevalidationError("S3_FACTOR_WEIGHTS_IMMUTABLE")
        if any(
            [
                self.top_n != 5,
                self.weighting != "equal_weight",
                self.rebalance != "monthly",
                self.buffer is not False,
                self.execution != "signal T -> next valid trading session T+1",
                self.costs != CostModel().summary(),
            ]
        ):
            raise HybridRevalidationError("S3_STRATEGY_IMMUTABLE")


def default_contract() -> ExperimentContract:
    contract = ExperimentContract()
    contract.validate()
    return contract


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".hybrid-", suffix=".parquet", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        pd.read_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def output_dir(root: str | Path) -> Path:
    return Path(root).resolve() / "data" / "research" / OUTPUT_NAMESPACE


def assert_single_experiment(root: str | Path) -> None:
    """Reject a completed namespace instead of silently starting experiment two."""

    if (output_dir(root) / "hybrid_revalidation_verdict.json").exists():
        raise HybridRevalidationError("SECOND_EXPERIMENT_FORBIDDEN")


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "date" not in result or "ticker" not in result:
        raise HybridRevalidationError("SOURCE_SCHEMA_MISMATCH")
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None)
    result["ticker"] = result["ticker"].astype(str).str.replace(".TW", "", regex=False)
    result["ticker"] = result["ticker"].str.strip()
    result = result.dropna(subset=["date", "ticker"])
    return result.sort_values(["date", "ticker"], kind="stable").drop_duplicates(
        ["date", "ticker"], keep="first"
    )


def _numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def normalize_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    frame = _clean_frame(calendar[["date", "ticker"]]) if "ticker" in calendar else calendar.copy()
    if "is_trading_session" in frame:
        frame = frame[frame["is_trading_session"].fillna(False).astype(bool)]
    result = pd.DataFrame({"date": pd.to_datetime(frame["date"]).dt.normalize()})
    result["source"] = calendar.get("source", "TWSE_OFFICIAL")
    result["is_trading_session"] = True
    return result.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def normalize_security_status(status: pd.DataFrame) -> pd.DataFrame:
    frame = status.copy()
    if "ticker" not in frame:
        raise HybridRevalidationError("SECURITY_STATUS_SCHEMA_MISMATCH")
    frame["ticker"] = frame["ticker"].astype(str).str.replace(".TW", "", regex=False)
    if "listed_date" not in frame:
        frame["listed_date"] = pd.NaT
    frame["listed_date"] = pd.to_datetime(frame["listed_date"], errors="coerce")
    if "active" not in frame:
        frame["active"] = True
    if "suspended" not in frame:
        frame["suspended"] = False
    frame["active"] = frame["active"].fillna(True).astype(bool)
    frame["suspended"] = frame["suspended"].fillna(False).astype(bool)
    return frame[
        ["ticker", "listed_date", "active", "suspended"]
    ].drop_duplicates("ticker", keep="last")


def normalize_volume_and_trade_value(frame: pd.DataFrame) -> pd.DataFrame:
    result = _numeric_columns(frame, ["volume", "trade_value"])
    result["volume_normalization"] = "TWSE_REPORTED_SHARES_UNCHANGED"
    result["trade_value_normalization"] = "TWSE_REPORTED_VALUE_UNCHANGED"
    return result


def _secondary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = _clean_frame(frame)
    result = _numeric_columns(
        result,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_value",
            "raw_close",
            "adjusted_close",
            "adj_close",
        ],
    )
    if "raw_close" not in frame:
        result["raw_close"] = result["close"]
    if "adjusted_close" not in frame:
        result["adjusted_close"] = result.get("adj_close", result["close"])
    result["source_sha"] = result.apply(
        lambda row: row.get("source_sha")
        if pd.notna(row.get("source_sha"))
        else value_sha(
            {
                "date": str(row["date"].date()),
                "ticker": row["ticker"],
                "raw_close": row["raw_close"],
                "adjusted_close": row["adjusted_close"],
            }
        ),
        axis=1,
    )
    return result


def _is_traded(row: pd.Series) -> bool:
    return bool(
        pd.notna(row.get("raw_close"))
        and float(row.get("raw_close")) > 0
        and pd.notna(row.get("volume"))
        and float(row.get("volume")) > 0
    )


def _approx_equal(left: Any, right: Any, atol: float = ATOL, rtol: float = RTOL) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(abs(float(left) - float(right)) <= atol + rtol * max(abs(float(left)), abs(float(right)), 1.0))


def corporate_action_normalizer(
    raw: pd.DataFrame, adjusted: pd.DataFrame, factors: pd.DataFrame
) -> dict[str, Any]:
    """Trace factor changes generically; no security-specific branches."""

    raw_close = raw.pivot(index="date", columns="ticker", values="close").sort_index()
    adj_close = adjusted.pivot(index="date", columns="ticker", values="close").sort_index()
    factor_matrix = factors.pivot(index="date", columns="ticker", values="adjustment_factor").sort_index()
    rows: list[dict[str, Any]] = []
    for ticker in factor_matrix.columns:
        series = factor_matrix[ticker].dropna()
        changes = series.div(series.shift(1)).sub(1).abs()
        for date in changes.index[changes.gt(0.02).fillna(False)]:
            raw_return = raw_close.get(ticker, pd.Series(dtype=float)).pct_change().get(date)
            adjusted_return = adj_close.get(ticker, pd.Series(dtype=float)).pct_change().get(date)
            action = "CORPORATE_ACTION_FACTOR_CHANGE"
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action_type": action,
                    "factor_before": series.shift(1).get(date),
                    "factor_after": series.get(date),
                    "raw_return": raw_return,
                    "adjusted_return": adjusted_return,
                    "false_return_shock": bool(pd.notna(adjusted_return) and abs(float(adjusted_return)) > 0.5),
                    "status": "PASS" if pd.isna(adjusted_return) or abs(float(adjusted_return)) <= 0.5 else "REVIEW_REQUIRED",
                }
            )
    trace = pd.DataFrame(rows)
    if trace.empty:
        trace = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "action_type",
                "factor_before",
                "factor_after",
                "raw_return",
                "adjusted_return",
                "false_return_shock",
                "status",
            ]
        )
    return {
        "status": "PASS" if not trace.get("status", pd.Series(dtype=str)).eq("REVIEW_REQUIRED").any() else "REVIEW_REQUIRED",
        "events": trace,
        "event_count": int(len(trace)),
        "generic_rule": True,
        "ticker_specific_rules": [],
        "validated_event_types": ["split", "capital_reduction", "ex_right", "ex_dividend"],
    }


class HybridCanonicalBuilder:
    """Build one fixed-role canonical dataset from supplied source frames."""

    def __init__(
        self,
        root: str | Path,
        twse: SourceAdapter,
        yfinance: SourceAdapter,
        *,
        contract: ExperimentContract | None = None,
        calendar: pd.DataFrame | None = None,
        security_status: pd.DataFrame | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.twse = twse
        self.yfinance = yfinance
        self.contract = contract or default_contract()
        self.calendar = calendar
        self.security_status = security_status
        self.contract.validate()
        if (twse.role, yfinance.role) != ("TWSE_OFFICIAL", "YFINANCE"):
            raise HybridRevalidationError("SOURCE_ROLE_CONTRACT_MISMATCH")

    def build(self) -> dict[str, Any]:
        primary = normalize_volume_and_trade_value(_clean_frame(self.twse.read()))
        secondary = _secondary_columns(self.yfinance.read())
        dates = (
            normalize_calendar(self.calendar)
            if self.calendar is not None
            else normalize_calendar(primary[["date"]].drop_duplicates())
        )
        status = normalize_security_status(
            self.security_status
            if self.security_status is not None
            else pd.DataFrame({"ticker": sorted(set(primary["ticker"]) | set(secondary["ticker"]))})
        )
        tickers = sorted(set(status["ticker"]) | set(primary["ticker"]) | set(secondary["ticker"]))
        expected = pd.MultiIndex.from_product(
            [dates["date"], tickers], names=["date", "ticker"]
        ).to_frame(index=False)
        expected = expected.merge(status, on="ticker", how="left")
        expected["listed_date"] = pd.to_datetime(expected["listed_date"], errors="coerce")
        expected["primary_status"] = "API_MISSING"
        expected.loc[expected["listed_date"].notna() & (expected["date"] < expected["listed_date"]), "primary_status"] = "NOT_YET_LISTED"
        expected.loc[~expected["active"].fillna(True), "primary_status"] = "DELISTED"
        expected.loc[expected["suspended"].fillna(False), "primary_status"] = "SUSPENDED"
        primary_columns = ["date", "ticker", "open", "high", "low", "close", "volume", "trade_value", "source", "payload_sha"]
        primary_columns += ["volume_normalization", "trade_value_normalization"]
        primary = primary[[column for column in primary_columns if column in primary.columns]].copy()
        primary["_primary_row_present"] = True
        primary = expected.merge(primary, on=["date", "ticker"], how="left", suffixes=("", "_primary"))
        has_price = primary[["open", "high", "low", "close"]].notna().any(axis=1)
        has_primary_row = primary["_primary_row_present"].fillna(False).astype(bool)
        primary.loc[has_price, "primary_status"] = "PRIMARY_PRESENT"
        primary.loc[~has_price & has_primary_row, "primary_status"] = "LEGITIMATE_NO_TRADE"
        primary = primary.merge(
            secondary.add_suffix("_secondary"),
            left_on=["date", "ticker"],
            right_on=["date_secondary", "ticker_secondary"],
            how="left",
        )
        fill_rows: list[dict[str, Any]] = []
        for field_name in ["open", "high", "low", "close", "volume", "trade_value"]:
            secondary_field = f"{field_name}_secondary"
            if secondary_field not in primary:
                continue
            eligible = primary["primary_status"].isin(GAP_STATUSES) & primary[field_name].isna() & primary[secondary_field].notna()
            for index in primary.index[eligible]:
                row = primary.loc[index]
                value = row[secondary_field]
                primary.at[index, field_name] = value
                fill_rows.append(
                    {
                        "date": row["date"],
                        "ticker": row["ticker"],
                        "field": field_name,
                        "primary_source": "TWSE_OFFICIAL",
                        "secondary_source": "YFINANCE",
                        "reason": "OFFICIAL_SOURCE_GAP_SECONDARY_FILL",
                        "primary_status": row["primary_status"],
                        "secondary_value": value,
                        "source_sha": row.get("source_sha_secondary"),
                    }
                )
        secondary_raw_close = primary["raw_close_secondary"] if "raw_close_secondary" in primary else primary["close_secondary"]
        secondary_adjusted_close = primary["adjusted_close_secondary"] if "adjusted_close_secondary" in primary else primary["adj_close_secondary"] if "adj_close_secondary" in primary else primary["close_secondary"]
        primary["adjustment_factor"] = pd.to_numeric(secondary_adjusted_close, errors="coerce").div(pd.to_numeric(secondary_raw_close, errors="coerce").replace(0, np.nan))
        primary["adjustment_factor"] = primary["adjustment_factor"].replace([np.inf, -np.inf], np.nan)
        primary["adjustment_status"] = np.where(
            primary["primary_status"].isin(LEGITIMATE_NO_DATA),
            primary["primary_status"],
            np.where(primary["adjustment_factor"].gt(0) & primary["close"].notna(), "MATCHED", "MISSING_FACTOR"),
        )
        traded = primary["close"].notna() & primary["close"].gt(0) & primary["volume"].fillna(0).gt(0)
        for field_name in ["open", "high", "low", "close"]:
            primary[f"raw_{field_name}"] = primary[field_name]
            primary[field_name] = primary[field_name].where(~traded | primary["adjustment_factor"].gt(0), np.nan)
            primary.loc[traded & primary["adjustment_factor"].gt(0), field_name] = primary.loc[traded & primary["adjustment_factor"].gt(0), field_name] * primary.loc[traded & primary["adjustment_factor"].gt(0), "adjustment_factor"]
        primary["price_source"] = np.where(primary["primary_status"].eq("PRIMARY_PRESENT"), "TWSE_OFFICIAL_RAW", "YFINANCE_SECONDARY_GAP_FILL")
        primary["adjustment_source"] = "YFINANCE_ADJ_CLOSE_CLOSE"
        primary["volume_source"] = np.where(primary["volume_secondary"].notna() & primary["volume"].eq(primary["volume_secondary"]), "YFINANCE_SECONDARY_GAP_FILL", "TWSE_OFFICIAL")
        primary["trade_value_source"] = "TWSE_OFFICIAL"
        primary["source_sha"] = primary["source_sha_secondary"].fillna(value_sha({"source": "TWSE_OFFICIAL"})) if "source_sha_secondary" in primary else value_sha({"source": "TWSE_OFFICIAL"})
        columns = [
            "date", "ticker", "open", "high", "low", "close", "volume", "trade_value",
            "raw_open", "raw_high", "raw_low", "raw_close", "primary_status", "adjustment_factor",
            "adjustment_status", "price_source", "adjustment_source", "volume_source", "trade_value_source",
            "volume_normalization", "trade_value_normalization", "source_sha",
        ]
        canonical = primary[columns].sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
        factors = canonical[["date", "ticker", "raw_close", "adjustment_factor", "adjustment_status", "adjustment_source", "source_sha"]].copy()
        factors["factor_formula"] = "yfinance_adjusted_close / yfinance_raw_close"
        factors["volume_semantics"] = self.contract.volume_semantics
        gap_log = pd.DataFrame(fill_rows, columns=GAP_COLUMNS)
        if gap_log.empty:
            gap_log = pd.DataFrame(columns=GAP_COLUMNS)
        action = corporate_action_normalizer(
            canonical[["date", "ticker", "raw_close"]].rename(columns={"raw_close": "close"}),
            canonical[["date", "ticker", "close"]],
            factors,
        )
        out = output_dir(self.root)
        _atomic_parquet(out / "hybrid_canonical_ohlcv.parquet", canonical)
        _atomic_parquet(out / "hybrid_gap_fill_log.parquet", gap_log)
        _atomic_parquet(out / "hybrid_adjustment_factors.parquet", factors)
        _atomic_parquet(out / "hybrid_security_status.parquet", status)
        _atomic_parquet(out / "hybrid_calendar.parquet", dates)
        _atomic_parquet(out / "hybrid_corporate_action_trace.parquet", action["events"])
        manifest = {
            "namespace": OUTPUT_NAMESPACE,
            "contract_version": "canonical-hybrid-revalidation-v1",
            "candidate_fingerprint": self.contract.candidate_fingerprint,
            "date_range": {"start": self.contract.start_date, "end": self.contract.end_date},
            "sessions": int(dates["date"].between(WINDOW_START, WINDOW_END).sum()),
            "tickers": int(canonical["ticker"].nunique()),
            "rows": int(len(canonical)),
            "gap_fills": int(len(gap_log)),
            "adjustment_status": canonical["adjustment_status"].value_counts(dropna=False).to_dict(),
            "volume_status": self.contract.volume_semantics,
            "trade_value_status": self.contract.trade_value_semantics,
            "corporate_action": {key: value for key, value in action.items() if key != "events"},
            "artifacts": {},
        }
        for path in sorted(out.glob("*.parquet")):
            manifest["artifacts"][path.name] = file_sha256(path)
        _atomic_json(out / "hybrid_data_manifest.json", manifest)
        return {
            "canonical": canonical,
            "factors": factors,
            "gap_log": gap_log,
            "calendar": dates,
            "security_status": status,
            "manifest": manifest,
            "corporate_action": action,
        }


def _matrix(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    result = frame.pivot(index="date", columns="ticker", values=field).sort_index()
    result.index = pd.DatetimeIndex(result.index)
    return result.sort_index(axis=1)


def _compare_frames(left: pd.DataFrame, right: pd.DataFrame, fields: list[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for field_name in fields:
        left_keys = left[["date", "ticker", field_name]].rename(columns={field_name: "old"})
        right_keys = right[["date", "ticker", field_name]].rename(columns={field_name: "hybrid"})
        merged = left_keys.merge(right_keys, on=["date", "ticker"], how="outer")
        matches = merged.apply(lambda row: _approx_equal(row["old"], row["hybrid"]), axis=1)
        rows[field_name] = {
            "status": "PASS" if matches.all() else "FAIL",
            "mismatch_count": int((~matches).sum()),
            "row_count": int(len(merged)),
            "examples": _jsonable(merged.loc[~matches].head(10).to_dict("records")),
        }
    return rows


def reconcile_data_layers(old: pd.DataFrame, hybrid: pd.DataFrame) -> dict[str, Any]:
    old = _clean_frame(old)
    hybrid = _clean_frame(hybrid)
    for frame in (old, hybrid):
        frame.drop(frame.index[~frame["date"].between(WINDOW_START, WINDOW_END)], inplace=True)
    calendar_old = set(old["date"].dropna())
    calendar_new = set(hybrid["date"].dropna())
    universe_old = set(old["ticker"].dropna())
    universe_new = set(hybrid["ticker"].dropna())
    comparisons = _compare_frames(old, hybrid, ["open", "high", "low", "close", "volume", "trade_value"])
    old_close, new_close = _matrix(old, "close"), _matrix(hybrid, "close")
    old_return = old_close.pct_change().stack(future_stack=True).rename("old").reset_index()
    new_return = new_close.pct_change().stack(future_stack=True).rename("hybrid").reset_index()
    returns = old_return.merge(new_return, on=["date", "ticker"], how="outer")
    return_match = returns.apply(lambda row: _approx_equal(row["old"], row["hybrid"]), axis=1)
    calendar = {
        "status": "PASS" if calendar_old == calendar_new else "FAIL",
        "old_sessions": len(calendar_old),
        "hybrid_sessions": len(calendar_new),
        "missing_from_hybrid": sorted(str(item.date()) for item in calendar_old - calendar_new),
        "unexpected_in_hybrid": sorted(str(item.date()) for item in calendar_new - calendar_old),
    }
    universe = {
        "status": "PASS" if universe_old == universe_new else "FAIL",
        "old_tickers": len(universe_old),
        "hybrid_tickers": len(universe_new),
        "missing_from_hybrid": sorted(universe_old - universe_new),
        "unexpected_in_hybrid": sorted(universe_new - universe_old),
    }
    return {
        "window": {"start": str(WINDOW_START.date()), "end": str(WINDOW_END.date())},
        "calendar": calendar,
        "universe": universe,
        "adjusted_ohlc": {field: comparisons[field] for field in ["open", "high", "low", "close"]},
        "volume": comparisons["volume"],
        "trade_value": comparisons["trade_value"],
        "daily_return": {
            "status": "PASS" if return_match.all() else "FAIL",
            "mismatch_count": int((~return_match).sum()),
            "row_count": int(len(returns)),
        },
        "status": "PASS" if calendar["status"] == universe["status"] == "PASS" and all(item["status"] == "PASS" for item in comparisons.values()) and return_match.all() else "FAIL",
    }


def factor_reconciliation(old_close: pd.DataFrame, old_volume: pd.DataFrame, hybrid_close: pd.DataFrame, hybrid_volume: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = old_close.index.union(hybrid_close.index).sort_values()
    columns = old_close.columns.union(hybrid_close.columns).sort_values()
    old_close, hybrid_close = old_close.reindex(index=index, columns=columns), hybrid_close.reindex(index=index, columns=columns)
    old_volume, hybrid_volume = old_volume.reindex(index=index, columns=columns), hybrid_volume.reindex(index=index, columns=columns)
    old_factors = build_controlled_price_factors(old_close, old_volume)
    hybrid_factors = build_controlled_price_factors(hybrid_close, hybrid_volume)
    rows: list[pd.DataFrame] = []
    for name in ("L2_AMIHUD_20D", "L4_DOLLAR_VOLUME_20D"):
        row = pd.DataFrame({"date": index.repeat(len(columns)), "ticker": list(columns) * len(index)})
        row["old"] = old_factors[name].stack(future_stack=True).to_numpy()
        row["hybrid"] = hybrid_factors[name].stack(future_stack=True).to_numpy()
        row["factor"] = name
        row["match"] = [
            _approx_equal(left, right) for left, right in zip(row["old"], row["hybrid"], strict=True)
        ]
        rows.append(row)
    result = pd.concat(rows, ignore_index=True)
    result = result[result["date"].between(WINDOW_START, WINDOW_END)].reset_index(drop=True)
    summary = {
        "L2_mismatch_count": int((~result.loc[result["factor"].eq("L2_AMIHUD_20D"), "match"]).sum()),
        "L4_mismatch_count": int((~result.loc[result["factor"].eq("L4_DOLLAR_VOLUME_20D"), "match"]).sum()),
        "status": "PASS" if result["match"].all() else "FAIL",
    }
    return result, summary


def _target_signature(targets: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    if targets.empty:
        return {}
    frame = targets.copy()
    frame["execution_date"] = pd.to_datetime(frame["execution_date"])
    return {
        date: dict(zip(group["ticker"].astype(str), group["target_weight"].astype(float), strict=True))
        for date, group in frame.groupby("execution_date", sort=True)
    }


def signal_reconciliation(old_close: pd.DataFrame, old_volume: pd.DataFrame, hybrid_close: pd.DataFrame, hybrid_volume: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    old_score, _ = build_composite(old_close, old_volume)
    hybrid_score, _ = build_composite(hybrid_close, hybrid_volume)
    old_targets, _ = build_targets(old_score, rebalance="monthly", buffer_on=False)
    hybrid_targets, _ = build_targets(hybrid_score, rebalance="monthly", buffer_on=False)
    old_sig, hybrid_sig = _target_signature(old_targets), _target_signature(hybrid_targets)
    dates = sorted(set(old_sig) | set(hybrid_sig))
    top5_mismatch = []
    target_mismatch = 0
    for date in dates:
        old_names = set(old_sig.get(date, {}))
        new_names = set(hybrid_sig.get(date, {}))
        if old_names != new_names:
            top5_mismatch.append(date)
        keys = old_names | new_names
        target_mismatch += sum(not _approx_equal(old_sig.get(date, {}).get(key, 0.0), hybrid_sig.get(date, {}).get(key, 0.0)) for key in keys)
    report = {
        "status": "PASS" if not top5_mismatch and target_mismatch == 0 else "FAIL",
        "top5_mismatch_count": len(top5_mismatch),
        "target_mismatch_count": target_mismatch,
        "top5_mismatch_dates": [str(date.date()) for date in top5_mismatch],
        "strategy": default_contract().as_dict()["strategy"],
    }
    return report, old_targets, hybrid_targets


def _metrics(close: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
    from twse_factor_lab.backtest.vectorbt_engine import _weights_matrix

    close = close.loc[close.index.intersection(pd.DatetimeIndex(pd.to_datetime(close.index).sort_values()))]
    weights = _weights_matrix(targets, close.index, close.columns)
    results, returns, turnover, _ = canonical_replay(close, weights, CostModel(), INITIAL_CASH)
    metrics = compute_metrics(returns, turnover=turnover, exposure=results["exposure"])
    return {**metrics, "exposure": metrics["avg_exposure"]}


def performance_reconciliation(old_close: pd.DataFrame, hybrid_close: pd.DataFrame, old_targets: pd.DataFrame, hybrid_targets: pd.DataFrame) -> dict[str, Any]:
    old_metrics = _metrics(old_close, old_targets)
    hybrid_metrics = _metrics(hybrid_close, hybrid_targets)
    fields = ["total_return", "cagr", "sharpe", "sortino", "max_drawdown", "turnover", "exposure"]
    comparisons = {}
    for field_name in fields:
        left, right = old_metrics.get(field_name), hybrid_metrics.get(field_name)
        comparisons[field_name] = {
            "old": left,
            "hybrid": right,
            "match": _approx_equal(left, right),
        }
    return {
        "classification": "NUMERICALLY_EQUIVALENT" if all(row["match"] for row in comparisons.values()) else "MATERIAL_DIFFERENCE",
        "metrics": comparisons,
        "numerical_contract": {"atol": ATOL, "rtol": RTOL},
    }


def evaluate_verdict(
    *,
    contract_status: str,
    data_report: dict[str, Any],
    factor_report: dict[str, Any],
    signal_report: dict[str, Any],
    performance_report: dict[str, Any] | None,
    gap_fill_governance: bool,
    adjustment_status: str,
    volume_status: str,
) -> dict[str, Any]:
    gates = {
        "data_contract_resolved": contract_status == "PASS",
        "calendar_reconciliation": data_report.get("calendar", {}).get("status") == "PASS",
        "gap_fill_governance": gap_fill_governance,
        "adjustment": adjustment_status == "PASS",
        "volume_semantics": volume_status == "PASS",
        "top5_mismatch_zero": signal_report.get("top5_mismatch_count") == 0,
        "target_mismatch_zero": signal_report.get("target_mismatch_count") == 0,
        "performance": performance_report is not None and performance_report.get("classification") == "NUMERICALLY_EQUIVALENT",
        "factor_reconciliation": factor_report.get("status") == "PASS",
    }
    passed = all(gates.values())
    return {
        "verdict": "CANONICAL_HYBRID_DATA_APPROVED" if passed else "HYBRID_REVALIDATION_FAILED",
        "gates": gates,
        "REVALIDATE_S3_ON_NEW_DATASET_REQUIRED": not passed,
        "READY_FOR_FORWARD_SHADOW": passed,
        "production_ready": False,
        "strategy_changed": "NO",
        "factor_changed": "NO",
        "fresh_oos_window_changed": "NO",
        "historical_evidence_changed": "NO",
        "second_experiment_executed": "NO",
    }


def write_report(root: str | Path, verdict: dict[str, Any], manifest: dict[str, Any]) -> None:
    out = output_dir(root)
    artifact_count = sum(
        1 for path in out.iterdir() if path.is_file() and path.name != "run_manifest.json"
    )
    lines = [
        "# Hybrid Canonical S3 Revalidation v1",
        "",
        f"verdict = {verdict['verdict']}",
        "single experiment = YES",
        "TWSE role = raw OHLCV, volume, trade value, calendar, security status authority",
        "yfinance role = adjustment factor, explicit secondary gap fill, reconciliation",
        "",
        f"READY_FOR_FORWARD_SHADOW = {'YES' if verdict['READY_FOR_FORWARD_SHADOW'] else 'NO'}",
        "production_ready = NO",
        "",
        "Governance:",
        "strategy changed = NO",
        "factor changed = NO",
        "Fresh OOS window changed = NO",
        "historical evidence changed = NO",
        "second experiment executed = NO",
        "",
        f"artifact count = {artifact_count}",
    ]
    (out / "hybrid_revalidation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "ATOL",
    "DataFrameSource",
    "ExperimentContract",
    "FINGERPRINT",
    "HybridCanonicalBuilder",
    "HybridRevalidationError",
    "OUTPUT_NAMESPACE",
    "SOURCE_ROLES",
    "WINDOW_END",
    "WINDOW_START",
    "corporate_action_normalizer",
    "default_contract",
    "evaluate_verdict",
    "factor_reconciliation",
    "file_sha256",
    "assert_single_experiment",
    "output_dir",
    "performance_reconciliation",
    "reconcile_data_layers",
    "signal_reconciliation",
    "value_sha",
]

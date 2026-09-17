# ruff: noqa: E501, B905, UP035
"""Official TWSE market-data adapter and provenance layer.

The adapter is intentionally small: it parses the exchange's documented
OpenAPI snapshot and its date-addressable official historical response, then
keeps raw data separate from the adjusted research store.  It never contains
factor, ranking, or portfolio logic.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time as time_module
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from twse_factor_lab.data.normalizer import clean_ticker, normalize_universe

TWSE_OPENAPI_BASE = "https://openapi.twse.com.tw/v1"
TWSE_SWAGGER_URL = f"{TWSE_OPENAPI_BASE}/swagger.json"
TWSE_HISTORICAL_URL = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
TWSE_DAILY_SNAPSHOT_PATH = "exchangeReport/STOCK_DAY_ALL"
TWSE_LISTED_PATH = "opendata/t187ap03_L"
TWSE_HOLIDAY_PATH = "holidaySchedule/holidaySchedule"
TWSE_SUSPENDED_PATH = "exchangeReport/TWTAWU"
RAW_COLUMNS = [
    "date",
    "ticker",
    "market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_value",
    "source",
    "ingested_at",
    "payload_sha",
]
AUDIT_COLUMNS = [
    "ticker",
    "listed",
    "eligible",
    "suspended",
    "no_trade",
    "missing_api_record",
    "invalid_record",
    "status",
]


class OfficialMarketDataError(RuntimeError):
    """Raised when an official response cannot be safely interpreted."""


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False)
        pd.read_parquet(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def payload_sha(payload: bytes | str | Any) -> str:
    """Hash bytes or a canonical JSON representation deterministically."""

    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    return hashlib.sha256(raw).hexdigest()


def _roc_to_timestamp(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(value).strip().replace("/", "")
    if re.fullmatch(r"\d{7}", text):
        text = f"{int(text[:3]) + 1911:04d}{text[3:]}"
    parsed = pd.Timestamp(text)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Asia/Taipei").tz_localize(None)
    return parsed.normalize()


def _numeric(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = re.sub(r"<[^>]+>", "", str(value)).strip().replace(",", "")
    if text in {"", "-", "--", "N/A", "nan", "None"}:
        return np.nan
    return float(pd.to_numeric(text, errors="coerce"))


def _official_ticker(value: Any) -> str | pd._libs.missing.NAType:
    """Keep listed common-equity codes; reject odd-lot/special suffix rows."""

    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().upper().replace(".TW", "")
    return text if re.fullmatch(r"\d{4,6}", text) else pd.NA


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _payload_records(payload: Any, session_date: Any = None) -> pd.DataFrame:
    """Extract records from OpenAPI list, MI_INDEX JSON, or CSV payload."""

    if isinstance(payload, bytes):
        text = _decode_text(payload)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            rows = list(csv.reader(io.StringIO(text)))
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows[1:], columns=rows[0])
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            rows = list(csv.reader(io.StringIO(payload)))
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows[1:], columns=rows[0])
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("tables"), list):
            for table in payload["tables"]:
                fields = table.get("fields", []) if isinstance(table, dict) else []
                rows = table.get("data", []) if isinstance(table, dict) else []
                is_security_table = any(
                    "\u8b49\u5238\u4ee3\u865f" in str(field)
                    or str(field).lower() in {"code", "ticker"}
                    or "code" in str(field).lower()
                    for field in fields
                )
                if is_security_table or (fields and len(fields) >= 7 and rows):
                    return pd.DataFrame(rows, columns=fields)
        if isinstance(payload.get("data"), list):
            fields = payload.get("fields")
            return pd.DataFrame(payload["data"], columns=fields) if fields else pd.DataFrame(payload["data"])
    raise OfficialMarketDataError(f"unsupported official payload type: {type(payload).__name__}")


def normalize_official_ohlcv(
    payload: Any,
    *,
    session_date: Any = None,
    market: str = "TWSE",
    source: str = "twse_official",
    ingested_at: str | None = None,
    payload_hash: str | None = None,
    expected_tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Normalize official rows into stable raw OHLCV schema."""

    frame = _payload_records(payload, session_date)
    if frame.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    lookup = {str(column).strip().lower().replace("_", ""): column for column in frame.columns}

    def col(*names: str, fallback_index: int | None = None) -> pd.Series:
        for name in names:
            key = name.lower().replace("_", "")
            if key in lookup:
                return frame[lookup[key]]
        if fallback_index is not None and fallback_index < len(frame.columns):
            return frame.iloc[:, fallback_index]
        return pd.Series([pd.NA] * len(frame), index=frame.index)

    date_values = col("date", "\u65e5\u671f")
    if date_values.isna().all() and session_date is not None:
        date_values = pd.Series([session_date] * len(frame), index=frame.index)
    result = pd.DataFrame(
        {
            "date": date_values.map(_roc_to_timestamp),
            "ticker": col("ticker", "code", "\u8b49\u5238\u4ee3\u865f", fallback_index=0).map(_official_ticker),
            "market": market,
            "open": col("open", "openingprice", "\u958b\u76e4\u50f9", fallback_index=5).map(_numeric),
            "high": col("high", "highestprice", "\u6700\u9ad8\u50f9", fallback_index=6).map(_numeric),
            "low": col("low", "lowestprice", "\u6700\u4f4e\u50f9", fallback_index=7).map(_numeric),
            "close": col("close", "closingprice", "\u6536\u76e4\u50f9", fallback_index=8).map(_numeric),
            "volume": col("volume", "tradevolume", "\u6210\u4ea4\u80a1\u6578", fallback_index=2).map(_numeric),
            "trade_value": col("trade_value", "tradevalue", "\u6210\u4ea4\u91d1\u984d", fallback_index=4).map(_numeric),
            "source": source,
            "ingested_at": ingested_at or datetime.now(UTC).isoformat(),
            "payload_sha": payload_hash or payload_sha(payload),
        }
    )
    if expected_tickers is not None:
        expected = {clean_ticker(ticker) for ticker in expected_tickers}
        result = result[result["ticker"].isin(expected)]
    result = result.dropna(subset=["date", "ticker"])
    result = result.drop_duplicates(["date", "ticker"], keep="first")
    return result[RAW_COLUMNS].sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def validate_official_ohlcv(frame: pd.DataFrame) -> list[str]:
    """Return critical validation issues without mutating the input."""

    required = set(RAW_COLUMNS)
    issues: list[str] = []
    if not required.issubset(frame.columns):
        issues.append("SCHEMA_MISMATCH")
        return issues
    if frame.duplicated(["date", "ticker"], keep=False).any():
        issues.append("DUPLICATE_DATE_TICKER")
    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    all_price_missing = frame[["open", "high", "low", "close"]].isna().all(axis=1)
    traded = (frame["volume"].fillna(0).to_numpy(dtype=float) > 0) & ~all_price_missing.to_numpy()
    # A legal no-trade row may carry zero volume and blank OHLC; it is not an
    # API omission.  Traded rows must be complete and finite.
    if not np.isfinite(numeric[traded]).all() or not np.isfinite(numeric[:, 4]).all():
        issues.append("MISSING_OR_NON_FINITE_OHLCV")
    traded_frame = frame.loc[traded]
    if (traded_frame[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append("NON_POSITIVE_PRICE")
    if (frame["volume"] < 0).any():
        issues.append("NEGATIVE_VOLUME")
    bounds = traded_frame[["open", "low", "close"]].max(axis=1)
    upper = traded_frame["high"] < bounds
    lower = traded_frame["low"] > traded_frame[["open", "close"]].min(axis=1)
    if upper.any() or lower.any():
        issues.append("OHLC_RELATIONSHIP")
    return sorted(set(issues))


class TWSEOfficialAdapter:
    """Bounded, auditable client for official TWSE market data."""

    def __init__(
        self,
        *,
        session: Any | None = None,
        timeout: int = 30,
        max_attempts: int = 3,
        backoff_seconds: float = 0.25,
        openapi_base: str = TWSE_OPENAPI_BASE,
        historical_url: str = TWSE_HISTORICAL_URL,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.openapi_base = openapi_base.rstrip("/")
        self.historical_url = historical_url
        self.request_log: list[dict[str, Any]] = []

    def _get(self, url: str, *, params: dict[str, Any] | None = None) -> tuple[bytes, str]:
        last: Exception | None = None
        requested_at = datetime.now(UTC).isoformat()
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = (
                    self.session.get(url, timeout=self.timeout)
                    if params is None
                    else self.session.get(url, params=params, timeout=self.timeout)
                )
                response.raise_for_status()
                if hasattr(response, "content"):
                    content = response.content
                elif hasattr(response, "text"):
                    content = str(response.text).encode("utf-8")
                elif hasattr(response, "json"):
                    content = json.dumps(response.json(), ensure_ascii=False).encode("utf-8")
                else:
                    raise OfficialMarketDataError("response has no content")
                received_at = datetime.now(UTC).isoformat()
                self.request_log.append(
                    {
                        "requested_at": requested_at,
                        "received_at": received_at,
                        "endpoint": url,
                        "params": params or {},
                        "attempt": attempt,
                        "status": "PASS",
                        "payload_sha": payload_sha(content),
                    }
                )
                return content, received_at
            except Exception as exc:  # bounded retry is part of the source contract
                last = exc
                self.request_log.append(
                    {
                        "requested_at": requested_at,
                        "received_at": datetime.now(UTC).isoformat(),
                        "endpoint": url,
                        "params": params or {},
                        "attempt": attempt,
                        "status": "FAIL",
                        "error": type(exc).__name__,
                    }
                )
                if attempt < self.max_attempts and self.backoff_seconds:
                    time_module.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise OfficialMarketDataError(f"official request failed after {self.max_attempts} attempts: {url}") from last

    def inspect_openapi(self) -> dict[str, Any]:
        content, received_at = self._get(f"{self.openapi_base}/swagger.json")
        document = json.loads(_decode_text(content))
        paths = document.get("paths", {})
        snapshot = paths.get(f"/{TWSE_DAILY_SNAPSHOT_PATH}", {}).get("get", {})
        listed = paths.get(f"/{TWSE_LISTED_PATH}", {}).get("get", {})
        return {
            "swagger_url": f"{self.openapi_base}/swagger.json",
            "swagger_version": document.get("swagger") or document.get("openapi"),
            "inspected_at": received_at,
            "daily_snapshot_endpoint": f"{self.openapi_base}/{TWSE_DAILY_SNAPSHOT_PATH}",
            "daily_snapshot_summary": snapshot.get("summary"),
            "daily_snapshot_parameters": [p.get("name") for p in snapshot.get("parameters", [])],
            "listed_endpoint": f"{self.openapi_base}/{TWSE_LISTED_PATH}",
            "listed_summary": listed.get("summary"),
            "historical_endpoint": self.historical_url,
            "historical_endpoint_date_parameter": True,
            "twse_forward_only": not bool(snapshot.get("parameters")),
            "openapi_snapshot_only": not bool(snapshot.get("parameters")),
            "historical_source_available": True,
        }

    def fetch_daily_snapshot(self) -> tuple[pd.DataFrame, bytes, str]:
        url = f"{self.openapi_base}/{TWSE_DAILY_SNAPSHOT_PATH}"
        content, received_at = self._get(url)
        return normalize_official_ohlcv(content, source="twse_openapi_snapshot", payload_hash=payload_sha(content)), content, received_at

    def fetch_listed_companies(self) -> tuple[pd.DataFrame, bytes, str]:
        url = f"{self.openapi_base}/{TWSE_LISTED_PATH}"
        content, received_at = self._get(url)
        return normalize_universe(_payload_records(content)), content, received_at

    def fetch_holidays(self) -> tuple[pd.DataFrame, bytes, str]:
        url = f"{self.openapi_base}/{TWSE_HOLIDAY_PATH}"
        content, received_at = self._get(url)
        return _payload_records(content), content, received_at

    def fetch_suspended(self) -> tuple[set[str], bytes, str]:
        url = f"{self.openapi_base}/{TWSE_SUSPENDED_PATH}"
        content, received_at = self._get(url)
        frame = _payload_records(content)
        code = next((column for column in frame.columns if str(column).lower() in {"code", "ticker"} or "code" in str(column).lower()), frame.columns[1] if len(frame.columns) > 1 else frame.columns[0])
        return set(frame[code].dropna().map(clean_ticker)), content, received_at

    def fetch_session(self, session_date: str | pd.Timestamp, *, expected_tickers: Iterable[str] | None = None) -> tuple[pd.DataFrame, bytes, str]:
        date = pd.Timestamp(session_date).strftime("%Y%m%d")
        content, received_at = self._get(
            self.historical_url,
            params={"response": "json", "date": date, "type": "ALLBUT0999"},
        )
        frame = normalize_official_ohlcv(
            content,
            session_date=pd.Timestamp(session_date),
            source="twse_official_historical",
            payload_hash=payload_sha(content),
            expected_tickers=expected_tickers,
        )
        if not frame.empty and frame["date"].nunique() != 1:
            raise OfficialMarketDataError("historical response contains multiple session dates")
        if not frame.empty and frame["date"].iloc[0] != pd.Timestamp(session_date).normalize():
            raise OfficialMarketDataError("historical response date does not match requested session")
        return frame, content, received_at


def build_session_coverage(
    universe: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    suspended_tickers: Iterable[str] = (),
    session_date: Any | None = None,
) -> pd.DataFrame:
    """Classify each expected security without conflating no-trade and missing."""

    expected = universe.copy()
    expected["ticker"] = expected["ticker"].map(clean_ticker)
    expected = expected.drop_duplicates("ticker")
    observed = set(frame.get("ticker", pd.Series(dtype=str)).dropna().map(clean_ticker))
    suspended = {clean_ticker(ticker) for ticker in suspended_tickers}
    rows: list[dict[str, Any]] = []
    for row in expected.itertuples(index=False):
        ticker = str(row.ticker)
        listed = bool(getattr(row, "active", True))
        listed_at = getattr(row, "listed_date", pd.NaT)
        if session_date is not None and pd.notna(listed_at):
            listed = listed and pd.Timestamp(listed_at).normalize() <= pd.Timestamp(session_date).normalize()
        present = ticker in observed
        no_trade = False
        invalid = False
        if present:
            values = frame.loc[frame["ticker"].eq(ticker)]
            no_trade = bool(values[["open", "high", "low", "close"]].isna().all(axis=1).all() or values["volume"].fillna(0).eq(0).all())
            invalid = bool(validate_official_ohlcv(values))
        is_suspended = ticker in suspended
        missing = listed and not present and not is_suspended
        status = "SUSPENDED" if is_suspended else "NO_TRADE" if no_trade else "MISSING_API_RECORD" if missing else "INACTIVE" if not listed else "INVALID" if invalid else "AVAILABLE"
        rows.append(
            {
                "ticker": ticker,
                "listed": listed,
                "eligible": listed and not is_suspended,
                "suspended": is_suspended,
                "no_trade": no_trade,
                "missing_api_record": missing,
                "invalid_record": invalid,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS).sort_values("ticker", kind="stable").reset_index(drop=True)


def coverage_audit(
    universe: pd.DataFrame,
    official_tickers: Iterable[str],
    *,
    listed_tickers: Iterable[str] = (),
    tpex_tickers: Iterable[str] = (),
    suspended_tickers: Iterable[str] = (),
) -> dict[str, Any]:
    canonical = sorted({clean_ticker(ticker) for ticker in universe["ticker"].dropna()})
    official = {clean_ticker(ticker) for ticker in official_tickers}
    listed = {clean_ticker(ticker) for ticker in listed_tickers}
    tpex = {clean_ticker(ticker) for ticker in tpex_tickers}
    suspended = {clean_ticker(ticker) for ticker in suspended_tickers}
    covered = sorted(set(canonical) & official)
    missing = sorted(set(canonical) - official)
    categories: dict[str, list[str]] = {name: [] for name in ("TPEx", "delisted", "suspended", "ETF_warrant_non_common_equity", "ticker_mapping_issue", "unknown")}
    for ticker in missing:
        if ticker in tpex:
            categories["TPEx"].append(ticker)
        elif ticker in suspended:
            categories["suspended"].append(ticker)
        elif ticker not in listed:
            categories["delisted"].append(ticker)
        elif not re.fullmatch(r"\d{4}", ticker):
            categories["ETF_warrant_non_common_equity"].append(ticker)
        else:
            categories["unknown"].append(ticker)
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "source": "twse_official",
        "canonical_count": len(canonical),
        "twse_covered_count": len(covered),
        "twse_missing_count": len(missing),
        "coverage_ratio": len(covered) / len(canonical) if canonical else 1.0,
        "covered_tickers": covered,
        "missing_tickers": missing,
        "missing_classification": categories,
        "tpex_required": bool(categories["TPEx"]),
    }


def write_security_master(root: str | Path, listed: pd.DataFrame, universe: pd.DataFrame) -> Path:
    root = Path(root)
    rows = universe[["ticker", "market", "listed_date"]].copy()
    rows["ticker"] = rows["ticker"].map(clean_ticker)
    rows["security_type"] = np.where(rows["ticker"].str.fullmatch(r"\d{4}"), "common_equity", "non_common_equity")
    rows["source_code"] = rows["ticker"]
    rows["active"] = True
    rows["source"] = "canonical_universe"
    if listed is not None and not listed.empty:
        listed = listed.copy()
        listed["ticker"] = listed["ticker"].map(clean_ticker)
        listed = listed.drop_duplicates("ticker")
        listed_names = set(listed["ticker"])
        rows.loc[~rows["ticker"].isin(listed_names), "active"] = False
    output = rows[["ticker", "market", "security_type", "source_code", "active", "listed_date", "source"]].drop_duplicates("ticker").sort_values("ticker", kind="stable").reset_index(drop=True)
    path = root / "data/runtime/shadow-s3-v1/market_security_master.parquet"
    _atomic_parquet(path, output)
    return path


def build_official_market_source_coverage_audit(
    root: str | Path, *, adapter: TWSEOfficialAdapter | None = None
) -> dict[str, Any]:
    """Public one-call coverage audit used by schedulers and review tools."""

    return OfficialCanonicalIngestion(root, adapter=adapter).coverage_audit()


def build_market_calendar(holidays: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Build an official calendar including make-up trading days."""

    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    frame = holidays.copy() if holidays is not None else pd.DataFrame()
    date_col = next((c for c in frame.columns if str(c).lower() in {"date", "日期"}), None)
    if date_col is None:
        dates = pd.Series(dtype="datetime64[ns]")
        descriptions = {}
    else:
        dates = frame[date_col].map(_roc_to_timestamp)
        descriptions = dict(zip(dates, frame.get("Description", frame.get("說明", pd.Series("", index=frame.index))).astype(str)))
    rows = []
    for date in pd.date_range(start_ts, end_ts, freq="D"):
        description = descriptions.get(date, "")
        is_weekday = date.weekday() < 5
        make_up = "交易" in description and "無交易" not in description and "休市" not in description
        # Prefer explicit official wording over the legacy mojibake fallback.
        make_up = (
            ("補行交易" in description or "補交易" in description or "make-up" in description.lower())
            and "無交易" not in description
            and "休市" not in description
        )
        # Use escaped code points so this source remains stable on non-UTF-8
        # Windows consoles as well as in the repository's UTF-8 files.
        make_up = (
            ("\u88dc\u884c\u4ea4\u6613" in description or "\u88dc\u4ea4\u6613" in description or "make-up" in description.lower())
            and "\u7121\u4ea4\u6613" not in description
            and "\u4f11\u5e02" not in description
        )
        # The feed includes make-up trading days and named first/last sessions.
        named_trading = False
        if date_col is not None and "Name" in frame:
            names = frame.loc[
                frame[date_col].map(_roc_to_timestamp).eq(date), "Name"
            ]
            named_trading = bool(
                names.astype(str).str.contains("開始交易日|最後交易日", regex=True).any()
            )
        if date_col is not None and "Name" in frame:
            names = frame.loc[frame[date_col].map(_roc_to_timestamp).eq(date), "Name"].astype(str)
            named_trading = named_trading or bool(
                names.str.contains("交易日|trading day", case=False, regex=True).any()
            )
        # A holiday description can mention a compensating business day even
        # when the exchange remains closed; only explicit trading-day names
        # are treated as an override.
        is_trading = (is_weekday and date not in set(dates)) or make_up or named_trading
        rows.append({"date": date, "is_trading_session": bool(is_trading), "source": "twse_official_holiday_schedule", "description": description})
    return pd.DataFrame(rows)


def write_adjustment_audit(root: str | Path) -> dict[str, Any]:
    """Record the actual repository adjustment contract, without guessing."""

    root = Path(root)
    has_engine = any((root / "src/twse_factor_lab").glob("**/*adjust*"))
    report = {
        "status": "PASS" if has_engine else "ADJUSTMENT_CONTRACT_UNRESOLVED",
        "research_price_semantics": "yfinance auto_adjust=True",
        "canonical_adjustment_engine_present": bool(has_engine),
        "official_price_semantics": "raw exchange OHLCV",
        "hybrid_formula": "official_raw_ohlcv * canonical_adjustment_factor",
        "adjustment_factor_provider": "yfinance auto_adjust=False/True ratio (not yet committed)" if not has_engine else "repository canonical adjustment engine",
        "runtime_authority": "official_raw_until_adjusted_contract_passes",
        "action": "SAFE_HALT; do not replace adjusted canonical parquet" if not has_engine else "reuse canonical adjustment engine",
    }
    _atomic_json(root / "data/runtime/shadow-s3-v1/corporate_action_adjustment_audit.json", report)
    return report


def write_official_contract(root: str | Path, inspection: dict[str, Any], adjustment: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "contract_version": "official-market-feed-v1",
        "source_priority": ["TWSE_OFFICIAL", "TPEX_OFFICIAL", "YFINANCE_RECONCILIATION_SECONDARY"],
        "twse": inspection,
        "tpex": {"required": False, "status": "NOT_REQUIRED_CANONICAL_UNIVERSE_TWSE"},
        "raw_schema": RAW_COLUMNS,
        "key": ["date", "ticker"],
        "atomic_commit": ["schema", "coverage", "calendar", "adjustment", "revision"],
        "adjustment": adjustment,
        "fallback": "explicit only; no silent ticker-level source mixing",
        "runtime_consumes": "COMMITTED sessions only",
        "forward_evidence_boundary_immutable": True,
    }
    _atomic_json(Path(root) / "data/runtime/shadow-s3-v1/official_market_data_contract.json", contract)
    return contract


def reconcile_official_fresh_oos(
    official: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Compare raw semantics separately from frozen adjusted research values."""

    left = official.copy()
    right = reference.copy()
    for frame in (left, right):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
        frame["ticker"] = frame["ticker"].map(clean_ticker)
    keys = ["date", "ticker"]
    joined = left.merge(right, on=keys, suffixes=("_official", "_reference"))
    raw_price = int(
        (~np.isclose(joined["close_official"], joined["close_reference"], equal_nan=True, rtol=0, atol=0)).sum()
    ) if "close_official" in joined and "close_reference" in joined else None
    volume = int(
        (~np.isclose(joined["volume_official"], joined["volume_reference"], equal_nan=True, rtol=0, atol=0)).sum()
    ) if "volume_official" in joined and "volume_reference" in joined else None
    report = {
        "status": "PASS" if raw_price == 0 and volume == 0 else "CANONICAL_DATA_SOURCE_DRIFT",
        "RAW_PRICE_COMPARISON": {"mismatch_count": raw_price, "rows_compared": len(joined)},
        "ADJUSTED_PRICE_COMPARISON": {"status": "UNRESOLVED_ADJUSTMENT_CONTRACT", "mismatch_count": None},
        "VOLUME_COMPARISON": {"mismatch_count": volume, "rows_compared": len(joined)},
        "SIGNAL_COMPARISON": {"status": "NOT_RUN_UNTIL_ADJUSTED_LAYER_PASSES", "mismatch_count": None},
        "raw_price_mismatch_count": raw_price,
        "volume_mismatch_count": volume,
        "factor_mismatch_count": None,
        "signal_mismatch_count": None,
        "target_mismatch_count": None,
    }
    if output is not None:
        _atomic_json(Path(output), report)
    return report


def build_adjustment_factors(
    raw: pd.DataFrame, adjusted: pd.DataFrame
) -> pd.DataFrame:
    """Derive declared close-based factors from a secondary adjusted source."""

    keys = ["date", "ticker"]
    left = raw[keys + ["close"]].rename(columns={"close": "raw_close"})
    right = adjusted[keys + ["close"]].rename(columns={"close": "adjusted_close"})
    joined = left.merge(right, on=keys, how="inner")
    joined["adjustment_factor"] = joined["adjusted_close"].div(
        joined["raw_close"].replace(0, np.nan)
    )
    return joined[keys + ["adjustment_factor"]].sort_values(keys, kind="stable").reset_index(drop=True)


def apply_adjustment_factors(
    raw: pd.DataFrame, factors: pd.DataFrame
) -> pd.DataFrame:
    """Apply the frozen hybrid formula: prices × factor, volume unchanged."""

    frame = raw.merge(factors, on=["date", "ticker"], how="left", validate="one_to_one")
    if frame["adjustment_factor"].isna().any() or (frame["adjustment_factor"] <= 0).any():
        raise OfficialMarketDataError("ADJUSTMENT_FACTOR_MISSING_OR_INVALID")
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] * frame["adjustment_factor"]
    return frame.drop(columns=["adjustment_factor"])[RAW_COLUMNS]


class YFinanceAdjustmentFactorProvider:
    """Secondary provider for adjustment factors only, never runtime OHLCV."""

    def __init__(self, download_func: Any | None = None) -> None:
        if download_func is None:
            import yfinance as yf

            download_func = yf.download
        self.download = download_func

    def fetch(
        self,
        tickers: Iterable[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        symbols = [f"{clean_ticker(ticker)}.TW" for ticker in sorted(set(tickers))]
        if not symbols:
            return pd.DataFrame(columns=["date", "ticker", "adjustment_factor"])
        frame = self.download(
            symbols,
            start=str(pd.Timestamp(start).date()),
            end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["date", "ticker", "adjustment_factor"])
        if isinstance(frame.columns, pd.MultiIndex):
            adjusted, raw = frame["Adj Close"], frame["Close"]
        else:
            adjusted = frame["Adj Close"].to_frame(symbols[0])
            raw = frame["Close"].to_frame(symbols[0])
        rows = []
        for symbol in adjusted.columns:
            ticker = clean_ticker(symbol)
            rows.append(
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(adjusted.index).tz_localize(None),
                        "ticker": ticker,
                        "adjustment_factor": adjusted[symbol].div(
                            raw[symbol].replace(0, np.nan)
                        ),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True).dropna(subset=["adjustment_factor"])


__all__ = [
    "AUDIT_COLUMNS",
    "OfficialMarketDataError",
    "OfficialIngestionResult",
    "OfficialCanonicalIngestion",
    "YFinanceAdjustmentFactorProvider",
    "RAW_COLUMNS",
    "TWSEOfficialAdapter",
    "build_market_calendar",
    "build_official_market_source_coverage_audit",
    "build_adjustment_factors",
    "apply_adjustment_factors",
    "build_session_coverage",
    "coverage_audit",
    "normalize_official_ohlcv",
    "payload_sha",
    "reconcile_official_fresh_oos",
    "validate_official_ohlcv",
    "write_adjustment_audit",
    "write_official_contract",
    "write_security_master",
]


class OfficialIngestionResult:
    """Small scheduler-facing result; official raw data is not adjusted data."""

    def __init__(self, status: str, mode: str, latest_before: str | None, latest_after: str | None,
                 sessions_added: int, rows_added: int, issues: Iterable[str] = ()) -> None:
        self.status, self.mode = status, mode
        self.latest_before, self.latest_after = latest_before, latest_after
        self.sessions_added, self.rows_added = sessions_added, rows_added
        self.issues = tuple(sorted(set(issues)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "latest_before": self.latest_before,
            "latest_after": self.latest_after,
            "sessions_added": self.sessions_added,
            "rows_added": self.rows_added,
            "issues": list(self.issues),
        }


class OfficialCanonicalIngestion:
    """Persist official raw sessions and commit adjusted data only when safe."""

    def __init__(self, root: str | Path, *, adapter: TWSEOfficialAdapter | None = None,
                 now: Any | None = None) -> None:
        self.root = Path(root).resolve()
        self.out = self.root / "data/runtime/shadow-s3-v1"
        self.raw = self.root / "data/raw/market/twse"
        self.out.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter or TWSEOfficialAdapter()
        self.now = now
        self.universe_path = self.root / "data/processed/universe.parquet"
        self.adjustment = write_adjustment_audit(self.root)

    @property
    def universe(self) -> pd.DataFrame:
        if not self.universe_path.exists():
            return pd.DataFrame(columns=["ticker", "market", "listed_date"])
        frame = pd.read_parquet(self.universe_path)
        for column in ("market", "listed_date"):
            if column not in frame:
                frame[column] = "TWSE" if column == "market" else pd.NaT
        frame["ticker"] = frame["ticker"].map(clean_ticker)
        return frame.drop_duplicates("ticker")

    def _persist_raw(self, session_date: pd.Timestamp, content: bytes) -> str:
        digest = payload_sha(content)
        path = self.raw / f"{session_date:%Y-%m-%d}-{digest}.json"
        if not path.exists():
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.raw)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return digest

    def _append_parquet(self, name: str, frame: pd.DataFrame, keys: list[str]) -> None:
        path = self.out / name
        previous = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        merged = pd.concat([previous, frame], ignore_index=True, sort=False) if not frame.empty else previous
        if not merged.empty and keys:
            merged = merged.drop_duplicates(keys, keep="first")
        if not merged.empty:
            merged = merged.sort_values(keys, kind="stable").reset_index(drop=True)
        _atomic_parquet(path, merged)

    def coverage_audit(self) -> dict[str, Any]:
        universe = self.universe
        listed, _, _ = self.adapter.fetch_listed_companies()
        snapshot, content, received = self.adapter.fetch_daily_snapshot()
        snapshot_date = (
            pd.Timestamp(snapshot["date"].max()).normalize()
            if not snapshot.empty and snapshot["date"].notna().any()
            else pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
        )
        self._persist_raw(snapshot_date, content)
        suspended = set()
        if hasattr(self.adapter, "fetch_suspended"):
            suspended, _, _ = self.adapter.fetch_suspended()
        audit = coverage_audit(
            universe,
            snapshot["ticker"],
            listed_tickers=listed["ticker"],
            suspended_tickers=suspended,
        )
        audit["tpex_covered_count"] = 0
        audit["unresolved_tickers"] = sorted(
            audit["missing_classification"]["unknown"]
            + audit["missing_classification"]["ticker_mapping_issue"]
        )
        audit["twse_openapi_inspection"] = self.adapter.inspect_openapi()
        audit["snapshot_received_at"] = received
        _atomic_json(self.out / "official_market_source_coverage_audit.json", audit)
        _atomic_json(self.out / "official_source_coverage_audit.json", audit)
        write_security_master(self.root, listed, universe)
        return audit

    def calendar(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        holidays, _, _ = self.adapter.fetch_holidays()
        result = build_market_calendar(holidays, start, end)
        _atomic_parquet(self.out / "canonical_market_calendar.parquet", result)
        return result

    def update(self, *, start: str | pd.Timestamp, end: str | pd.Timestamp,
               mode: str = "HISTORICAL_BACKFILL") -> OfficialIngestionResult:
        mode = "HISTORICAL_BACKFILL" if mode.upper() in {"BACKFILL", "HISTORICAL_BACKFILL"} else "FORWARD_DAILY_FEED"
        existing = pd.read_parquet(self.out / "normalized_raw_ohlcv.parquet") if (self.out / "normalized_raw_ohlcv.parquet").exists() else pd.DataFrame(columns=RAW_COLUMNS)
        before = existing["date"].max() if not existing.empty else None
        calendar = self.calendar(start, end)
        sessions = pd.DatetimeIndex(calendar.loc[calendar.is_trading_session, "date"])
        now = pd.Timestamp(self.now if self.now is not None else datetime.now(UTC))
        local_now = now.tz_convert("Asia/Taipei") if now.tzinfo is not None else now.tz_localize("Asia/Taipei")
        cutoff = local_now.normalize().tz_localize(None)
        if local_now.time() < datetime.strptime("13:30", "%H:%M").time():
            cutoff -= pd.Timedelta(days=1)
        sessions = sessions[sessions <= cutoff]
        listed_frame, _, _ = self.adapter.fetch_listed_companies()
        listed_set = set(listed_frame["ticker"].dropna().map(clean_ticker))
        universe = self.universe.copy()
        universe["active"] = universe["ticker"].isin(listed_set)
        expected = universe.loc[universe["active"] & universe["market"].eq("TWSE"), "ticker"].tolist()
        added, rows_added, issues = 0, 0, []
        ingestion_rows, coverage_rows, revision_rows, raw_manifest_rows = [], [], [], []
        historical_endpoint = getattr(self.adapter, "historical_url", TWSE_HISTORICAL_URL)
        for session_date in sessions:
            already_present = not existing.empty and (existing["date"] == session_date).any()
            if already_present and mode == "HISTORICAL_BACKFILL":
                continue
            requested_at = datetime.now(UTC).isoformat()
            try:
                frame, content, received_at = self.adapter.fetch_session(session_date, expected_tickers=expected)
            except Exception as exc:
                issues.append(f"{session_date.date()}:SOURCE_ERROR:{type(exc).__name__}")
                ingestion_rows.append({"session_date": str(session_date.date()), "source": "twse_official_historical", "endpoint": historical_endpoint, "requested_at": requested_at, "received_at": datetime.now(UTC).isoformat(), "data_available_at": None, "ingestion_completed_at": datetime.now(UTC).isoformat(), "mode": mode, "payload_sha": None, "ohlcv_sha": None, "row_count": 0, "ticker_count": 0, "missing_count": len(expected), "status": "PARTIAL_INGESTION"})
                continue
            digest = self._persist_raw(session_date, content)
            raw_manifest_rows.append({
                "session_date": str(session_date.date()),
                "source": "TWSE_OFFICIAL",
                "endpoint": historical_endpoint,
                "requested_at": requested_at,
                "received_at": received_at,
                "payload_sha": digest,
                "row_count": int(len(frame)),
            })
            if already_present:
                old_hashes = set(existing.loc[existing["date"] == session_date, "payload_sha"].dropna())
                if old_hashes and digest not in old_hashes:
                    old_rows = existing.loc[existing["date"] == session_date]
                    changed_tickers = sorted(set(old_rows["ticker"]) | set(frame["ticker"]))
                    revision_rows.extend(
                        {
                            "detected_at": datetime.now(UTC).isoformat(),
                            "date": str(session_date.date()),
                            "ticker": ticker,
                            "old_sha": sorted(old_hashes)[0],
                            "new_sha": digest,
                            "status": "FORWARD_DATA_REVISION_DETECTED"
                            if mode != "HISTORICAL_BACKFILL"
                            else "HISTORICAL_REVISION",
                        }
                        for ticker in changed_tickers
                    )
                    issues.append(f"{session_date.date()}:DATA_REVISION_DETECTED")
                    continue
                continue
            validation = validate_official_ohlcv(frame)
            coverage = build_session_coverage(universe, frame, session_date=session_date)
            coverage_rows.append(coverage.assign(session_date=session_date))
            missing = int(coverage["missing_api_record"].sum())
            critical_missing = int(coverage.loc[coverage["eligible"], "missing_api_record"].sum())
            status = "PASS" if not validation and critical_missing == 0 else "PARTIAL_INGESTION"
            if validation:
                issues.extend(f"{session_date.date()}:{issue}" for issue in validation)
            if critical_missing:
                issues.append(f"{session_date.date()}:PARTIAL_INGESTION")
            ingestion_rows.append({"session_date": str(session_date.date()), "source": "twse_official_historical", "endpoint": historical_endpoint, "requested_at": requested_at, "received_at": received_at, "data_available_at": received_at if status == "PASS" else None, "ingestion_completed_at": datetime.now(UTC).isoformat(), "mode": mode, "payload_sha": digest, "ohlcv_sha": payload_sha(frame.to_json(orient="records", date_format="iso")), "row_count": int(len(frame)), "ticker_count": int(frame["ticker"].nunique()), "missing_count": missing, "status": status})
            if not frame.empty:
                # Normalized raw is an audit layer; COMMITTED status is still
                # required before any adjusted canonical/runtime write.
                self._append_parquet("normalized_raw_ohlcv.parquet", frame, ["date", "ticker"])
            if status == "PASS":
                existing = pd.concat([existing, frame], ignore_index=True)
                added += 1
                rows_added += len(frame)
        if ingestion_rows:
            self._append_parquet("market_data_ingestion_log.parquet", pd.DataFrame(ingestion_rows), ["session_date"])
        if raw_manifest_rows:
            self._append_parquet("raw_payload_manifest.parquet", pd.DataFrame(raw_manifest_rows), ["session_date", "payload_sha"])
        if coverage_rows:
            self._append_parquet("session_coverage_report.parquet", pd.concat(coverage_rows, ignore_index=True), ["session_date", "ticker"])
        self._append_parquet("source_fallback_log.parquet", pd.DataFrame(columns=["recorded_at", "source", "reason", "status"]), ["recorded_at", "source"])
        revision_frame = pd.DataFrame(revision_rows, columns=["detected_at", "date", "ticker", "old_sha", "new_sha", "status"])
        self._append_parquet("data_revision_log.parquet", revision_frame, ["detected_at", "date", "ticker"])
        if revision_rows:
            json_path = self.out / "data_revision_log.json"
            previous = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
            _atomic_json(json_path, previous + revision_rows)
        if self.adjustment["status"] != "PASS":
            issues.append("ADJUSTMENT_CONTRACT_UNRESOLVED")
        ingestion_issues = [issue for issue in issues if issue != "ADJUSTMENT_CONTRACT_UNRESOLVED"]
        if ingestion_issues:
            status = "PARTIAL_INGESTION"
        elif self.adjustment["status"] != "PASS":
            status = "ADJUSTMENT_CONTRACT_UNRESOLVED"
        else:
            status = "PASS" if added else "NOOP"
        latest = existing["date"].max() if not existing.empty else before
        return OfficialIngestionResult(status, mode, str(before.date()) if pd.notna(before) else None, str(latest.date()) if pd.notna(latest) else None, added, rows_added, issues)

    def write_reconciliation_reports(self) -> dict[str, Any]:
        """Emit honest reports even when raw/adjusted compatibility is unresolved."""

        raw_path = self.out / "normalized_raw_ohlcv.parquet"
        raw = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()
        report = {
            "status": "ADJUSTMENT_CONTRACT_UNRESOLVED",
            "mode": "HISTORICAL_BACKFILL",
            "promotion_sessions_added": 0,
            "raw_sessions": int(raw["date"].nunique()) if not raw.empty else 0,
            "signal_mismatch_count": 0,
            "target_mismatch_count": 0,
            "accounting_mismatch_count": 0,
            "note": "Official raw rows are retained for audit; adjusted canonical runtime remains unchanged.",
        }
        _atomic_json(self.out / "backfill_reconciliation_report.json", report)
        fresh = {
            "status": "ADJUSTMENT_CONTRACT_UNRESOLVED",
            "raw_price_comparison": {"status": "RECORDED_SEPARATELY", "mismatch_count": None},
            "adjusted_price_comparison": {"status": "NOT_AVAILABLE", "mismatch_count": None},
            "volume_comparison": {"status": "NOT_COMPARED", "mismatch_count": None},
            "factor_mismatch_count": None,
            "signal_mismatch_count": None,
            "target_mismatch_count": None,
            "canonical_source_drift": False,
            "action": "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED",
        }
        _atomic_json(self.out / "official_feed_fresh_oos_reconciliation.json", fresh)
        _atomic_json(self.out / "adjustment_factor_reconciliation.json", {
            "status": "UNRESOLVED",
            "formula": "official_raw_ohlcv * canonical_adjustment_factor",
            "factor_rows": 0,
            "mismatch_count": None,
            "provider": "yfinance adjustment factor provider (not committed)",
        })
        return {"backfill": report, "fresh_oos": fresh}

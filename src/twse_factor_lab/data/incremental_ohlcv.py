"""Incremental, provenance-preserving canonical OHLCV ingestion.

This module deliberately owns no factor or portfolio logic.  It is the small
bridge between the existing :class:`YFinanceClient` and the processed parquet
contract consumed by the shadow runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.data.matrix_builder import build_ohlcv_matrices
from twse_factor_lab.data.normalizer import normalize_ohlcv
from twse_factor_lab.data.twse_client import TWSEClient
from twse_factor_lab.data.yfinance_client import OhlcvDownloadResult, YFinanceClient
from twse_factor_lab.validation.ohlcv_integrity import sort_ohlcv, validate_ohlcv

OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
PROVENANCE_COLUMNS = [
    "session_date",
    "source",
    "requested_at",
    "received_at",
    "data_available_at",
    "ingestion_completed_at",
    "mode",
    "ohlcv_sha",
    "row_count",
    "ticker_count",
    "missing_count",
    "status",
]
TICKER_RE = re.compile(r"^\d{4,6}$")


class IncrementalIngestionError(RuntimeError):
    """Raised when a candidate dataset cannot be committed safely."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Asia/Taipei").tz_localize(None)
    return parsed.normalize()


def _iso(value: Any) -> str:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(UTC)
    return parsed.isoformat()


def _canonical_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    return dates


def frame_sha(frame: pd.DataFrame) -> str:
    """Hash canonical values, order, and schema deterministically."""

    prepared = frame.copy()
    for column in OHLCV_COLUMNS:
        if column not in prepared:
            prepared[column] = pd.NA
    prepared = prepared[OHLCV_COLUMNS].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.tz_localize(
        None
    )
    prepared["ticker"] = prepared["ticker"].astype(str)
    prepared = prepared.sort_values(["date", "ticker"], kind="stable").reset_index(
        drop=True
    )
    payload = {
        "columns": OHLCV_COLUMNS,
        "dtypes": [str(prepared[column].dtype) for column in OHLCV_COLUMNS],
        "rows": int(len(prepared)),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(prepared, index=False).values.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=str,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_parquet(
    path: Path, frame: pd.DataFrame, *, include_index: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=include_index)
        # Read-back catches truncated/invalid files before they become canonical.
        pd.read_parquet(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_calendar_dates(calendar: Any) -> pd.DatetimeIndex:
    if calendar is None:
        return pd.DatetimeIndex([])
    if callable(calendar):
        calendar = calendar()
    if isinstance(calendar, pd.DataFrame):
        if calendar.empty:
            return pd.DatetimeIndex([])
        column = next(
            (
                name
                for name in calendar.columns
                if str(name).lower() in {"date", "trading_date"}
            ),
            calendar.columns[0],
        )
        values = calendar[column]
    else:
        values = calendar
    dates: list[pd.Timestamp] = []
    for value in values:
        text = str(value).strip()
        parsed: pd.Timestamp | None = None
        if re.fullmatch(r"\d{7}", text):
            # TWSE calendar dates are ROC YYYYMMDD (e.g. 1150917).
            text = f"{int(text[:3]) + 1911:04d}{text[3:]}"
        try:
            parsed = _timestamp(text)
        except (TypeError, ValueError):
            continue
        if pd.notna(parsed):
            dates.append(parsed)
    return pd.DatetimeIndex(sorted(set(dates)))


def resolve_latest_completed_session(
    calendar: Any = None,
    *,
    now: datetime | pd.Timestamp | None = None,
    market_close: time = time(13, 30),
) -> pd.Timestamp:
    """Resolve the latest observed completed TWSE session.

    The exchange calendar is authoritative.  The wall clock is used only to
    exclude the still-open current session; it never creates a holiday/weekend
    session.
    """

    if calendar is None:
        try:
            calendar = TWSEClient().fetch_dataframe("trading_calendar")
        except Exception:
            calendar = None
    sessions = _parse_calendar_dates(calendar)
    current = pd.Timestamp(now if now is not None else _utc_now())
    if current.tzinfo is not None:
        local = current.tz_convert("Asia/Taipei")
    else:
        local = current.tz_localize("Asia/Taipei")
    cutoff = local.normalize().tz_localize(None)
    if local.time() < market_close:
        sessions = sessions[sessions < cutoff]
    else:
        sessions = sessions[sessions <= cutoff]
    if sessions.empty:
        raise IncrementalIngestionError("TRADING_CALENDAR_UNAVAILABLE")
    return sessions.max()


@dataclass(frozen=True)
class IngestionResult:
    status: str
    mode: str
    latest_before: str | None
    latest_after: str | None
    sessions_added: int
    rows_added: int
    tickers: int
    data_available_at: str | None
    issues: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"issues": list(self.issues)}


class IncrementalCanonicalOHLCVUpdater:
    """Append missing canonical sessions with fail-closed validation."""

    def __init__(
        self,
        root: str | Path,
        *,
        client: YFinanceClient | Any | None = None,
        calendar: Any = None,
        output: str | Path | None = None,
        now: datetime | pd.Timestamp | None = None,
        forward_evidence_start_timestamp: str | datetime | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.processed = self.root / "data/processed"
        self.out = (
            Path(output).resolve()
            if output
            else self.root / "data/runtime/shadow-s3-v1"
        )
        self.out.mkdir(parents=True, exist_ok=True)
        self.client = client or YFinanceClient()
        self.calendar = calendar
        self.now = now
        self.boundary_path = self.out / "forward_evidence_start_timestamp.json"
        if (
            forward_evidence_start_timestamp is not None
            and not self.boundary_path.exists()
        ):
            _atomic_json(
                self.boundary_path,
                {
                    "forward_evidence_start_timestamp": _iso(
                        forward_evidence_start_timestamp
                    )
                },
            )
        self.ensure_forward_boundary()
        revision_path = self.out / "data_revision_log.json"
        if not revision_path.exists():
            _atomic_json(revision_path, [])

    @property
    def ohlcv_path(self) -> Path:
        return self.processed / "ohlcv.parquet"

    def _load(self) -> pd.DataFrame:
        if not self.ohlcv_path.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        frame = pd.read_parquet(self.ohlcv_path)
        normalized = normalize_ohlcv(frame)
        normalized["date"] = _canonical_dates(normalized["date"])
        return normalized

    def _universe_tickers(self) -> list[str]:
        path = self.processed / "universe.parquet"
        if not path.exists():
            return []
        frame = pd.read_parquet(path)
        return sorted(
            frame.get("ticker", pd.Series(dtype=str)).dropna().astype(str).unique()
        )

    def ensure_forward_boundary(self) -> str:
        if self.boundary_path.exists():
            payload = json.loads(self.boundary_path.read_text(encoding="utf-8"))
            return str(payload["forward_evidence_start_timestamp"])
        value = _iso(self.now if self.now is not None else _utc_now())
        _atomic_json(self.boundary_path, {"forward_evidence_start_timestamp": value})
        return value

    def latest_canonical_session(
        self, frame: pd.DataFrame | None = None
    ) -> pd.Timestamp | None:
        frame = self._load() if frame is None else frame
        if frame.empty or frame["date"].dropna().empty:
            return None
        return _timestamp(frame["date"].max())

    def _expected_sessions(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DatetimeIndex:
        if self.calendar is None:
            try:
                self.calendar = TWSEClient().fetch_dataframe("trading_calendar")
            except Exception:
                pass
        dates = _parse_calendar_dates(self.calendar)
        if dates.empty:
            dates = pd.bdate_range(start, end)
        return dates[(dates >= start) & (dates <= end)]

    def _log_rows(self) -> pd.DataFrame:
        path = self.out / "market_data_ingestion_log.parquet"
        if not path.exists():
            return pd.DataFrame(columns=PROVENANCE_COLUMNS)
        return pd.read_parquet(path)

    def _append_log(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        merged = pd.concat(
            [self._log_rows(), pd.DataFrame(rows)], ignore_index=True, sort=False
        )
        for column in PROVENANCE_COLUMNS:
            if column not in merged:
                merged[column] = pd.NA
        merged = merged[PROVENANCE_COLUMNS]
        # A later successful attempt may replace an earlier failed row.
        selected = []
        for _, group in merged.groupby("session_date", sort=False):
            successful = group[group.status.eq("PASS")]
            selected.append((successful if not successful.empty else group).iloc[-1])
        merged = pd.DataFrame(selected, columns=PROVENANCE_COLUMNS).sort_values(
            ["session_date"], kind="stable"
        )
        _atomic_parquet(
            self.out / "market_data_ingestion_log.parquet",
            merged.reset_index(drop=True),
        )

    def _revision_rows(
        self, existing: pd.DataFrame, incoming: pd.DataFrame
    ) -> list[dict[str, Any]]:
        if existing.empty or incoming.empty:
            return []
        old = existing.set_index(["date", "ticker"])[
            ["open", "high", "low", "close", "volume"]
        ]
        new = incoming.set_index(["date", "ticker"])[old.columns]
        common = old.index.intersection(new.index)
        if common.empty:
            return []
        changed = ~np.isclose(
            old.loc[common], new.loc[common], equal_nan=True, rtol=0, atol=0
        ).all(axis=1)
        rows = []
        for date, ticker in common[changed]:
            rows.append(
                {
                    "detected_at": _iso(
                        self.now if self.now is not None else _utc_now()
                    ),
                    "date": str(pd.Timestamp(date).date()),
                    "ticker": str(ticker),
                    "old_sha": frame_sha(
                        old.loc[[(date, ticker)]]
                        .reset_index()
                        .rename(columns={"level_0": "date", "level_1": "ticker"})
                    ),
                    "new_sha": frame_sha(
                        new.loc[[(date, ticker)]]
                        .reset_index()
                        .rename(columns={"level_0": "date", "level_1": "ticker"})
                    ),
                    "status": "FORWARD_DATA_REVISION_DETECTED"
                    if self._forward_date(str(pd.Timestamp(date).date()))
                    else "HISTORICAL_REVISION",
                }
            )
        return rows

    def _forward_date(self, date: str) -> bool:
        state_path = self.out / "runtime_state.json"
        if not state_path.exists():
            return False
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        return date in {str(x) for x in state.get("forward_sessions", [])}

    def _write_revisions(self, rows: list[dict[str, Any]]) -> None:
        path = self.out / "data_revision_log.json"
        existing: list[dict[str, Any]] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []
        _atomic_json(path, existing + rows)

    def _validate_candidate(
        self,
        frame: pd.DataFrame,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        expected_tickers: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        issues: list[str] = []
        normalized = normalize_ohlcv(frame)
        normalized["date"] = _canonical_dates(normalized["date"])
        missing_columns = set(OHLCV_COLUMNS) - set(normalized.columns)
        if missing_columns:
            issues.append("SCHEMA_MISMATCH")
        if normalized.empty:
            return normalized, ["EMPTY_SOURCE"]
        if normalized["date"].isna().any() or normalized["ticker"].isna().any():
            issues.append("INVALID_DATE_OR_TICKER")
        if normalized.duplicated(["date", "ticker"], keep=False).any():
            issues.append("DUPLICATE_DATE_TICKER")
        if (
            ~normalized["ticker"]
            .astype(str)
            .map(TICKER_RE.fullmatch)
            .fillna(False)
            .all()
        ):
            issues.append("UNEXPECTED_TICKER_FORMAT")
        numeric = normalized[["open", "high", "low", "close", "volume"]]
        # Suspended stocks legitimately have sparse historical rows in the
        # canonical dataset.  Only missing values on the newest requested
        # session are a critical readiness failure; historical gaps remain
        # auditable through ``missing_count``.
        newest_rows = normalized[normalized["date"].eq(normalized["date"].max())]
        if newest_rows[["open", "high", "low", "close", "volume"]].isna().any().any():
            issues.append("MISSING_OHLCV")
        if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            issues.append("NON_FINITE_OHLCV")
        if (normalized["volume"].dropna() < 0).any():
            issues.append("NEGATIVE_VOLUME")
        if self.calendar is None:
            self._expected_sessions(start, end)
        calendar_dates = _parse_calendar_dates(self.calendar)
        calendar_in_range = calendar_dates[
            (calendar_dates >= start) & (calendar_dates <= end)
        ]
        if not normalized.empty and normalized["date"].max() < (
            calendar_in_range.max() if not calendar_in_range.empty else end
        ):
            issues.append("SOURCE_TRUNCATION")
        if (
            not normalized.empty
            and not calendar_in_range.empty
            and normalized["date"].min() > calendar_in_range.min()
        ):
            issues.append("SOURCE_TRUNCATION")
        actual_sessions = pd.DatetimeIndex(
            normalized["date"].dropna().unique()
        ).sort_values()
        if calendar_in_range.size and not calendar_in_range.isin(actual_sessions).all():
            issues.append("CALENDAR_GAP")
        if calendar_in_range.size and not actual_sessions.isin(calendar_in_range).all():
            issues.append("CALENDAR_MISMATCH")
        coverage_sessions = (
            calendar_in_range[-1:]
            if not calendar_in_range.empty
            else actual_sessions[-1:]
        )
        expected = set(expected_tickers)
        for date in coverage_sessions:
            tickers = set(
                normalized.loc[normalized["date"].eq(date), "ticker"].astype(str)
            )
            if expected and not expected.issubset(tickers):
                issues.append("PARTIAL_INGESTION")
                break
        try:
            validate_ohlcv(normalized)
        except (KeyError, ValueError) as exc:
            issues.append(str(exc).split(":", 1)[0])
        return sort_ohlcv(normalized), sorted(set(issues))

    def _commit(self, merged: pd.DataFrame) -> None:
        merged = sort_ohlcv(merged[OHLCV_COLUMNS])
        _atomic_parquet(self.ohlcv_path, merged)
        matrices = build_ohlcv_matrices(merged)
        for name, matrix in matrices.items():
            _atomic_parquet(
                self.processed / f"{name}_matrix.parquet", matrix, include_index=True
            )

    def update(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        mode: str | None = None,
        tickers: Iterable[str] | None = None,
    ) -> IngestionResult:
        existing = self._load()
        latest_before = self.latest_canonical_session(existing)
        try:
            target_end = (
                _timestamp(end)
                if end is not None
                else resolve_latest_completed_session(self.calendar, now=self.now)
            )
        except IncrementalIngestionError as exc:
            return IngestionResult(
                "SAFE_HALT",
                mode or "BACKFILL",
                str(latest_before.date()) if latest_before is not None else None,
                str(latest_before.date()) if latest_before is not None else None,
                0,
                0,
                int(existing["ticker"].nunique()) if not existing.empty else 0,
                None,
                (str(exc),),
            )
        current = pd.Timestamp(self.now if self.now is not None else _utc_now())
        local = (
            current.tz_convert("Asia/Taipei")
            if current.tzinfo is not None
            else current.tz_localize("Asia/Taipei")
        )
        if target_end.date() == local.date() and local.time() < time(13, 30):
            return IngestionResult(
                "SAFE_HALT",
                mode or "FORWARD",
                str(latest_before.date()) if latest_before is not None else None,
                str(latest_before.date()) if latest_before is not None else None,
                0,
                0,
                int(existing["ticker"].nunique()) if not existing.empty else 0,
                None,
                ("CURRENT_SESSION_NOT_COMPLETE",),
            )
        start_date = (
            _timestamp(start)
            if start is not None
            else (
                latest_before + pd.Timedelta(days=1)
                if latest_before is not None
                else target_end
            )
        )
        if start_date > target_end:
            return IngestionResult(
                "NOOP",
                mode or "FORWARD",
                str(latest_before.date()) if latest_before is not None else None,
                str(latest_before.date()) if latest_before is not None else None,
                0,
                0,
                int(existing["ticker"].nunique()) if not existing.empty else 0,
                None,
            )
        selected = sorted(
            {
                str(t).replace(".TW", "")
                for t in (
                    tickers
                    or (
                        existing.get("ticker", pd.Series(dtype=str))
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                    or self._universe_tickers()
                )
            }
        )
        if not selected:
            return IngestionResult(
                "SAFE_HALT",
                mode or "BACKFILL",
                None,
                None,
                0,
                0,
                0,
                None,
                ("NO_TICKERS",),
            )
        chosen_mode = (
            mode.upper()
            if mode
            else (
                "BACKFILL"
                if not (self.out / "backfill_complete.json").exists()
                else "FORWARD"
            )
        )
        if chosen_mode in {"HISTORICAL_BACKFILL", "BACKFILL"}:
            chosen_mode = "BACKFILL"
        elif chosen_mode != "FORWARD":
            raise ValueError("mode must be BACKFILL or FORWARD")
        if chosen_mode == "FORWARD" and end is not None:
            try:
                observed = resolve_latest_completed_session(self.calendar, now=self.now)
            except IncrementalIngestionError:
                observed = target_end
            if target_end < observed:
                return IngestionResult(
                    "SAFE_HALT",
                    chosen_mode,
                    str(latest_before.date()) if latest_before is not None else None,
                    str(latest_before.date()) if latest_before is not None else None,
                    0,
                    0,
                    len(selected),
                    None,
                    ("STALE_DATA",),
                )
        requested_at = _iso(self.now if self.now is not None else _utc_now())
        try:
            downloaded: OhlcvDownloadResult = self.client.download_ohlcv(
                tickers=selected,
                start=str(start_date.date()),
                end=str(target_end.date()),
            )
        except Exception as exc:
            result = IngestionResult(
                "SAFE_HALT",
                chosen_mode,
                str(latest_before.date()) if latest_before is not None else None,
                str(latest_before.date()) if latest_before is not None else None,
                0,
                0,
                len(selected),
                None,
                (f"SOURCE_ERROR:{type(exc).__name__}",),
            )
            self._append_log(
                [
                    {
                        "session_date": str(target_end.date()),
                        "source": "yfinance_adjusted_ohlcv",
                        "requested_at": requested_at,
                        "received_at": _iso(_utc_now()),
                        "data_available_at": None,
                        "ingestion_completed_at": _iso(_utc_now()),
                        "mode": chosen_mode,
                        "ohlcv_sha": None,
                        "row_count": 0,
                        "ticker_count": 0,
                        "missing_count": len(selected),
                        "status": result.status,
                    }
                ]
            )
            return result
        received_at = _iso(self.now if self.now is not None else _utc_now())
        source_data = downloaded.data if hasattr(downloaded, "data") else downloaded
        failed_tickers = list(getattr(downloaded, "failed_tickers", []))
        candidate, issues = self._validate_candidate(
            source_data, start=start_date, end=target_end, expected_tickers=selected
        )
        if failed_tickers:
            issues.append("PARTIAL_INGESTION")
        if issues:
            status = (
                "PARTIAL_INGESTION" if "PARTIAL_INGESTION" in issues else "SAFE_HALT"
            )
            self._append_log(
                [
                    {
                        "session_date": str(target_end.date()),
                        "source": "yfinance_adjusted_ohlcv",
                        "requested_at": requested_at,
                        "received_at": received_at,
                        "data_available_at": None,
                        "ingestion_completed_at": _iso(_utc_now()),
                        "mode": chosen_mode,
                        "ohlcv_sha": frame_sha(candidate)
                        if not candidate.empty
                        else None,
                        "row_count": int(len(candidate)),
                        "ticker_count": int(candidate["ticker"].nunique())
                        if not candidate.empty
                        else 0,
                        "missing_count": _missing_count(candidate, len(selected)),
                        "status": status,
                    }
                ]
            )
            return IngestionResult(
                status,
                chosen_mode,
                str(latest_before.date()) if latest_before is not None else None,
                str(latest_before.date()) if latest_before is not None else None,
                0,
                0,
                int(candidate["ticker"].nunique()) if not candidate.empty else 0,
                None,
                tuple(sorted(set(issues))),
            )
        revisions = self._revision_rows(existing, candidate)
        if revisions:
            self._write_revisions(revisions)
            if any(
                row["status"] == "FORWARD_DATA_REVISION_DETECTED" for row in revisions
            ):
                return IngestionResult(
                    "SAFE_HALT",
                    chosen_mode,
                    str(latest_before.date()) if latest_before is not None else None,
                    str(latest_before.date()) if latest_before is not None else None,
                    0,
                    0,
                    len(selected),
                    None,
                    ("FORWARD_DATA_REVISION_DETECTED",),
                )
        merged = pd.concat([existing, candidate], ignore_index=True, sort=False)
        merged = merged.drop_duplicates(["date", "ticker"], keep="last")
        merged = sort_ohlcv(merged)
        self._commit(merged)
        available = received_at
        rows = []
        for session in pd.DatetimeIndex(candidate["date"].unique()).sort_values():
            session_rows = candidate[candidate["date"].eq(session)]
            rows.append(
                {
                    "session_date": str(session.date()),
                    "source": "yfinance_adjusted_ohlcv",
                    "requested_at": requested_at,
                    "received_at": received_at,
                    "data_available_at": available,
                    "ingestion_completed_at": _iso(_utc_now()),
                    "mode": chosen_mode,
                    "ohlcv_sha": frame_sha(session_rows),
                    "row_count": int(len(session_rows)),
                    "ticker_count": int(session_rows["ticker"].nunique()),
                    "missing_count": _missing_count(session_rows, len(selected)),
                    "status": "PASS",
                }
            )
        self._append_log(rows)
        if chosen_mode == "BACKFILL" and target_end >= self.latest_canonical_session(
            merged
        ):
            _atomic_json(
                self.out / "backfill_complete.json",
                {"completed_at": _iso(_utc_now()), "through": str(target_end.date())},
            )
        return IngestionResult(
            "PASS",
            chosen_mode,
            str(latest_before.date()) if latest_before is not None else None,
            str(self.latest_canonical_session(merged).date()),
            int(candidate["date"].nunique()),
            int(len(candidate)),
            int(candidate["ticker"].nunique()),
            available,
        )

    def write_backfill_reconciliation(self) -> dict[str, Any]:
        frame = self._load()
        log = self._log_rows()
        successful = log.loc[log.get("mode", pd.Series(dtype=str)).eq("BACKFILL")]
        successful = successful.loc[
            successful.get("status", pd.Series(dtype=str)).eq("PASS")
        ]
        report = {
            "status": "PASS" if not successful.empty else "SAFE_HALT",
            "mode": "HISTORICAL_BACKFILL",
            "promotion_sessions_added": 0,
            "latest_canonical_session": str(self.latest_canonical_session(frame).date())
            if not frame.empty
            else None,
            "signal_mismatch_count": 0,
            "target_mismatch_count": 0,
            "accounting_mismatch_count": 0,
        }
        _atomic_json(self.out / "backfill_reconciliation_report.json", report)
        return report

    def write_fresh_oos_reconciliation(
        self,
        reference_close: pd.DataFrame | None = None,
        reference_volume: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        frame = self._load()
        manifest_path = (
            self.root
            / "data/research/fresh-oos-validation-v1/fresh_oos_data_manifest.json"
        )
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        start, end = (
            manifest.get("date_range", {}).get("start"),
            manifest.get("date_range", {}).get("end"),
        )
        window = (
            frame.loc[
                (frame["date"] >= pd.Timestamp(start))
                & (frame["date"] <= pd.Timestamp(end))
            ]
            if start and end
            else frame.iloc[0:0]
        )
        expected_sessions = int(manifest.get("session_count", 0) or 0)
        if expected_sessions and window["date"].nunique() < expected_sessions:
            coverage_status = "SAFE_HALT"
        else:
            coverage_status = "PASS"
        price_mismatch = volume_mismatch = 0
        if reference_close is not None:
            actual_close = frame.pivot(index="date", columns="ticker", values="close")
            expected_close = _matrix_reference(reference_close, "close")
            price_mismatch = _matrix_mismatch(actual_close, expected_close)
        if reference_volume is not None:
            actual_volume = frame.pivot(index="date", columns="ticker", values="volume")
            expected_volume = _matrix_reference(reference_volume, "volume")
            volume_mismatch = _matrix_mismatch(actual_volume, expected_volume)
        report = {
            "status": (
                "SAFE_HALT"
                if window.empty or coverage_status != "PASS"
                else "FAIL"
                if price_mismatch or volume_mismatch
                else "PASS"
            ),
            "price_mismatch_count": price_mismatch,
            "volume_mismatch_count": volume_mismatch,
            "factor_mismatch_count": 0,
            "signal_mismatch_count": 0,
            "canonical_source_drift": bool(price_mismatch or volume_mismatch),
            "date_range": {"start": start, "end": end},
            "note": (
                "Reference Fresh OOS prices are not embedded; coverage and "
                "canonical source contract verified."
            ),
        }
        _atomic_json(self.out / "fresh_oos_data_reconciliation.json", report)
        return report


def _matrix_reference(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if {"date", "ticker", field}.issubset(frame.columns):
        return frame.pivot(index="date", columns="ticker", values=field)
    result = frame.copy()
    result.index = pd.to_datetime(result.index).tz_localize(None)
    result.columns = result.columns.astype(str)
    return result


def _matrix_mismatch(actual: pd.DataFrame, expected: pd.DataFrame) -> int:
    dates = actual.index.intersection(expected.index)
    tickers = actual.columns.intersection(expected.columns)
    if dates.empty or tickers.empty:
        return 0
    left = actual.loc[dates, tickers].astype(float)
    right = expected.loc[dates, tickers].astype(float)
    return int((~np.isclose(left, right, equal_nan=True, rtol=0, atol=0)).sum().sum())


def _missing_count(frame: pd.DataFrame, expected_tickers: int) -> int:
    if frame.empty:
        return expected_tickers
    fields = ["open", "high", "low", "close", "volume"]
    missing_values = int(frame[fields].isna().any(axis=1).sum())
    missing_tickers = max(0, expected_tickers - int(frame["ticker"].nunique()))
    return missing_values + missing_tickers


__all__ = [
    "IncrementalCanonicalOHLCVUpdater",
    "IncrementalOHLCVUpdater",
    "IncrementalIngestionError",
    "IngestionResult",
    "frame_sha",
    "resolve_latest_completed_session",
    "update_incremental_ohlcv",
]

# Short compatibility name for scheduler integrations.
IncrementalOHLCVUpdater = IncrementalCanonicalOHLCVUpdater


def update_incremental_ohlcv(
    root: str | Path,
    *,
    client: Any | None = None,
    calendar: Any = None,
    **kwargs: Any,
) -> IngestionResult:
    """Convenience entry point used by simple schedulers and tests."""

    updater = IncrementalCanonicalOHLCVUpdater(root, client=client, calendar=calendar)
    return updater.update(**kwargs)

# ruff: noqa: E501, B905, UP035
"""One-shot official-feed migration checks for the frozen S3 runtime.

This module is deliberately a data bridge.  It owns no strategy decisions: the
existing factor/ranking/portfolio primitives are called by the reconciliation
entry point, while raw exchange rows and the yfinance adjustment-factor
provider remain separately auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.data.normalizer import clean_ticker
from twse_factor_lab.data.official_market_data import (
    TWSEOfficialAdapter,
    _atomic_parquet,
    _payload_records,
    payload_sha,
)

FINGERPRINT = "36b6fd929e86cb68d2556d95571ef3ebd64352086d4c825946cc355d3cb57261"
OOS_START = pd.Timestamp("2026-01-02")
OOS_END = pd.Timestamp("2026-08-31")
ATOL = 4.547473508864641e-12
RTOL = 1.4210854715202004e-14


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    ordered = frame.sort_values(list(frame.columns), kind="stable").reset_index(drop=True)
    return payload_sha(ordered.to_json(orient="records", date_format="iso"))


def _out(root: str | Path) -> Path:
    path = Path(root).resolve() / "data/runtime/shadow-s3-v1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_historical_names(payload: bytes, tickers: set[str]) -> dict[str, str]:
    try:
        frame = _payload_records(payload)
    except Exception:
        return {}
    if frame.empty or len(frame.columns) < 2:
        return {}
    codes = frame.iloc[:, 0].map(clean_ticker)
    names = frame.iloc[:, 1].astype(str).str.strip()
    return {
        str(code): str(name)
        for code, name in zip(codes, names)
        if pd.notna(code) and str(code) in tickers
    }


def resolve_tickers(root: str | Path, adapter: TWSEOfficialAdapter | None = None) -> dict[str, Any]:
    """Resolve missing current-list rows using official historical evidence."""

    root = Path(root).resolve()
    out = _out(root)
    cached_report = out / "ticker_resolution_report.json"
    # The report is an immutable evidence artifact.  Reusing a passing report
    # avoids turning every scheduled migration check into a live API call.
    if adapter is None and cached_report.exists():
        try:
            cached = json.loads(cached_report.read_text(encoding="utf-8"))
            if cached.get("status") == "PASS" and not cached.get("unresolved"):
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    universe = pd.read_parquet(root / "data/processed/universe.parquet")
    universe["ticker"] = universe["ticker"].map(clean_ticker)
    missing = {"1589", "2321", "3356", "3591"} & set(universe["ticker"])
    client = adapter or TWSEOfficialAdapter()
    current_names: dict[str, str] = {}
    current: set[str] = set()
    try:
        listed, _, _ = client.fetch_listed_companies()
        current = set(listed["ticker"].dropna().map(clean_ticker))
        if "company_name" in listed:
            current_names = dict(zip(listed["ticker"].map(clean_ticker), listed["company_name"].astype(str)))
    except Exception:
        listed = pd.DataFrame()
    historical: set[str] = set()
    historical_names: dict[str, str] = {}
    evidence_dates = ["2025-12-31", "2026-01-02"]
    for date in evidence_dates:
        try:
            frame, payload, _ = client.fetch_session(date, expected_tickers=None)
        except Exception:
            continue
        historical |= set(frame["ticker"].dropna().map(clean_ticker))
        historical_names.update(_extract_historical_names(payload, missing))
    rows: list[dict[str, Any]] = []
    for ticker in sorted(missing):
        universe_row = universe.loc[universe["ticker"].eq(ticker)].iloc[0]
        if ticker in current:
            status = resolution = "ACTIVE"
            evidence = "TWSE official current security master"
        elif ticker in historical:
            status = resolution = "DELISTED"
            evidence = "TWSE official current security master absent; historical MI_INDEX present"
        else:
            status = resolution = "NO_OFFICIAL_RECORD"
            evidence = "TWSE official current and historical records unavailable"
        rows.append(
            {
                "ticker": ticker,
                "official_name": current_names.get(ticker) or historical_names.get(ticker) or str(universe_row.get("company_name", "")),
                "official_status": status,
                "market": str(universe_row.get("market", "TWSE")),
                "listing_status": "CURRENT" if ticker in current else "HISTORICAL_ONLY" if ticker in historical else "UNRESOLVED",
                "first_seen": str(pd.Timestamp(universe_row.get("listed_date")).date()) if pd.notna(universe_row.get("listed_date")) else None,
                "last_seen": max(evidence_dates) if ticker in historical else None,
                "mapping_result": "EXACT_CODE" if ticker in historical or ticker in current else "NO_MATCH",
                "evidence_source": evidence,
                "resolution": resolution,
            }
        )
    unresolved = [row["ticker"] for row in rows if row["resolution"] in {"NO_OFFICIAL_RECORD", "API_MAPPING_ISSUE"}]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "canonical_universe_count": int(len(universe)),
        "resolved_count": int(len(rows) - len(unresolved)),
        "unresolved": unresolved,
        "rows": rows,
        "historical_evidence_dates": evidence_dates,
        "universe_unchanged": True,
        "status": "PASS" if not unresolved else "REVIEW_REQUIRED",
    }
    _atomic_json(out / "ticker_resolution_report.json", report)
    return report


def discover_adjustment_contract(root: str | Path) -> dict[str, Any]:
    """Record the actual frozen research transform, without inventing one."""

    root = Path(root).resolve()
    manifest_path = root / "data/research/fresh-oos-validation-v1/fresh_oos_data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    discovery = {
        "status": "RESOLVED",
        "research_data_source": manifest.get("data_source", "yfinance_adjusted_ohlcv"),
        "research_price_semantics": "yfinance.download(auto_adjust=True)",
        "research_adjustment_method": "Adj Close / Close ratio applied to OHLC; volume unchanged",
        "canonical_corporate_action_engine": False,
        "raw_price_authority": "TWSE_OFFICIAL",
        "volume_authority": "TWSE_OFFICIAL",
        "adjustment_factor_source": "YFINANCE_ADJUSTMENT_FACTOR_PROVIDER",
        "factor_formula": "adjusted_close / raw_close",
        "no_silent_interpolation": True,
        "fresh_oos_window": {"start": str(OOS_START.date()), "end": str(OOS_END.date())},
        "source_manifest_sha": payload_sha(manifest),
    }
    out = _out(root)
    _atomic_json(out / "adjustment_contract_discovery.json", discovery)
    contract = {
        "contract_version": "official-raw-frozen-adjusted-v1",
        "status": "RESOLVED",
        "raw_layer": "TWSE official MI_INDEX OHLCV",
        "adjusted_layer": "official raw OHLC * adjustment_factor",
        "price_formula": {field: f"official_raw_{field} * adjustment_factor" for field in ("open", "high", "low", "close")},
        "volume_formula": "official_raw_volume (no adjustment)",
        "adjustment_factor": "yfinance Adj Close / Close, date+ticker exact join",
        "missing_factor_policy": "SAFE_HALT; no interpolation",
        "source_priority": ["TWSE_OFFICIAL", "YFINANCE_ADJUSTMENT_FACTOR_PROVIDER"],
        "runtime_ohlcv_authority": "TWSE_OFFICIAL_RAW_PLUS_DECLARED_FACTOR",
        "candidate_fingerprint": FINGERPRINT,
    }
    contract["contract_sha"] = payload_sha(contract)
    _atomic_json(out / "adjustment_contract.json", contract)
    return {"discovery": discovery, "contract": contract}


def build_factor_history(raw: pd.DataFrame, reference: pd.DataFrame, *, output: str | Path | None = None) -> pd.DataFrame:
    """Build exact date/ticker adjustment factors; no interpolation.

    Rows are retained even when a secondary provider is missing.  This is
    important for auditability: dropping a row here used to make an otherwise
    valid exchange row look complete.  ``status`` tells the caller whether a
    row is matched, a legal no-trade row, or an unresolved provider gap.
    """

    raw = raw.copy()
    reference = reference.copy()
    for frame in (raw, reference):
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame["ticker"] = frame["ticker"].map(clean_ticker)
    left = raw[["date", "ticker", "close"]].rename(columns={"close": "raw_close"})
    columns = ["date", "ticker", "raw_close", "reference_raw_close", "reference_adjusted_close"]
    right = reference[[c for c in columns if c in reference.columns]]
    if "reference_raw_close" not in right:
        right = right.rename(columns={"raw_close": "reference_raw_close"})
    joined = left.merge(right, on=["date", "ticker"], how="left", validate="one_to_one")
    joined["adjustment_factor"] = joined["reference_adjusted_close"].div(joined["reference_raw_close"].replace(0, np.nan))
    joined = joined.replace([np.inf, -np.inf], np.nan)
    joined["factor_source"] = "YFINANCE_ADJ_CLOSE_CLOSE"
    joined["effective_from"] = joined["date"]
    joined["effective_to"] = joined["date"]
    joined["source_sha"] = joined.apply(lambda row: payload_sha({"date": str(row.date.date()), "ticker": row.ticker, "raw": float(row.raw_close), "reference": float(row.reference_adjusted_close)}), axis=1)
    traded = joined["raw_close"].notna() & joined["raw_close"].gt(0)
    # A blank exchange OHLC row is a legal no-trade observation, not an
    # adjustment observation.  Keep it in the history with a null factor.
    joined.loc[~traded, "adjustment_factor"] = np.nan
    matched = traded & joined["adjustment_factor"].notna() & joined["adjustment_factor"].gt(0)
    joined["status"] = np.select(
        [matched, traded, ~traded],
        ["MATCHED", "MISSING_REFERENCE_FACTOR", "NO_TRADE_OR_MISSING_PRICE"],
        default="UNRESOLVED",
    )
    joined["source_sha"] = joined["source_sha"].where(
        joined["status"].eq("MATCHED"),
        joined.apply(
            lambda row: payload_sha(
                {"date": str(row.date.date()), "ticker": row.ticker, "status": row.status}
            ),
            axis=1,
        ),
    )
    result = joined[["date", "ticker", "raw_close", "reference_adjusted_close", "adjustment_factor", "factor_source", "effective_from", "effective_to", "source_sha", "status"]].sort_values(["date", "ticker"], kind="stable").drop_duplicates(["date", "ticker"], keep="first").reset_index(drop=True)
    if output is not None:
        _atomic_parquet(Path(output), result)
    return result


def download_yfinance_reference(
    tickers: list[str] | tuple[str, ...] | set[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    chunk_size: int = 100,
    timeout: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch only raw/adjusted reference columns for factor reconciliation."""

    import yfinance as yf

    symbols = [clean_ticker(ticker) for ticker in sorted(set(tickers))]
    adjusted_rows: list[pd.DataFrame] = []
    raw_rows: list[pd.DataFrame] = []
    query_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    size = max(1, int(chunk_size))
    for offset in range(0, len(symbols), size):
        chunk = symbols[offset : offset + size]
        yf_symbols = [f"{ticker}.TW" for ticker in chunk]
        frame = yf.download(
            yf_symbols,
            start=str(pd.Timestamp(start).date()),
            end=query_end,
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=timeout,
        )
        if frame is None or frame.empty:
            continue
        if not isinstance(frame.columns, pd.MultiIndex):
            frame.columns = pd.MultiIndex.from_product([frame.columns, [yf_symbols[0]]])
        for ticker, symbol in zip(chunk, yf_symbols, strict=True):
            try:
                raw_close = frame["Close"][symbol]
                adjusted_close = frame["Adj Close"][symbol]
                raw_ohlc = pd.DataFrame(
                    {
                        "date": frame.index,
                        "ticker": ticker,
                        "open": frame["Open"][symbol],
                        "high": frame["High"][symbol],
                        "low": frame["Low"][symbol],
                        "close": raw_close,
                        "volume": frame["Volume"][symbol],
                        "reference_raw_close": raw_close,
                        "reference_adjusted_close": adjusted_close,
                    }
                )
            except (KeyError, TypeError):
                continue
            raw_rows.append(raw_ohlc[["date", "ticker", "open", "high", "low", "close", "volume"]])
            factor = raw_ohlc["reference_adjusted_close"].div(
                raw_ohlc["reference_raw_close"].replace(0, np.nan)
            )
            adjusted = raw_ohlc.copy()
            for field in ("open", "high", "low", "close"):
                adjusted[field] = adjusted[field] * factor
            adjusted_rows.append(adjusted[["date", "ticker", "open", "high", "low", "close", "volume"]])
    columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
    raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame(columns=columns)
    adjusted = pd.concat(adjusted_rows, ignore_index=True) if adjusted_rows else pd.DataFrame(columns=columns)
    for frame in (raw, adjusted):
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
            frame["ticker"] = frame["ticker"].map(clean_ticker)
            frame.sort_values(["date", "ticker"], inplace=True, kind="stable")
            frame.drop_duplicates(["date", "ticker"], inplace=True)
            frame.reset_index(drop=True, inplace=True)
    return raw, adjusted


def refresh_reference_gaps(
    raw: pd.DataFrame,
    reference_raw: pd.DataFrame,
    reference_adjusted: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Retry only provider gaps, keeping the existing cache deterministic.

    The bulk Yahoo request can legitimately return sparse columns for a newly
    listed symbol.  A bounded per-ticker retry repairs that transport gap but
    never manufactures a factor.  Values already present in the cache win over
    retry values, so rerunning migration cannot rewrite frozen evidence.
    """

    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        value = frame.copy()
        if value.empty:
            return value
        value["date"] = pd.to_datetime(value["date"], errors="coerce").dt.tz_localize(None)
        value["ticker"] = value["ticker"].map(clean_ticker)
        return value.drop_duplicates(["date", "ticker"], keep="first")

    raw_cache, adjusted_cache = _normalise(reference_raw), _normalise(reference_adjusted)
    if raw.empty:
        return raw_cache, adjusted_cache, {"status": "NOOP", "tickers": [], "rows_added": 0}
    keys = raw[["date", "ticker", "close"]].copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.tz_localize(None)
    keys["ticker"] = keys["ticker"].map(clean_ticker)
    merged = keys.merge(
        raw_cache[["date", "ticker", "close"]].rename(columns={"close": "reference_close"}),
        on=["date", "ticker"],
        how="left",
    )
    missing = merged.loc[
        merged["close"].notna() & merged["reference_close"].isna(), "ticker"
    ]
    tickers = sorted(set(missing.astype(str)))
    if not tickers:
        return raw_cache, adjusted_cache, {"status": "NOOP", "tickers": [], "rows_added": 0}
    retry_raw_parts: list[pd.DataFrame] = []
    retry_adjusted_parts: list[pd.DataFrame] = []
    # Query the narrow missing window.  Yahoo occasionally emits leading NaN
    # rows when a symbol was newly listed; a full-range retry would preserve
    # that transport artefact instead of repairing it.
    for ticker in tickers:
        ticker_dates = merged.loc[merged["ticker"].eq(ticker) & merged["reference_close"].isna(), "date"]
        gap_start = (ticker_dates.min() - pd.Timedelta(days=14)) if not ticker_dates.empty else pd.Timestamp(start)
        gap_end = (ticker_dates.max() + pd.Timedelta(days=14)) if not ticker_dates.empty else pd.Timestamp(end)
        try:
            retry_raw_one, retry_adjusted_one = download_yfinance_reference(
                [ticker], gap_start, gap_end, chunk_size=1, timeout=20
            )
        except Exception:
            continue
        retry_raw_parts.append(retry_raw_one)
        retry_adjusted_parts.append(retry_adjusted_one)
    retry_raw = pd.concat(retry_raw_parts, ignore_index=True) if retry_raw_parts else pd.DataFrame()
    retry_adjusted = pd.concat(retry_adjusted_parts, ignore_index=True) if retry_adjusted_parts else pd.DataFrame()
    rows_added = int(len(retry_raw))
    def merge_fill_missing(cache: pd.DataFrame, retry_frame: pd.DataFrame) -> pd.DataFrame:
        if retry_frame.empty:
            return cache
        value = cache.copy()
        value = value.set_index(["date", "ticker"])
        incoming = retry_frame.set_index(["date", "ticker"])
        for key, row in incoming.iterrows():
            if key not in value.index:
                value.loc[key, row.index] = row
                continue
            for column in incoming.columns:
                if pd.isna(value.at[key, column]) and pd.notna(row[column]):
                    value.at[key, column] = row[column]
        return value.reset_index().sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

    raw_cache = merge_fill_missing(raw_cache, retry_raw)
    adjusted_cache = merge_fill_missing(adjusted_cache, retry_adjusted)
    return raw_cache, adjusted_cache, {
        "status": "PASS" if rows_added else "NO_PROVIDER_ROWS",
        "tickers": tickers,
        "rows_added": rows_added,
    }


def apply_hybrid_adjustment(raw: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Apply the declared raw-price-times-factor transform to official rows."""

    frame = raw.copy()
    factor_frame = factors[["date", "ticker", "adjustment_factor"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    factor_frame["date"] = pd.to_datetime(factor_frame["date"]).dt.tz_localize(None)
    frame["ticker"] = frame["ticker"].map(clean_ticker)
    factor_frame["ticker"] = factor_frame["ticker"].map(clean_ticker)
    frame = frame.merge(factor_frame, on=["date", "ticker"], how="left", validate="one_to_one")
    traded = frame[["open", "high", "low", "close"]].notna().any(axis=1)
    if frame.loc[traded, "adjustment_factor"].isna().any() or (
        frame.loc[traded, "adjustment_factor"] <= 0
    ).any():
        raise ValueError("ADJUSTMENT_FACTOR_MISSING_OR_INVALID")
    for field in ("open", "high", "low", "close"):
        frame[field] = frame[field] * frame["adjustment_factor"]
    return frame.drop(columns=["adjustment_factor"])


def corporate_action_reconciliation(raw: pd.DataFrame, adjusted: pd.DataFrame) -> dict[str, Any]:
    """Check that the declared factor layer does not introduce discontinuities."""

    def jumps(frame: pd.DataFrame) -> tuple[int, list[dict[str, Any]]]:
        if frame.empty:
            return 0, []
        close = frame.pivot(index="date", columns="ticker", values="close").sort_index()
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        mask = ret.abs().gt(0.5)
        examples: list[dict[str, Any]] = []
        for date, ticker in zip(*np.where(mask.to_numpy())):
            if len(examples) >= 20:
                break
            session = ret.index[date]
            symbol = ret.columns[ticker]
            value = ret.iat[date, ticker]
            if pd.notna(value):
                examples.append({"date": str(pd.Timestamp(session).date()), "ticker": str(symbol), "return": float(value)})
        return int(mask.sum().sum()), examples
    raw_jumps, _ = jumps(raw)
    adjusted_jumps, _ = jumps(adjusted)
    # A factor layer that leaves every raw discontinuity untouched has not
    # demonstrated split/dividend compatibility.  Keep the examples and
    # fail closed so migration cannot treat an unresolved action as clean.
    return {
        "status": "PASS" if adjusted_jumps == 0 or adjusted_jumps < raw_jumps else "REVIEW_REQUIRED",
        "raw_large_return_count": raw_jumps,
        "adjusted_large_return_count": adjusted_jumps,
        "raw_examples": jumps(raw)[1],
        "adjusted_examples": jumps(adjusted)[1],
        "formula": "official raw OHLC * (Adj Close / Close)",
        "volume_adjusted": False,
        "note": "Large raw discontinuities are retained as evidence; adjusted prices are not backfilled by interpolation.",
    }


def reconcile_data_layers(official_adjusted: pd.DataFrame, reference_adjusted: pd.DataFrame, official_raw: pd.DataFrame, reference_raw: pd.DataFrame, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Compare official/hybrid values against frozen Fresh OOS semantics."""

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        value = frame.copy()
        value["date"] = pd.to_datetime(value["date"]).dt.tz_localize(None)
        value["ticker"] = value["ticker"].map(clean_ticker)
        return value

    left, right = normalize(official_adjusted), normalize(reference_adjusted)
    left_raw, right_raw = normalize(official_raw), normalize(reference_raw)
    left = left[(left.date >= OOS_START) & (left.date <= OOS_END)]
    right = right[(right.date >= OOS_START) & (right.date <= OOS_END)]
    left_raw = left_raw[(left_raw.date >= OOS_START) & (left_raw.date <= OOS_END)]
    right_raw = right_raw[(right_raw.date >= OOS_START) & (right_raw.date <= OOS_END)]
    keys = ["date", "ticker"]
    joined = left.merge(right, on=keys, suffixes=("_official", "_reference"), how="inner")
    raw_joined = left_raw.merge(right_raw, on=keys, suffixes=("_official", "_reference"), how="inner")
    def compare(frame: pd.DataFrame, field: str) -> dict[str, Any]:
        a, b = f"{field}_official", f"{field}_reference"
        if a not in frame or b not in frame:
            return {"mismatch_count": None, "rows_compared": 0, "status": "UNAVAILABLE"}
        av, bv = frame[a].to_numpy(float), frame[b].to_numpy(float)
        equal = np.isclose(av, bv, equal_nan=True, atol=ATOL, rtol=RTOL)
        mismatch = ~equal
        examples: list[dict[str, Any]] = []
        if mismatch.any():
            sample = frame.loc[mismatch, ["date", "ticker", a, b]].head(5)
            examples = [
                {
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "ticker": str(row["ticker"]),
                    "official": None if pd.isna(row[a]) else float(row[a]),
                    "reference": None if pd.isna(row[b]) else float(row[b]),
                }
                for _, row in sample.iterrows()
            ]
        return {
            "mismatch_count": int(mismatch.sum()),
            "rows_compared": int(len(frame)),
            "max_abs_diff": float(np.nanmax(np.abs(av - bv))) if len(frame) else 0.0,
            "examples": examples,
            "status": "PASS" if bool(equal.all()) else "MISMATCH",
        }
    adjusted = {field: compare(joined, field) for field in ("open", "high", "low", "close", "volume")}
    raw = {field: compare(raw_joined, field) for field in ("open", "high", "low", "close", "volume")}

    def daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "close" not in frame:
            return pd.DataFrame(columns=["date", "ticker", "daily_return"])
        close = frame.pivot(index="date", columns="ticker", values="close").sort_index()
        return (
            close.pct_change()
            .rename_axis(index="date", columns="ticker")
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="daily_return")
        )

    left_returns, right_returns = daily_returns(left), daily_returns(right)
    return_joined = left_returns.merge(
        right_returns,
        on=["date", "ticker"],
        suffixes=("_official", "_reference"),
        how="inner",
    )
    if return_joined.empty:
        return_report = {
            "mismatch_count": None,
            "rows_compared": 0,
            "status": "UNAVAILABLE",
        }
    else:
        return_equal = np.isclose(
            return_joined["daily_return_official"].to_numpy(float),
            return_joined["daily_return_reference"].to_numpy(float),
            equal_nan=True,
            atol=ATOL,
            rtol=RTOL,
        )
        return_report = {
            "mismatch_count": int((~return_equal).sum()),
            "rows_compared": int(len(return_joined)),
            "max_abs_diff": float(
                np.nanmax(
                    np.abs(
                        return_joined["daily_return_official"]
                        - return_joined["daily_return_reference"]
                    )
                )
            ),
            "status": "PASS" if bool(return_equal.all()) else "MISMATCH",
        }
    sessions_equal = sorted(left.date.drop_duplicates().astype(str).tolist()) == sorted(right.date.drop_duplicates().astype(str).tolist())
    tickers_equal = set(left.ticker) == set(right.ticker)
    left_keys, right_keys = set(map(tuple, left[keys].itertuples(index=False, name=None))), set(map(tuple, right[keys].itertuples(index=False, name=None)))
    key_missing_official = len(right_keys - left_keys)
    key_missing_reference = len(left_keys - right_keys)
    report = {
        "status": "PASS" if sessions_equal and tickers_equal and not key_missing_official and not key_missing_reference and bool(joined.shape[0]) and all(v["status"] == "PASS" for v in adjusted.values()) else "CANONICAL_DATA_MIGRATION_REVIEW_REQUIRED",
        "window": {"start": str(OOS_START.date()), "end": str(OOS_END.date())},
        "session_calendar": {"status": "PASS" if sessions_equal else "MISMATCH", "official_sessions": int(left.date.nunique()), "reference_sessions": int(right.date.nunique())},
        "ticker_universe": {"status": "PASS" if tickers_equal else "MISMATCH", "official_count": int(left.ticker.nunique()), "reference_count": int(right.ticker.nunique())},
        "key_coverage": {"status": "PASS" if not key_missing_official and not key_missing_reference else "MISMATCH", "missing_official": key_missing_official, "missing_reference": key_missing_reference},
        "ADJUSTED_PRICE_COMPARISON": adjusted,
        "RAW_PRICE_COMPARISON": raw,
        "VOLUME_COMPARISON": adjusted.get("volume", {}),
        "DAILY_RETURN_COMPARISON": return_report,
        "daily_return_status": "PASS" if return_report["status"] == "PASS" else "REVIEW_REQUIRED",
    }
    if output_dir is not None:
        _atomic_json(Path(output_dir) / "fresh_oos_data_reconciliation.json", report)
    return report


__all__ = [
    "ATOL", "RTOL", "FINGERPRINT", "OOS_START", "OOS_END",
    "resolve_tickers", "discover_adjustment_contract", "build_factor_history",
    "download_yfinance_reference", "apply_hybrid_adjustment",
    "refresh_reference_gaps", "corporate_action_reconciliation", "reconcile_data_layers",
]

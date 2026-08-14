"""PIT fundamental ingestion, normalization, and research-matrix helpers."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from twse_factor_lab.data.endpoints import get_fundamental_endpoint
from twse_factor_lab.data.normalizer import clean_ticker

PIT_COLUMNS = [
    "ticker",
    "metric",
    "period_end",
    "publication_date",
    "available_date",
    "value",
    "source",
    "pit_status",
]


class FundamentalDataError(RuntimeError):
    """Raised when a source response cannot safely become PIT data."""


RESPONSE_CLASSIFICATIONS = {
    "SUCCESS",
    "TIMEOUT",
    "HTTP_ERROR",
    "SECURITY_BLOCK",
    "EMPTY_RESPONSE",
    "INVALID_HTML",
    "PARSE_ERROR",
}


def classify_response(status_code: int, content: str, *, expects_html: bool) -> str:
    """Classify a source reply before allowing it into the raw success cache."""

    if not 200 <= status_code < 300:
        return "HTTP_ERROR"
    if not content.strip():
        return "EMPTY_RESPONSE"
    lowered = content.lower()
    security_markers = (
        "access denied",
        "security check",
        "security verification",
        "for security reasons",
        "captcha",
        "cf-chl-",
        "請完成驗證",
        "拒絕存取",
        "禁止存取",
        "因為安全性考量",
    )
    if any(marker in lowered for marker in security_markers):
        return "SECURITY_BLOCK"
    if expects_html and "<html" not in lowered and "<table" not in lowered:
        return "INVALID_HTML"
    return "SUCCESS"


class RawFundamentalCache:
    """Small content cache with request metadata for reproducible ingestion."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(url: str, params: dict[str, Any] | None) -> str:
        payload = json.dumps([url, params or {}], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, url: str, params: dict[str, Any] | None = None) -> str | None:
        path = self.root / f"{self._key(url, params)}.raw"
        if path.exists():
            path.with_suffix(".gap.json").unlink(missing_ok=True)
            self.hits += 1
            return path.read_text(encoding="utf-8")
        self.misses += 1
        return None

    def statistics(self) -> dict[str, int]:
        """Return this run's cache activity for the coverage report."""

        return {"hits": self.hits, "misses": self.misses}

    def put(
        self,
        url: str,
        params: dict[str, Any] | None,
        content: str,
        *,
        metadata: dict[str, Any],
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        key = self._key(url, params)
        raw_path = self.root / f"{key}.raw"
        raw_path.write_text(content, encoding="utf-8")
        (self.root / f"{key}.gap.json").unlink(missing_ok=True)
        (self.root / f"{key}.json").write_text(
            json.dumps(
                {
                    "url": url,
                    "params": params or {},
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                    **metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return raw_path

    def record_gap(
        self, url: str, params: dict[str, Any] | None, *, metadata: dict[str, Any]
    ) -> None:
        """Persist the latest failed checkpoint without treating it as cached data."""

        self.root.mkdir(parents=True, exist_ok=True)
        key = self._key(url, params)
        (self.root / f"{key}.gap.json").write_text(
            json.dumps({"url": url, "params": params or {}, **metadata}, indent=2),
            encoding="utf-8",
        )


class FundamentalClient:
    """HTTP client that caches source responses before parsing them."""

    browser_user_agent = "Mozilla/5.0 (compatible; twse-factor-lab/0.1)"

    def __init__(
        self,
        *,
        cache: RawFundamentalCache,
        twse_base_url: str,
        timeout: int = 30,
        mops_timeout: int | None = None,
        throttle_seconds: float = 0.5,
        retry: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.cache = cache
        self.twse_base_url = twse_base_url.rstrip("/")
        self.timeout = timeout
        self.mops_timeout = mops_timeout or timeout
        self.throttle_seconds = max(0.0, throttle_seconds)
        self.retry = max(0, retry)
        self.session = session or requests.Session()

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        request_params = {"params": params or {}, "data": data or {}}
        cached = self.cache.get(url, request_params)
        if cached is not None:
            return cached
        headers = {"User-Agent": self.browser_user_agent}
        error: Exception | None = None
        requested_at = datetime.now(UTC).isoformat()
        for attempt in range(self.retry + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    headers=headers,
                    timeout=(
                        self.mops_timeout
                        if "mopsov.twse.com.tw" in url
                        else self.timeout
                    ),
                )
                response.raise_for_status()
                encoding = "big5" if "t21sc03_" in url else response.encoding or "utf-8"
                content = response.content.decode(encoding, "replace")
                classification = classify_response(
                    getattr(response, "status_code", 200),
                    content,
                    expects_html=any(
                        marker in url
                        for marker in (
                            "ajax_t163",
                            "ajax_t164",
                            "t57sb01",
                            "t21sc03_",
                        )
                    ),
                )
                if classification != "SUCCESS":
                    raise FundamentalDataError(f"{classification}: {url}")
                return self.cache.put(
                    url,
                    request_params,
                    content,
                    metadata={
                        "source": urlparse(url).netloc,
                        "endpoint": urlparse(url).path,
                        "request_params": request_params,
                        "requested_at": requested_at,
                        "response_status": getattr(response, "status_code", 200),
                        "classification": classification,
                        "encoding": encoding,
                    },
                ).read_text(encoding="utf-8")
            except Exception as exc:  # requests errors are retriable here.
                error = exc
                if attempt < self.retry:
                    time.sleep(
                        self.throttle_seconds * (2**attempt)
                        + random.uniform(0, min(0.25, self.throttle_seconds))
                    )
        classification = (
            "TIMEOUT"
            if isinstance(error, requests.Timeout) or "timed out" in str(error).lower()
            else "HTTP_ERROR"
        )
        if isinstance(error, FundamentalDataError):
            classification = str(error).split(":", maxsplit=1)[0]
        self.cache.record_gap(
            url,
            request_params,
            metadata={
                "source": urlparse(url).netloc,
                "endpoint": urlparse(url).path,
                "request_params": request_params,
                "requested_at": requested_at,
                "classification": classification,
                "retry_count": self.retry + 1,
                "error": str(error),
            },
        )
        raise FundamentalDataError(
            f"Source request failed after {self.retry + 1} attempts: {url} ({error})"
        ) from error

    def mops_statement(self, year: int, season: int, statement: str) -> pd.DataFrame:
        endpoint = "mops_income" if statement == "income" else "mops_balance"
        payload = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "year": year,
            "season": season,
            "TYPEK": "sii",
        }
        html = self._request("POST", get_fundamental_endpoint(endpoint), data=payload)
        parsed = parse_mops_statement(html, year=year, season=season)
        metrics = (
            {"revenue", "net_income", "eps"} if statement == "income" else {"equity"}
        )
        return parsed.loc[parsed["metric"].isin(metrics)].reset_index(drop=True)

    def publication_dates(self, ticker: str, year: int) -> pd.DataFrame:
        records: list[pd.DataFrame] = []
        for season in range(1, 5):
            payload = {
                "step": "1",
                "co_id": clean_ticker(ticker),
                "year": year,
                "seamon": season,
                "mtype": "A",
                "dtype": "AI1",
            }
            html = self._request(
                "POST", get_fundamental_endpoint("publication_dates"), data=payload
            )
            try:
                parsed = parse_publication_dates(
                    html, ticker=clean_ticker(ticker), year=year
                )
            except FundamentalDataError:
                if "查無所需資料" not in html:
                    raise
                continue
            if not parsed.empty:
                records.append(parsed)
        return (
            pd.concat(records, ignore_index=True).drop_duplicates()
            if records
            else pd.DataFrame(
                columns=["ticker", "roc_year", "season", "publication_date"]
            )
        )

    def monthly_revenue(self, year: int, month: int) -> pd.DataFrame:
        url = get_fundamental_endpoint("monthly_revenue", year=year, month=month)
        return parse_monthly_revenue(self._request("GET", url), year=year, month=month)

    def valuation_daily(self, date: str | pd.Timestamp) -> pd.DataFrame:
        date_value = pd.Timestamp(date).strftime("%Y%m%d")
        url = get_fundamental_endpoint("valuation_daily")
        raw = self._request("GET", url, params={"date": date_value, "response": "json"})
        return parse_valuation_daily(raw, date=date)


class _TableParser(HTMLParser):
    """Minimal HTML-table reader; MOPS does not justify an extra parser package."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[tuple[list[list[str]], bool]] = []
        self.rows: list[list[str]] = []
        self.row: list[str] = []
        self.cell: list[str] | None = None
        self.in_table = False
        self.first_row_has_header = False
        self.row_has_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.in_table = True
            self.rows = []
            self.first_row_has_header = False
        elif self.in_table and tag == "tr":
            self.row = []
            self.row_has_header = False
        elif self.in_table and tag in {"td", "th"}:
            self.cell = []
            self.row_has_header |= tag == "th"

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join(self.cell).strip())
            self.cell = None
        elif self.in_table and tag == "tr" and self.row:
            if not self.rows:
                self.first_row_has_header = self.row_has_header
            self.rows.append(self.row)
        elif tag == "table" and self.in_table:
            if self.rows:
                self.tables.append((self.rows, self.first_row_has_header))
            self.in_table = False


def _tables(html: str) -> list[pd.DataFrame]:
    parser = _TableParser()
    parser.feed(html)
    tables: list[pd.DataFrame] = []
    for rows, has_header in parser.tables:
        has_header = (
            has_header
            or bool(rows)
            and any(
                value in {"公司", "代號", "公司代號", "ticker"} for value in rows[0]
            )
        )
        if has_header and len(rows) > 1:
            header_index = next(
                (
                    index
                    for index, row in enumerate(rows)
                    if any(
                        "公司" in value or "ticker" in value.lower() for value in row
                    )
                ),
                0,
            )
            header = rows[header_index]
            data = [
                row[: len(header)]
                for row in rows[header_index + 1 :]
                if len(row) >= len(header)
            ]
            if data:
                tables.append(pd.DataFrame(data, columns=header))
        else:
            tables.append(pd.DataFrame(rows))
    if not tables:
        raise FundamentalDataError("Source HTML contains no table")
    return [table for table in tables if not table.empty]


def _flatten(columns: pd.Index) -> list[str]:
    return [
        " ".join(map(str, value)).strip() if isinstance(value, tuple) else str(value)
        for value in columns
    ]


def _column(frame: pd.DataFrame, terms: tuple[str, ...]) -> str | None:
    for column in frame.columns:
        text = re.sub(r"\s+", "", str(column)).lower()
        if any(term.lower() in text for term in terms):
            return str(column)
    return None


def parse_mops_statement(html: str, *, year: int, season: int) -> pd.DataFrame:
    """Parse MOPS statement tables into common financial metrics where available."""

    rows: list[dict[str, Any]] = []
    mappings = {
        "revenue": ("營業收入", "收入合計", "revenue"),
        "net_income": ("本期淨利", "本期淨", "淨利", "net income"),
        "eps": ("基本每股盈餘", "基本每股", "eps"),
        "equity": ("權益總計", "權益", "equity"),
    }
    common_aliases = {
        "revenue": ("營業收入", "收入"),
        "net_income": ("本期淨利", "淨利"),
        "eps": ("基本每股盈餘", "每股盈餘"),
        "equity": ("權益總額", "權益總計", "權益"),
    }
    for table in _tables(html):
        frame = table.copy()
        frame.columns = _flatten(frame.columns)
        ticker_column = _column(frame, ("公司代號", "證券代號", "公司", "ticker"))
        ticker_column = ticker_column or _column(
            frame, ("公司代號", "證券代號", "股票代號")
        )
        ticker_column = ticker_column or _column(
            frame, ("公司代號", "證券代號", "代號")
        )
        if ticker_column is None:
            continue
        for _, row in frame.iterrows():
            ticker = clean_ticker(row[ticker_column])
            if pd.isna(ticker) or not str(ticker).isdigit():
                continue
            for metric, aliases in mappings.items():
                if metric == "equity":
                    column = _column(frame, ("權益總計", "權益合計", "權益總額"))
                    column = column or _column(
                        frame, (*aliases, *common_aliases[metric])
                    )
                else:
                    column = _column(frame, (*aliases, *common_aliases[metric]))
                value = (
                    pd.to_numeric(str(row[column]).replace(",", ""), errors="coerce")
                    if column
                    else pd.NA
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "roc_year": year,
                        "season": season,
                        "metric": metric,
                        "value": value,
                    }
                )
    if not rows:
        raise FundamentalDataError("MOPS statement lacks a recognizable ticker table")
    return pd.DataFrame(rows)


def parse_publication_dates(html: str, *, ticker: str, year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table in _tables(html):
        frame = table.copy()
        frame.columns = _flatten(frame.columns)
        period_column = _column(frame, ("資料年度", "report year"))
        date_column = _column(frame, ("上傳日期", "upload date"))
        period_column = period_column or next(
            (
                column
                for column in frame.columns
                if frame[column].astype(str).str.contains("\\d{2,3}\\s*\u5e74").any()
            ),
            None,
        )
        date_column = date_column or next(
            (
                column
                for column in frame.columns
                if frame[column]
                .astype(str)
                .str.contains(r"\d{2,3}[/-]\d{1,2}[/-]\d{1,2}")
                .any()
            ),
            None,
        )
        if period_column is None or date_column is None:
            continue
        for _, row in frame.iterrows():
            period_match = re.search(
                r"(\d{2,3})\s*年.*?([1-4])\s*季", str(row[period_column])
            )
            period_match = period_match or re.search(
                "(\\d{2,3})\\s*\u5e74.*?([1-4\u4e00\u4e8c\u4e09\u56db])\\s*\u5b63",
                str(row[period_column]),
            )
            date_match = re.search(
                r"(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})", str(row[date_column])
            )
            if period_match and date_match:
                roc_year, season_text = period_match.groups()
                roc_year = int(roc_year)
                season = (
                    int(season_text)
                    if season_text.isdigit()
                    else "一二三四".index(season_text) + 1
                )
                date_year, month, day = map(int, date_match.groups())
                rows.append(
                    {
                        "ticker": ticker,
                        "roc_year": roc_year,
                        "season": season,
                        "publication_date": pd.Timestamp(date_year + 1911, month, day),
                    }
                )
    if rows:
        return pd.DataFrame(rows).drop_duplicates()
    for table in _tables(html):
        for value in table.astype(str).to_numpy().ravel():
            date_match = re.search(r"(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})", value)
            season_match = re.search(r"第?([1-4])季|Q([1-4])", value, re.I)
            if date_match:
                season_match = season_match or re.search(
                    r"(?:第?\s*([1-4])\s*季|Q\s*([1-4]))", value, re.I
                )
            if date_match and season_match:
                roc_year, month, day = map(int, date_match.groups())
                season = int(next(group for group in season_match.groups() if group))
                rows.append(
                    {
                        "ticker": ticker,
                        "roc_year": roc_year if roc_year > 100 else year,
                        "season": season,
                        "publication_date": pd.Timestamp(roc_year + 1911, month, day),
                    }
                )
    return (
        pd.DataFrame(rows).drop_duplicates()
        if rows
        else pd.DataFrame(columns=["ticker", "roc_year", "season", "publication_date"])
    )


def parse_monthly_revenue(html: str, *, year: int, month: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table in _tables(html):
        frame = table.copy()
        frame.columns = _flatten(frame.columns)
        ticker_col = _column(frame, ("公司代號", "證券代號", "ticker"))
        revenue_col = _column(frame, ("當月營收", "本月營收", "revenue"))
        prior_col = _column(frame, ("去年當月", "去年同期", "prior"))
        ticker_col = ticker_col or _column(frame, ("公司代號", "證券代號"))
        revenue_col = revenue_col or _column(frame, ("當月營收", "營業收入"))
        prior_col = prior_col or _column(frame, ("去年當月營收", "去年同期"))
        if ticker_col is None or revenue_col is None:
            continue
        for _, row in frame.iterrows():
            ticker = clean_ticker(row[ticker_col])
            if pd.isna(ticker) or not str(ticker).isdigit():
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "revenue_month": pd.Timestamp(year + 1911, month, 1)
                    + pd.offsets.MonthEnd(0),
                    "revenue": pd.to_numeric(
                        str(row[revenue_col]).replace(",", ""), errors="coerce"
                    ),
                    "prior_year_revenue": pd.to_numeric(
                        str(row[prior_col]).replace(",", ""), errors="coerce"
                    )
                    if prior_col
                    else pd.NA,
                }
            )
    if not rows:
        raise FundamentalDataError("Monthly-revenue HTML lacks required columns")
    return pd.DataFrame(rows)


def parse_valuation_daily(raw: str, *, date: str | pd.Timestamp) -> pd.DataFrame:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise FundamentalDataError("BWIBBU_d returned a snapshot-style payload")
    if payload.get("stat") != "OK":
        return pd.DataFrame(
            columns=["date", "ticker", "pe", "pb", "dividend_yield", "report_period"]
        )
    frame = pd.DataFrame(payload.get("data", []), columns=payload.get("fields", []))
    if frame.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "pe", "pb", "dividend_yield", "report_period"]
        )
    frame.columns = _flatten(frame.columns)
    ticker = _column(frame, ("證券代號", "公司代號", "ticker"))
    pe = _column(frame, ("本益比", "pe"))
    pb = _column(frame, ("股價淨值比", "pb"))
    dividend = _column(frame, ("殖利率", "dividend"))
    report_period = _column(frame, ("財報年/季", "財報年季"))
    ticker = ticker or _column(frame, ("證券代號", "股票代號"))
    pe = pe or _column(frame, ("本益比",))
    pb = pb or _column(frame, ("股價淨值比",))
    dividend = dividend or _column(frame, ("殖利率",))
    report_period = report_period or _column(frame, ("財報年/季", "財報年季"))
    if ticker is None:
        raise FundamentalDataError("BWIBBU_d lacks ticker column")
    result = pd.DataFrame(
        {"date": pd.Timestamp(date), "ticker": frame[ticker].map(clean_ticker)}
    )
    for name, column in {"pe": pe, "pb": pb, "dividend_yield": dividend}.items():
        result[name] = (
            pd.to_numeric(frame[column].replace("-", pd.NA), errors="coerce")
            if column
            else pd.NA
        )
    result["report_period"] = frame[report_period] if report_period else pd.NA
    return result


def next_trading_day(date: object, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(trading_days))))
    timestamp = pd.Timestamp(date)
    if timestamp < days[0]:
        raise FundamentalDataError(
            "Publication date precedes trading calendar coverage"
        )
    position = days.searchsorted(timestamp, side="right")
    if position >= len(days):
        raise FundamentalDataError("No trading day after publication date")
    return days[position]


def _available_date_or_gap(
    date: object, trading_days: pd.DatetimeIndex
) -> pd.Timestamp:
    """Never fabricate an available_date outside the known trading calendar."""

    try:
        return next_trading_day(date, trading_days)
    except FundamentalDataError:
        return pd.NaT


def _quarter_end(roc_year: int, season: int) -> pd.Timestamp:
    return pd.Timestamp(roc_year + 1911, season * 3, 1) + pd.offsets.MonthEnd(0)


def build_financial_pit(
    statements: pd.DataFrame,
    publications: pd.DataFrame,
    *,
    trading_days: pd.DatetimeIndex,
    missing_publication_date: str = "exclude",
) -> pd.DataFrame:
    frame = statements.merge(
        publications, how="left", on=["ticker", "roc_year", "season"]
    )
    frame["period_end"] = [
        _quarter_end(int(year), int(season))
        for year, season in zip(frame["roc_year"], frame["season"], strict=True)
    ]
    missing = frame["publication_date"].isna()
    if missing.any() and missing_publication_date != "legal_deadline":
        frame = frame.loc[~missing].copy()
    if missing.any() and missing_publication_date == "legal_deadline":
        frame.loc[missing, "publication_date"] = frame.loc[
            missing, "period_end"
        ] + pd.Timedelta(days=45)
    frame["publication_date"] = pd.to_datetime(frame["publication_date"])
    frame["available_date"] = frame["publication_date"].map(
        lambda value: _available_date_or_gap(value, trading_days)
    )
    frame["source"] = "mops+doc.twse"
    frame["pit_status"] = "PUBLICATION_DATE_AWARE"
    if missing_publication_date == "legal_deadline" and missing.any():
        frame.loc[missing, "source"] = "mops+legal_deadline"
        frame.loc[missing, "pit_status"] = "PERIOD_ONLY"
    return (
        frame[PIT_COLUMNS]
        .dropna(subset=["value", "available_date"])
        .reset_index(drop=True)
    )


def build_monthly_revenue_pit(
    revenue: pd.DataFrame, *, trading_days: pd.DatetimeIndex
) -> pd.DataFrame:
    frame = revenue.copy()
    frame["period_end"] = pd.to_datetime(frame["revenue_month"])
    frame["publication_date"] = frame["period_end"] + pd.Timedelta(days=10)
    frame["available_date"] = frame["publication_date"].map(
        lambda value: _available_date_or_gap(value, trading_days)
    )
    frame["metric"] = "revenue"
    frame["value"] = frame["revenue"]
    frame["source"] = "mops_t21sc03"
    frame["pit_status"] = "PERIOD_ONLY"
    return (
        frame[PIT_COLUMNS]
        .dropna(subset=["value", "available_date"])
        .reset_index(drop=True)
    )


def build_valuation_pit(valuation: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for metric in ("pe", "pb", "dividend_yield"):
        frame = (
            valuation[["date", "ticker", metric]]
            .rename(columns={"date": "period_end", metric: "value"})
            .copy()
        )
        frame["period_end"] = pd.to_datetime(frame["period_end"])
        frame["publication_date"] = frame["period_end"]
        frame["available_date"] = frame["period_end"]
        frame["metric"] = metric
        frame["source"] = "twse_BWIBBU_d"
        frame["pit_status"] = "FULL_PIT"
        records.append(frame[PIT_COLUMNS])
    return pd.concat(records, ignore_index=True).dropna(subset=["value"])


def derive_metrics(pit: pd.DataFrame) -> pd.DataFrame:
    records = [pit]
    base = pit.pivot_table(
        index=["ticker", "period_end"], columns="metric", values="value", aggfunc="last"
    )
    dates = pit.groupby(["ticker", "period_end"], as_index=False)[
        "available_date"
    ].max()
    if {"net_income", "equity"}.issubset(base.columns):
        roe = base.assign(value=base["net_income"] / base["equity"]).reset_index()
        roe = roe.merge(dates, on=["ticker", "period_end"])
        roe = roe.dropna(subset=["value"])
        roe["metric"] = "roe"
        roe["publication_date"] = roe["available_date"]
        roe["source"] = "derived_net_income_over_equity"
        roe["pit_status"] = "PUBLICATION_DATE_AWARE"
        records.append(roe[PIT_COLUMNS])
    revenue = pit[pit["metric"] == "revenue"].copy()
    revenue = revenue.sort_values(["ticker", "period_end", "available_date"])
    revenue["previous"] = revenue.groupby("ticker")["value"].shift(12)
    current = pd.to_numeric(revenue["value"], errors="coerce")
    # A zero prior-year base makes YoY growth undefined; exclude rather than
    # fabricate +/-inf (previous dtype can be a plain Python object here, so
    # true-division by zero raises ZeroDivisionError instead of producing inf).
    previous = pd.to_numeric(revenue["previous"], errors="coerce").replace(0, pd.NA)
    revenue["value"] = current.div(previous).sub(1)
    revenue = revenue.dropna(subset=["value"])
    revenue["metric"] = "revenue_yoy"
    revenue["source"] = "derived_revenue_yoy"
    records.append(revenue[PIT_COLUMNS])
    return pd.concat(records, ignore_index=True)


def validate_pit_records(pit: pd.DataFrame) -> None:
    missing = set(PIT_COLUMNS) - set(pit.columns)
    if missing:
        raise KeyError(f"Missing PIT columns: {sorted(missing)}")
    frame = pit.copy()
    for column in ("period_end", "publication_date", "available_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if frame[["period_end", "publication_date", "available_date"]].isna().any().any():
        raise ValueError("PIT records contain invalid dates")
    if (frame["publication_date"] < frame["period_end"]).any() or (
        frame["available_date"] < frame["publication_date"]
    ).any():
        raise ValueError("PIT records violate period/publication/available ordering")
    if frame.duplicated(["ticker", "metric", "period_end", "publication_date"]).any():
        raise ValueError("PIT records contain duplicate version keys")


def build_fundamental_matrix(
    pit: pd.DataFrame, research_universe: pd.DataFrame
) -> pd.DataFrame:
    validate_pit_records(pit)
    if (pit["pit_status"] == "SNAPSHOT_ONLY").any():
        raise ValueError("SNAPSHOT_ONLY records cannot enter the historical matrix")
    pit = pit.copy()
    for column in ("period_end", "publication_date", "available_date"):
        pit[column] = pd.to_datetime(pit[column]).astype("datetime64[ns]")
    eligible = research_universe.loc[
        research_universe["is_eligible"].astype(bool), ["date", "ticker"]
    ].copy()
    eligible["date"] = pd.to_datetime(eligible["date"]).astype("datetime64[ns]")
    eligible["ticker"] = eligible["ticker"].astype(str)
    records: list[pd.DataFrame] = []
    for (ticker, _metric), values in pit.groupby(["ticker", "metric"], sort=False):
        left = eligible.loc[eligible["ticker"] == str(ticker)].sort_values("date")
        if left.empty:
            continue
        right = values.drop(columns="ticker").sort_values("available_date")
        joined = pd.merge_asof(
            left, right, left_on="date", right_on="available_date", direction="backward"
        )
        joined = joined.dropna(subset=["metric"])
        records.append(joined)
    if not records:
        return pd.DataFrame(columns=["date", *PIT_COLUMNS])
    result = pd.concat(records, ignore_index=True)
    return (
        result[["date", *PIT_COLUMNS]]
        .sort_values(["date", "ticker", "metric"])
        .reset_index(drop=True)
    )


def query_fundamentals(
    matrix: pd.DataFrame, research_date: object, ticker: str
) -> pd.DataFrame:
    date = pd.Timestamp(research_date)
    return (
        matrix.loc[
            (pd.to_datetime(matrix["date"]) == date)
            & (matrix["ticker"] == clean_ticker(ticker))
        ]
        .sort_values("metric")
        .reset_index(drop=True)
    )


def build_fundamental_coverage(
    pit: pd.DataFrame, research_universe: pd.DataFrame
) -> pd.DataFrame:
    eligible_tickers = set(
        research_universe.loc[
            research_universe["is_eligible"].astype(bool), "ticker"
        ].astype(str)
    )
    denominator = len(eligible_tickers)
    pit = pit.loc[pit["ticker"].astype(str).isin(eligible_tickers)]
    counts = {
        "financial_statement_coverage": pit.loc[
            pit["source"].eq("mops+doc.twse"), "ticker"
        ].nunique(),
        "publication_date_coverage": pit.loc[
            pit["pit_status"].eq("PUBLICATION_DATE_AWARE"), "ticker"
        ].nunique(),
        "monthly_revenue_coverage": pit.loc[
            pit["metric"].eq("revenue"), "ticker"
        ].nunique(),
        "valuation_pit_coverage": pit.loc[
            pit["pit_status"].eq("FULL_PIT"), "ticker"
        ].nunique(),
    }
    rows = [
        {
            "metric": name,
            "covered_tickers": value,
            "eligible_tickers": denominator,
            "coverage_ratio": value / denominator if denominator else 0.0,
        }
        for name, value in counts.items()
    ]
    ready = pit.loc[pit["pit_status"].ne("SNAPSHOT_ONLY"), "ticker"].nunique()
    rows.append(
        {
            "metric": "pit_ready_ratio",
            "covered_tickers": ready,
            "eligible_tickers": denominator,
            "coverage_ratio": ready / denominator if denominator else 0.0,
        }
    )
    return pd.DataFrame(rows)


def build_publication_coverage_audit(
    statements: pd.DataFrame,
    publications: pd.DataFrame,
    *,
    years: tuple[int, ...] = (102, 103, 104),
) -> pd.DataFrame:
    """Audit actual doc.twse matches; absent dates are never invented."""

    keys = ["ticker", "roc_year", "season"]
    expected = (
        statements[keys].drop_duplicates()
        if not statements.empty
        else pd.DataFrame(columns=keys)
    )
    found = (
        publications[keys].drop_duplicates()
        if not publications.empty
        else pd.DataFrame(columns=keys)
    )
    rows: list[dict[str, object]] = []
    for year in years:
        period = expected.loc[expected["roc_year"] == year]
        merged = period.merge(found, on=keys, how="left", indicator=True)
        statement_count = len(period)
        found_count = int(merged["_merge"].eq("both").sum())
        missing = statement_count - found_count
        rows.append(
            {
                "year": year + 1911,
                "financial_statement_count": statement_count,
                "publication_date_found": found_count,
                "publication_date_missing": missing,
                "coverage_ratio": found_count / statement_count
                if statement_count
                else 0.0,
                "downgraded_count": 0,
                "excluded_count": missing,
            }
        )
    return pd.DataFrame(rows)


def build_fundamental_coverage_report(
    coverage: pd.DataFrame,
    pit: pd.DataFrame,
    *,
    publication_audit: pd.DataFrame | None = None,
    live_pipeline_status: str = "PASS",
    failed_requests: list[str] | None = None,
    cache_statistics: dict[str, int] | None = None,
) -> str:
    statuses = pit["pit_status"].value_counts().to_dict()
    lines = [
        "# Fundamental Coverage Report",
        "",
        "## Live Pipeline Status",
        "",
        f"- status: {live_pipeline_status}",
    ]
    if cache_statistics:
        lines += [
            f"- raw_cache_hits: {cache_statistics['hits']}",
            f"- raw_cache_misses: {cache_statistics['misses']}",
        ]
    lines += ["", "## Coverage", ""]
    lines += [
        f"- {row.metric}: {row.coverage_ratio:.4f} "
        f"({row.covered_tickers}/{row.eligible_tickers})"
        for row in coverage.itertuples(index=False)
    ]
    lines += ["", "## PIT Status", ""]
    lines += [f"- {name}: {count}" for name, count in sorted(statuses.items())]
    if publication_audit is not None:
        lines += ["", "## 2013–2015 Publication-Date Coverage", ""]
        lines += [
            "- {year}: financial_statement_count={financial_statement_count}, "
            "publication_date_found={publication_date_found}, "
            "publication_date_missing={publication_date_missing}, "
            "coverage_ratio={coverage_ratio:.4f}, downgraded_count={downgraded_count}, "
            "excluded_count={excluded_count}".format(**row)
            for row in publication_audit.to_dict("records")
        ]
    lines += ["", "## Failed Requests", ""]
    lines += [f"- {item}" for item in (failed_requests or ["None"])]
    lines += [
        "",
        "## Known Limitations",
        "",
        "- Monthly revenue is PERIOD_ONLY, using the statutory next-month-10th policy.",
        "- Financial statements require doc.twse publication-date coverage and may be excluded when absent.",
        "- 2013–2015 publication-date completeness and financial-industry mappings require coverage review.",
        "- Valuation snapshots remain SNAPSHOT_ONLY and are excluded from historical matrices.",
        "- Records whose publication_date precedes the OHLCV trading calendar's "
        "earliest date have no real next-trading-day to derive available_date from; "
        "they are excluded from fundamental_pit rather than fabricated (see "
        "Publication-Date Coverage above for raw statement/publication counts in "
        "that range).",
    ]
    return "\n".join(lines)

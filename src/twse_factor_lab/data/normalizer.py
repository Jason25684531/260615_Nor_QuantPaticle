"""Normalize market data into stable research schemas."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def to_snake_case(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text)
    return text.strip("_").lower()


def clean_ticker(value: object) -> str:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    match = re.search(r"\d+", text)
    return match.group(0) if match else text.replace(".TW", "")


def _first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {to_snake_case(column): column for column in frame.columns}
    compact_lookup = {
        to_snake_case(column).replace("_", ""): column for column in frame.columns
    }
    for candidate in candidates:
        key = to_snake_case(candidate)
        if key in lookup:
            return lookup[key]
        compact_key = key.replace("_", "")
        if compact_key in compact_lookup:
            return compact_lookup[compact_key]
    return None


def _series_or_na(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    column = _first_existing(frame, candidates)
    if column is None:
        return pd.Series([pd.NA] * len(frame), index=frame.index)
    return frame[column]


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.replace({"": pd.NA, "-": pd.NA, "--": pd.NA, "N/A": pd.NA})
    if cleaned.dtype == object:
        cleaned = cleaned.astype(str).str.replace(",", "", regex=False)
        cleaned = cleaned.replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_universe(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    result = pd.DataFrame(index=frame.index)
    result["ticker"] = _series_or_na(
        frame,
        ["ticker", "code", "stock_code", "證券代號", "股票代號", "公司代號"],
    ).map(clean_ticker)
    result["company_name"] = _series_or_na(
        frame,
        ["company_name", "name", "公司名稱", "股票名稱"],
    )
    result["industry"] = _series_or_na(
        frame,
        ["industry", "industry_category", "產業別"],
    )
    market = _series_or_na(frame, ["market", "市場別", "上市別"]).replace("", pd.NA)
    result["market"] = market.fillna("TWSE")
    result["listed_date"] = pd.to_datetime(
        _series_or_na(frame, ["listed_date", "listeddate", "上市日期"]),
        errors="coerce",
    )
    return result[
        ["ticker", "company_name", "industry", "market", "listed_date"]
    ].reset_index(drop=True)


def normalize_valuation(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    result = pd.DataFrame(index=frame.index)
    result["date"] = pd.to_datetime(
        _series_or_na(frame, ["date", "日期", "資料日期"]),
        errors="coerce",
    )
    result["ticker"] = _series_or_na(
        frame,
        ["ticker", "code", "股票代號"],
    ).map(clean_ticker)
    result["pe"] = _numeric(
        _series_or_na(frame, ["pe", "peratio", "pe_ratio", "本益比"])
    )
    result["pb"] = _numeric(
        _series_or_na(frame, ["pb", "pbratio", "pb_ratio", "股價淨值比"])
    )
    result["dividend_yield"] = _numeric(
        _series_or_na(
            frame,
            ["dividend_yield", "dividendyield", "殖利率", "股利殖利率"],
        )
    )
    return result[["date", "ticker", "pe", "pb", "dividend_yield"]].reset_index(
        drop=True
    )


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    result = pd.DataFrame(index=frame.index)
    result["date"] = pd.to_datetime(
        _series_or_na(frame, ["date", "datetime", "日期"]),
        errors="coerce",
    )
    result["ticker"] = _series_or_na(
        frame,
        ["ticker", "code", "股票代號"],
    ).map(clean_ticker)
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = _numeric(_series_or_na(frame, [column, column.title()]))
    return result[
        ["date", "ticker", "open", "high", "low", "close", "volume"]
    ].reset_index(drop=True)

"""yfinance OHLCV fallback client."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from twse_factor_lab.data.normalizer import clean_ticker

DownloadFunc = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class OhlcvDownloadResult:
    data: pd.DataFrame
    failed_tickers: list[str]


class YFinanceClient:
    """Download Taiwan equity OHLCV through yfinance."""

    def __init__(self, download_func: DownloadFunc | None = None) -> None:
        self._download = download_func or yf.download

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        ticker = str(ticker).strip().upper()
        if ticker.endswith(".TW"):
            return ticker
        return f"{clean_ticker(ticker)}.TW"

    def download_ohlcv(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
    ) -> OhlcvDownloadResult:
        frames: list[pd.DataFrame] = []
        failed: list[str] = []
        query_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        for ticker in tickers:
            raw_ticker = clean_ticker(str(ticker))
            yf_ticker = self.normalize_ticker(str(ticker))
            try:
                frame = self._download(
                    yf_ticker,
                    start=start,
                    end=query_end,
                    progress=False,
                    auto_adjust=True,
                )
            except Exception:
                failed.append(raw_ticker)
                continue

            if frame is None or frame.empty:
                failed.append(raw_ticker)
                continue

            formatted = self._format_download(frame, raw_ticker)
            if formatted.empty:
                failed.append(raw_ticker)
                continue
            frames.append(formatted)

        if not frames:
            return OhlcvDownloadResult(pd.DataFrame(), failed)

        return OhlcvDownloadResult(pd.concat(frames, ignore_index=True), failed)

    @staticmethod
    def _format_download(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        frame = frame.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(parts[0]) for parts in frame.columns]

        frame = frame.reset_index()
        rename_map = {
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        frame = frame.rename(columns=rename_map)
        frame["ticker"] = ticker
        columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
        if not set(columns).issubset(frame.columns):
            return pd.DataFrame(columns=columns)
        return frame[columns]

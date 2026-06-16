import pandas as pd

from twse_factor_lab.data.normalizer import (
    normalize_ohlcv,
    normalize_universe,
    normalize_valuation,
)


def test_normalize_universe_outputs_stable_schema_and_clean_tickers():
    raw = pd.DataFrame(
        {
            "Code": ["2330.TW"],
            "Name": ["TSMC"],
            "Industry": ["Semiconductor"],
            "Market": ["上市"],
            "ListedDate": ["1994-09-05"],
        }
    )

    frame = normalize_universe(raw)

    assert list(frame.columns) == [
        "ticker",
        "company_name",
        "industry",
        "market",
        "listed_date",
    ]
    assert frame.loc[0, "ticker"] == "2330"
    assert pd.api.types.is_datetime64_any_dtype(frame["listed_date"])


def test_normalize_universe_supports_twse_chinese_columns_and_defaults_market():
    raw = pd.DataFrame(
        {
            "公司代號": ["1101"],
            "公司名稱": ["臺灣水泥股份有限公司"],
            "產業別": ["01"],
            "上市日期": ["19620209"],
        }
    )

    frame = normalize_universe(raw)

    assert frame.loc[0, "ticker"] == "1101"
    assert frame.loc[0, "company_name"] == "臺灣水泥股份有限公司"
    assert frame.loc[0, "industry"] == "01"
    assert frame.loc[0, "market"] == "TWSE"


def test_normalize_valuation_outputs_numeric_schema_and_nan_values():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Code": ["2330"],
            "PEratio": ["20.5"],
            "PBratio": ["-"],
            "DividendYield": ["3.2"],
        }
    )

    frame = normalize_valuation(raw)

    assert list(frame.columns) == ["date", "ticker", "pe", "pb", "dividend_yield"]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert frame.loc[0, "ticker"] == "2330"
    assert frame.loc[0, "pe"] == 20.5
    assert pd.isna(frame.loc[0, "pb"])
    assert frame.loc[0, "dividend_yield"] == 3.2


def test_normalize_ohlcv_outputs_long_schema():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Ticker": ["2330.TW"],
            "Open": ["1"],
            "High": ["2"],
            "Low": ["0.5"],
            "Close": ["1.5"],
            "Volume": ["1000"],
        }
    )

    frame = normalize_ohlcv(raw)

    assert list(frame.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert frame.loc[0, "ticker"] == "2330"
    assert frame.loc[0, "volume"] == 1000

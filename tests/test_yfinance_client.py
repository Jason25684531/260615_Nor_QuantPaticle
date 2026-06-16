import pandas as pd

from twse_factor_lab.data.yfinance_client import YFinanceClient


def test_yfinance_client_adds_taiwan_suffix():
    assert YFinanceClient.normalize_ticker("2330") == "2330.TW"


def test_yfinance_client_preserves_existing_suffix():
    assert YFinanceClient.normalize_ticker("2330.TW") == "2330.TW"


def test_yfinance_client_downloads_ohlcv_and_tracks_failed_tickers():
    def fake_download(symbol, start, end, progress, auto_adjust):
        assert start == "2024-01-01"
        assert end == "2024-01-03"
        assert progress is False
        assert auto_adjust is False
        if symbol == "9999.TW":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "Open": [1.0],
                "High": [2.0],
                "Low": [0.5],
                "Close": [1.5],
                "Volume": [1000],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )

    client = YFinanceClient(download_func=fake_download)

    result = client.download_ohlcv(["2330", "9999"], "2024-01-01", "2024-01-03")

    assert result.failed_tickers == ["9999"]
    assert result.data["ticker"].tolist() == ["2330"]
    assert result.data["close"].tolist() == [1.5]

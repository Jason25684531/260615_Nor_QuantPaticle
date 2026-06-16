import pytest

from twse_factor_lab.data.endpoints import get_endpoint


def test_endpoint_registry_returns_supported_paths():
    assert get_endpoint("listed_companies") == "opendata/t187ap03_L"
    assert get_endpoint("daily_prices") == "exchangeReport/STOCK_DAY_ALL"
    assert get_endpoint("valuation") == "exchangeReport/BWIBBU_ALL"
    assert get_endpoint("trading_calendar") == "exchangeReport/TWTB4U"


def test_endpoint_registry_rejects_unknown_key():
    with pytest.raises(KeyError, match="unknown"):
        get_endpoint("unknown")

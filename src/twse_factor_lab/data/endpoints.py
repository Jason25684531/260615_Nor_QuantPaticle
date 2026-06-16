"""TWSE OpenAPI endpoint registry."""

ENDPOINTS: dict[str, str] = {
    "listed_companies": "opendata/t187ap03_L",
    "daily_prices": "exchangeReport/STOCK_DAY_ALL",
    "valuation": "exchangeReport/BWIBBU_ALL",
    "trading_calendar": "exchangeReport/TWTB4U",
}


def get_endpoint(key: str) -> str:
    """Return a registered TWSE OpenAPI path."""
    try:
        return ENDPOINTS[key]
    except KeyError as exc:
        available = ", ".join(sorted(ENDPOINTS))
        raise KeyError(
            f"Unknown TWSE endpoint key: {key}. Available: {available}"
        ) from exc

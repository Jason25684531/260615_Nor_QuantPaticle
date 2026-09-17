"""TWSE OpenAPI endpoint registry."""

ENDPOINTS: dict[str, str] = {
    "listed_companies": "opendata/t187ap03_L",
    "daily_prices": "exchangeReport/STOCK_DAY_ALL",
    "valuation": "exchangeReport/BWIBBU_ALL",
    "trading_calendar": "exchangeReport/TWTB4U",
    "valuation_daily": "exchangeReport/BWIBBU_d",
    # Official source contract endpoints inspected from the live Swagger file.
    "official_daily_snapshot": "exchangeReport/STOCK_DAY_ALL",
    "official_listed_companies": "opendata/t187ap03_L",
    "official_holiday_schedule": "holidaySchedule/holidaySchedule",
}

FUNDAMENTAL_ENDPOINTS: dict[str, str] = {
    "mops_income": "https://mopsov.twse.com.tw/mops/web/ajax_t163sb04",
    "mops_balance": "https://mopsov.twse.com.tw/mops/web/ajax_t163sb05",
    "publication_dates": "https://doc.twse.com.tw/server-java/t57sb01",
    "monthly_revenue": "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{year}_{month}_0.html",
    "valuation_daily": "https://www.twse.com.tw/exchangeReport/BWIBBU_d",
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


def get_fundamental_endpoint(key: str, **params: object) -> str:
    """Return a configured fundamental-source URL."""

    try:
        return FUNDAMENTAL_ENDPOINTS[key].format(**params)
    except KeyError as exc:
        available = ", ".join(sorted(FUNDAMENTAL_ENDPOINTS))
        raise KeyError(
            f"Unknown fundamental endpoint key: {key}. Available: {available}"
        ) from exc

# Data Quality Summary

Generated at: 2026-06-16T12:44:49.571646+00:00

Configured date range: 2018-01-01 to 2025-12-31
Actual OHLCV date range: 2018-01-02 to 2025-12-30
OHLCV ticker subset size: 100
universe_total_tickers: 1090
ohlcv_requested_tickers: 100
ohlcv_successful_tickers: 100
ohlcv_failed_tickers: 0
ohlcv_coverage_ratio: 0.0917
configured_ticker_limit: 100
actual_ohlcv_ticker_count: 100
failed_yfinance_tickers: None
Universe rows: 1090
Universe total count: 1090
Valuation rows: 1078
OHLCV rows: 193534
Ticker count: 1090
OHLCV date range: 2018-01-02 to 2025-12-30

## Sources And Limitations

- OHLCV source: yfinance fallback
- Valuation source: TWSE latest snapshot valuation endpoint
- market source: TWSE listed-company universe; if the source field is missing, normalization defaults market to TWSE.
- valuation.date is empty because the current TWSE valuation snapshot endpoint does not return historical valuation dates.
- data source limitations: OHLCV is a bounded yfinance fallback subset, while valuation data is latest snapshot data rather than point-in-time history.
- survivorship bias warning: the current universe is a present-day listed universe and can bias historical research if used without a dated membership source.

## Missing Ratios

### universe
- ticker: 0.0000
- company_name: 0.0000
- industry: 0.0000
- market: 0.0000
- listed_date: 0.0000

### valuation
- date: 1.0000
- ticker: 0.0000
- pe: 0.2291
- pb: 0.0000
- dividend_yield: 0.2004

### ohlcv
- date: 0.0000
- ticker: 0.0000
- open: 0.0000
- high: 0.0000
- low: 0.0000
- close: 0.0000
- volume: 0.0000

## Failed yfinance tickers

None

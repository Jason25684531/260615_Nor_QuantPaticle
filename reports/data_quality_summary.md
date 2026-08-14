# Data Quality Summary

Generated at: 2026-08-11T03:40:50.756135+00:00

Configured date range: 2018-01-01 to 2025-12-31
Actual OHLCV date range: 2018-01-02 to 2025-12-31
OHLCV ticker subset size: 100
universe_total_tickers: 1094
ohlcv_requested_tickers: 100
ohlcv_successful_tickers: 100
ohlcv_failed_tickers: 0
ohlcv_coverage_ratio: 0.0914
configured_ticker_limit: 100
actual_ohlcv_ticker_count: 100
failed_yfinance_tickers: None
Universe rows: 1094
Universe total count: 1094
Valuation rows: 1083
OHLCV rows: 193634
Ticker count: 1094
OHLCV date range: 2018-01-02 to 2025-12-31

## Sources And Limitations

- live_pipeline_status: PARTIAL
- OHLCV source: yfinance fallback
- price_adjustment: auto_adjusted
- volume_basis: reported_shares
- Valuation source: TWSE latest snapshot valuation endpoint
- market source: TWSE listed-company universe; if the source field is missing, normalization defaults market to TWSE.
- valuation.date is empty because the current TWSE valuation snapshot endpoint does not return historical valuation dates.
- data source limitations: OHLCV is a bounded yfinance fallback subset, while valuation data is latest snapshot data rather than point-in-time history.
- survivorship bias warning: the current universe is a present-day listed universe and can bias historical research if used without a dated membership source.

## Research Universe Quality

- membership: current_listed_only
- membership_missing_listed_date_count: 0
- survivorship_disclosure: current_listed_only is not point-in-time.
- universe_scope: PARTIAL
- liquidity_rule: trailing 20-observation median adjusted close x reported volume > 5e+07
- liquidity_measure_source: proxy_close_times_volume
- minimum_universe_coverage: 0.9
- coverage_date_rows: 1943
- latest_total_listing_eligible: 1070
- latest_ohlcv_available: 100
- latest_liquidity_pass: 13
- latest_analysis_ready: 13
- latest_analysis_ready_ratio: 0.0121
- mean_analysis_ready_ratio: 0.0166
- in_sample: {'start': '2018-01-02', 'end': '2022-12-31'}
- out_of_sample: {'start': '2023-01-01', 'end': '2025-12-31'}

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
- pe: 0.2170
- pb: 0.0000
- dividend_yield: 0.2170

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

## Fundamental Coverage

- report: `reports/fundamental_coverage_report.md`
- live_fundamental_pipeline_status: PASS
- artifact policy: `run_fundamental_pipeline.py` preserves existing artifacts
  when a live backfill is incomplete.



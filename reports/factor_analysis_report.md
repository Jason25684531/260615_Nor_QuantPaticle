# Factor Analysis Report

## Run Metadata

- generated_at: 2026-08-11T03:41:31.042219+00:00
- config_path: config\strategy.yaml
- pipeline_name: run_factor_analysis.py

## Scope

- historical factors analyzed: momentum_60d, low_volatility_20d, volume_ratio_5d_60d, historical_price_volume
- snapshot factors excluded:
  - pb_inverse: snapshot_only_not_historical_ready
  - pe_inverse: snapshot_only_not_historical_ready
  - dividend_yield: snapshot_only_not_historical_ready
  - latest_snapshot_mixed: snapshot_only_not_historical_ready
- reason for exclusion: valuation snapshot factors are not point-in-time historical series.

## Input Artifacts

- close_matrix: D:\01_Project\260615_Nor_QuantPaticle\data\processed\close_matrix.parquet
- factors_price_volume: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factors_price_volume.parquet
- factors_composite: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factors_composite.parquet
- close_matrix_shape: (1943, 100)
- close_matrix_date_range: 2018-01-02 to 2025-12-31
- ohlcv_subset_ticker_count: 100

## Forward Return Setup

- horizons: 1D, 5D, 10D, 20D
- target return definition: forward return from date T close to date T+h close.
- no-lookahead note: factor values are evaluated at date T without shifting the factor forward.

## IC / IR Summary

- historical_price_volume | 1D | IC mean=0.0104, IC std=0.1662, IR=0.0627, valid dates=1922, avg assets=99.65
- historical_price_volume | 5D | IC mean=0.0256, IC std=0.1769, IR=0.1449, valid dates=1918, avg assets=99.65
- historical_price_volume | 10D | IC mean=0.0364, IC std=0.1722, IR=0.2115, valid dates=1913, avg assets=99.65
- historical_price_volume | 20D | IC mean=0.0467, IC std=0.1737, IR=0.2689, valid dates=1903, avg assets=99.65
- low_volatility_20d | 1D | IC mean=-0.0429, IC std=0.2051, IR=-0.2094, valid dates=1922, avg assets=99.65
- low_volatility_20d | 5D | IC mean=-0.0667, IC std=0.2040, IR=-0.3269, valid dates=1918, avg assets=99.65
- low_volatility_20d | 10D | IC mean=-0.0783, IC std=0.1930, IR=-0.4055, valid dates=1913, avg assets=99.65
- low_volatility_20d | 20D | IC mean=-0.0928, IC std=0.1826, IR=-0.5080, valid dates=1903, avg assets=99.65
- momentum_60d | 1D | IC mean=-0.0104, IC std=0.1616, IR=-0.0644, valid dates=1882, avg assets=99.65
- momentum_60d | 5D | IC mean=-0.0043, IC std=0.1731, IR=-0.0248, valid dates=1878, avg assets=99.65
- momentum_60d | 10D | IC mean=0.0035, IC std=0.1747, IR=0.0198, valid dates=1873, avg assets=99.64
- momentum_60d | 20D | IC mean=0.0148, IC std=0.1704, IR=0.0867, valid dates=1863, avg assets=99.64
- volume_ratio_5d_60d | 1D | IC mean=-0.0089, IC std=0.1324, IR=-0.0674, valid dates=1883, avg assets=99.54
- volume_ratio_5d_60d | 5D | IC mean=-0.0107, IC std=0.1339, IR=-0.0800, valid dates=1879, avg assets=99.54
- volume_ratio_5d_60d | 10D | IC mean=-0.0087, IC std=0.1310, IR=-0.0666, valid dates=1874, avg assets=99.54
- volume_ratio_5d_60d | 20D | IC mean=-0.0168, IC std=0.1304, IR=-0.1291, valid dates=1864, avg assets=99.54

## Quantile Returns

- historical_price_volume | 1D | Q1=0.0002, Q2=0.0001, Q3=0.0003, Q4=0.0004, Q5=0.0006 | top-bottom=0.0004
- historical_price_volume | 5D | Q1=0.0010, Q2=0.0011, Q3=0.0020, Q4=0.0018, Q5=0.0027 | top-bottom=0.0017
- historical_price_volume | 10D | Q1=0.0021, Q2=0.0025, Q3=0.0043, Q4=0.0032, Q5=0.0056 | top-bottom=0.0035
- historical_price_volume | 20D | Q1=0.0038, Q2=0.0057, Q3=0.0085, Q4=0.0077, Q5=0.0100 | top-bottom=0.0061
- low_volatility_20d | 1D | Q1=0.0002, Q2=0.0004, Q3=0.0003, Q4=0.0003, Q5=0.0004 | top-bottom=0.0002
- low_volatility_20d | 5D | Q1=0.0013, Q2=0.0018, Q3=0.0017, Q4=0.0017, Q5=0.0020 | top-bottom=0.0007
- low_volatility_20d | 10D | Q1=0.0029, Q2=0.0033, Q3=0.0040, Q4=0.0033, Q5=0.0043 | top-bottom=0.0014
- low_volatility_20d | 20D | Q1=0.0067, Q2=0.0065, Q3=0.0074, Q4=0.0067, Q5=0.0084 | top-bottom=0.0016
- momentum_60d | 1D | Q1=0.0003, Q2=0.0002, Q3=0.0003, Q4=0.0003, Q5=0.0005 | top-bottom=0.0002
- momentum_60d | 5D | Q1=0.0011, Q2=0.0011, Q3=0.0017, Q4=0.0023, Q5=0.0022 | top-bottom=0.0012
- momentum_60d | 10D | Q1=0.0015, Q2=0.0024, Q3=0.0039, Q4=0.0046, Q5=0.0045 | top-bottom=0.0030
- momentum_60d | 20D | Q1=0.0019, Q2=0.0052, Q3=0.0080, Q4=0.0094, Q5=0.0098 | top-bottom=0.0079
- volume_ratio_5d_60d | 1D | Q1=0.0002, Q2=0.0003, Q3=0.0003, Q4=0.0005, Q5=0.0004 | top-bottom=0.0002
- volume_ratio_5d_60d | 5D | Q1=0.0015, Q2=0.0017, Q3=0.0014, Q4=0.0018, Q5=0.0020 | top-bottom=0.0005
- volume_ratio_5d_60d | 10D | Q1=0.0029, Q2=0.0036, Q3=0.0025, Q4=0.0031, Q5=0.0046 | top-bottom=0.0016
- volume_ratio_5d_60d | 20D | Q1=0.0065, Q2=0.0073, Q3=0.0057, Q4=0.0070, Q5=0.0077 | top-bottom=0.0012

## Factor Turnover

- historical_price_volume | avg top-quantile turnover=0.1539 | dates=1923
- low_volatility_20d | avg top-quantile turnover=0.0572 | dates=1923
- momentum_60d | avg top-quantile turnover=0.0957 | dates=1883
- volume_ratio_5d_60d | avg top-quantile turnover=0.1707 | dates=1884

## Monotonicity Check

- historical_price_volume | 1D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- historical_price_volume | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- historical_price_volume | 10D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- historical_price_volume | 20D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 10D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 20D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- momentum_60d | 1D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- momentum_60d | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- momentum_60d | 10D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- momentum_60d | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- volume_ratio_5d_60d | 1D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- volume_ratio_5d_60d | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- volume_ratio_5d_60d | 10D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- volume_ratio_5d_60d | 20D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering

## Interpretation

- Factors with consistently positive IC, positive top-bottom spreads, and monotonic quantile structure appear more promising.
- Weak or unstable signals should be treated as research findings, not pipeline failures.
- Current OHLCV coverage is limited to 100 tickers, so results are subset-level rather than full-market conclusions.

## Limitations

- only 100 tickers are covered by the current OHLCV subset
- yfinance fallback data may differ from official adjusted TWSE data
- valuation factors are excluded because they remain snapshot-only
- no transaction cost model, portfolio construction, or backtest is included
- not investment advice

## Generated Artifacts

- factor_forward_returns: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_forward_returns.parquet
- factor_ic_summary: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_ic_summary.parquet
- factor_quantile_returns: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_quantile_returns.parquet
- factor_turnover: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_turnover.parquet
- factor_monotonicity: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_monotonicity.parquet
- factor_analysis_report: D:\01_Project\260615_Nor_QuantPaticle\reports\factor_analysis_report.md
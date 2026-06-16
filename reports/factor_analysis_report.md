# Factor Analysis Report

## Run Metadata

- generated_at: 2026-06-16T12:46:03.074748+00:00
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
- close_matrix_shape: (1942, 100)
- close_matrix_date_range: 2018-01-02 to 2025-12-30
- ohlcv_subset_ticker_count: 100

## Forward Return Setup

- horizons: 1D, 5D, 10D, 20D
- target return definition: forward return from date T close to date T+h close.
- no-lookahead note: factor values are evaluated at date T without shifting the factor forward.

## IC / IR Summary

- historical_price_volume | 1D | IC mean=0.0085, IC std=0.1645, IR=0.0516, valid dates=1921, avg assets=99.65
- historical_price_volume | 5D | IC mean=0.0179, IC std=0.1751, IR=0.1021, valid dates=1917, avg assets=99.65
- historical_price_volume | 10D | IC mean=0.0250, IC std=0.1705, IR=0.1469, valid dates=1912, avg assets=99.65
- historical_price_volume | 20D | IC mean=0.0332, IC std=0.1727, IR=0.1921, valid dates=1902, avg assets=99.65
- low_volatility_20d | 1D | IC mean=-0.0421, IC std=0.2046, IR=-0.2055, valid dates=1921, avg assets=99.65
- low_volatility_20d | 5D | IC mean=-0.0617, IC std=0.2029, IR=-0.3040, valid dates=1917, avg assets=99.65
- low_volatility_20d | 10D | IC mean=-0.0695, IC std=0.1911, IR=-0.3638, valid dates=1912, avg assets=99.65
- low_volatility_20d | 20D | IC mean=-0.0788, IC std=0.1810, IR=-0.4353, valid dates=1902, avg assets=99.65
- momentum_60d | 1D | IC mean=-0.0103, IC std=0.1606, IR=-0.0638, valid dates=1881, avg assets=99.65
- momentum_60d | 5D | IC mean=-0.0063, IC std=0.1710, IR=-0.0370, valid dates=1877, avg assets=99.65
- momentum_60d | 10D | IC mean=0.0002, IC std=0.1722, IR=0.0010, valid dates=1872, avg assets=99.64
- momentum_60d | 20D | IC mean=0.0095, IC std=0.1658, IR=0.0571, valid dates=1862, avg assets=99.64
- volume_ratio_5d_60d | 1D | IC mean=-0.0120, IC std=0.1321, IR=-0.0912, valid dates=1882, avg assets=99.54
- volume_ratio_5d_60d | 5D | IC mean=-0.0177, IC std=0.1335, IR=-0.1323, valid dates=1878, avg assets=99.54
- volume_ratio_5d_60d | 10D | IC mean=-0.0179, IC std=0.1299, IR=-0.1374, valid dates=1873, avg assets=99.54
- volume_ratio_5d_60d | 20D | IC mean=-0.0231, IC std=0.1289, IR=-0.1796, valid dates=1863, avg assets=99.54

## Quantile Returns

- historical_price_volume | 1D | Q1=0.0002, Q2=-0.0000, Q3=0.0003, Q4=0.0002, Q5=0.0004 | top-bottom=0.0002
- historical_price_volume | 5D | Q1=0.0007, Q2=0.0005, Q3=0.0016, Q4=0.0011, Q5=0.0016 | top-bottom=0.0009
- historical_price_volume | 10D | Q1=0.0014, Q2=0.0013, Q3=0.0033, Q4=0.0019, Q5=0.0035 | top-bottom=0.0021
- historical_price_volume | 20D | Q1=0.0021, Q2=0.0033, Q3=0.0062, Q4=0.0054, Q5=0.0062 | top-bottom=0.0041
- low_volatility_20d | 1D | Q1=0.0001, Q2=0.0003, Q3=0.0002, Q4=0.0002, Q5=0.0002 | top-bottom=0.0002
- low_volatility_20d | 5D | Q1=0.0010, Q2=0.0014, Q3=0.0011, Q4=0.0009, Q5=0.0011 | top-bottom=0.0002
- low_volatility_20d | 10D | Q1=0.0022, Q2=0.0023, Q3=0.0026, Q4=0.0018, Q5=0.0025 | top-bottom=0.0003
- low_volatility_20d | 20D | Q1=0.0052, Q2=0.0046, Q3=0.0048, Q4=0.0040, Q5=0.0047 | top-bottom=-0.0005
- momentum_60d | 1D | Q1=0.0002, Q2=0.0001, Q3=0.0002, Q4=0.0001, Q5=0.0004 | top-bottom=0.0002
- momentum_60d | 5D | Q1=0.0006, Q2=0.0005, Q3=0.0010, Q4=0.0014, Q5=0.0018 | top-bottom=0.0011
- momentum_60d | 10D | Q1=0.0005, Q2=0.0011, Q3=0.0026, Q4=0.0028, Q5=0.0036 | top-bottom=0.0031
- momentum_60d | 20D | Q1=-0.0002, Q2=0.0024, Q3=0.0055, Q4=0.0062, Q5=0.0078 | top-bottom=0.0080
- volume_ratio_5d_60d | 1D | Q1=0.0001, Q2=0.0002, Q3=0.0002, Q4=0.0003, Q5=0.0001 | top-bottom=-0.0000
- volume_ratio_5d_60d | 5D | Q1=0.0011, Q2=0.0013, Q3=0.0009, Q4=0.0010, Q5=0.0009 | top-bottom=-0.0002
- volume_ratio_5d_60d | 10D | Q1=0.0022, Q2=0.0026, Q3=0.0014, Q4=0.0016, Q5=0.0026 | top-bottom=0.0005
- volume_ratio_5d_60d | 20D | Q1=0.0047, Q2=0.0049, Q3=0.0031, Q4=0.0041, Q5=0.0047 | top-bottom=0.0000

## Factor Turnover

- historical_price_volume | avg top-quantile turnover=0.1559 | dates=1922
- low_volatility_20d | avg top-quantile turnover=0.0587 | dates=1922
- momentum_60d | avg top-quantile turnover=0.0970 | dates=1882
- volume_ratio_5d_60d | avg top-quantile turnover=0.1706 | dates=1883

## Monotonicity Check

- historical_price_volume | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- historical_price_volume | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- historical_price_volume | 10D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- historical_price_volume | 20D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 10D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 20D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- momentum_60d | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- momentum_60d | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- momentum_60d | 10D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- momentum_60d | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- volume_ratio_5d_60d | 1D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- volume_ratio_5d_60d | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
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
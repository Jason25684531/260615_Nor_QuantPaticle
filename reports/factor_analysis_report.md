# Factor Analysis Report

## Run Metadata

- generated_at: 2026-08-14T09:52:16.549163+00:00
- config_path: config\strategy.yaml
- pipeline_name: run_factor_analysis.py

## Scope

- historical factors analyzed: momentum_60d, low_volatility_20d, volume_ratio_5d_60d, historical_price_volume, rsi_z, momentum_40d, risk_adjusted_momentum, obv_z90, natr_14, revenue_yoy, eps, roe, pe, pb, dividend_yield
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
- close_matrix_shape: (1943, 1092)
- close_matrix_date_range: 2018-01-02 to 2025-12-31
- ohlcv_subset_ticker_count: 1092

## Forward Return Setup

- horizons: 1D, 5D, 10D, 20D
- target return definition: forward return from date T close to date T+h close.
- no-lookahead note: factor values are evaluated at date T without shifting the factor forward.

## IC / IR Summary

- dividend_yield | 1D | IC mean=0.0336, IC std=0.3086, IR=0.1089, valid dates=1921, avg assets=15.96
- dividend_yield | 5D | IC mean=0.0453, IC std=0.3092, IR=0.1464, valid dates=1918, avg assets=15.96
- dividend_yield | 10D | IC mean=0.0604, IC std=0.3021, IR=0.2000, valid dates=1913, avg assets=15.97
- dividend_yield | 20D | IC mean=0.0797, IC std=0.3134, IR=0.2543, valid dates=1903, avg assets=15.97
- eps | 1D | IC mean=0.0281, IC std=0.5489, IR=0.0511, valid dates=1602, avg assets=6.38
- eps | 5D | IC mean=0.0624, IC std=0.5468, IR=0.1140, valid dates=1600, avg assets=6.37
- eps | 10D | IC mean=0.1173, IC std=0.5363, IR=0.2187, valid dates=1595, avg assets=6.36
- eps | 20D | IC mean=0.1768, IC std=0.5375, IR=0.3290, valid dates=1585, avg assets=6.34
- historical_price_volume | 1D | IC mean=0.0079, IC std=0.1539, IR=0.0515, valid dates=1922, avg assets=994.58
- historical_price_volume | 5D | IC mean=0.0179, IC std=0.1612, IR=0.1110, valid dates=1918, avg assets=994.18
- historical_price_volume | 10D | IC mean=0.0274, IC std=0.1588, IR=0.1727, valid dates=1913, avg assets=993.89
- historical_price_volume | 20D | IC mean=0.0380, IC std=0.1599, IR=0.2376, valid dates=1903, avg assets=993.41
- low_volatility_20d | 1D | IC mean=-0.0444, IC std=0.2172, IR=-0.2044, valid dates=1922, avg assets=950.31
- low_volatility_20d | 5D | IC mean=-0.0558, IC std=0.2124, IR=-0.2629, valid dates=1918, avg assets=949.84
- low_volatility_20d | 10D | IC mean=-0.0626, IC std=0.2057, IR=-0.3044, valid dates=1913, avg assets=949.39
- low_volatility_20d | 20D | IC mean=-0.0739, IC std=0.2057, IR=-0.3594, valid dates=1903, avg assets=948.57
- momentum_40d | 1D | IC mean=-0.0172, IC std=0.1321, IR=-0.1301, valid dates=1902, avg assets=993.92
- momentum_40d | 5D | IC mean=-0.0175, IC std=0.1353, IR=-0.1295, valid dates=1898, avg assets=993.52
- momentum_40d | 10D | IC mean=-0.0151, IC std=0.1326, IR=-0.1139, valid dates=1893, avg assets=993.25
- momentum_40d | 20D | IC mean=-0.0116, IC std=0.1356, IR=-0.0853, valid dates=1883, avg assets=992.74
- momentum_60d | 1D | IC mean=-0.0139, IC std=0.1302, IR=-0.1067, valid dates=1882, avg assets=993.10
- momentum_60d | 5D | IC mean=-0.0130, IC std=0.1351, IR=-0.0965, valid dates=1878, avg assets=992.71
- momentum_60d | 10D | IC mean=-0.0097, IC std=0.1330, IR=-0.0732, valid dates=1873, avg assets=992.42
- momentum_60d | 20D | IC mean=-0.0047, IC std=0.1307, IR=-0.0357, valid dates=1863, avg assets=991.95
- natr_14 | 1D | IC mean=0.0473, IC std=0.2223, IR=0.2128, valid dates=1928, avg assets=998.47
- natr_14 | 5D | IC mean=0.0592, IC std=0.2175, IR=0.2723, valid dates=1924, avg assets=998.07
- natr_14 | 10D | IC mean=0.0659, IC std=0.2118, IR=0.3112, valid dates=1919, avg assets=997.80
- natr_14 | 20D | IC mean=0.0777, IC std=0.2120, IR=0.3663, valid dates=1909, avg assets=997.33
- obv_z90 | 1D | IC mean=-0.0141, IC std=0.0893, IR=-0.1583, valid dates=1853, avg assets=857.99
- obv_z90 | 5D | IC mean=-0.0137, IC std=0.0819, IR=-0.1671, valid dates=1849, avg assets=857.41
- obv_z90 | 10D | IC mean=-0.0087, IC std=0.0810, IR=-0.1070, valid dates=1844, avg assets=856.75
- obv_z90 | 20D | IC mean=-0.0057, IC std=0.0812, IR=-0.0707, valid dates=1834, avg assets=855.47
- pb | 1D | IC mean=-0.0003, IC std=0.3160, IR=-0.0009, valid dates=1921, avg assets=15.96
- pb | 5D | IC mean=-0.0013, IC std=0.3176, IR=-0.0040, valid dates=1918, avg assets=15.96
- pb | 10D | IC mean=-0.0047, IC std=0.3178, IR=-0.0146, valid dates=1913, avg assets=15.97
- pb | 20D | IC mean=-0.0145, IC std=0.3183, IR=-0.0456, valid dates=1903, avg assets=15.97
- pe | 1D | IC mean=0.0206, IC std=0.3208, IR=0.0643, valid dates=1921, avg assets=15.93
- pe | 5D | IC mean=0.0293, IC std=0.3175, IR=0.0922, valid dates=1918, avg assets=15.94
- pe | 10D | IC mean=0.0379, IC std=0.3164, IR=0.1196, valid dates=1913, avg assets=15.94
- pe | 20D | IC mean=0.0529, IC std=0.3219, IR=0.1644, valid dates=1903, avg assets=15.95
- revenue_yoy | 1D | IC mean=0.0124, IC std=0.3157, IR=0.0394, valid dates=1689, avg assets=16.39
- revenue_yoy | 5D | IC mean=0.0187, IC std=0.3083, IR=0.0607, valid dates=1686, avg assets=16.40
- revenue_yoy | 10D | IC mean=0.0290, IC std=0.3119, IR=0.0931, valid dates=1681, avg assets=16.41
- revenue_yoy | 20D | IC mean=0.0379, IC std=0.3186, IR=0.1191, valid dates=1671, avg assets=16.42
- risk_adjusted_momentum | 1D | IC mean=-0.0126, IC std=0.1051, IR=-0.1196, valid dates=1902, avg assets=993.16
- risk_adjusted_momentum | 5D | IC mean=-0.0076, IC std=0.1087, IR=-0.0695, valid dates=1898, avg assets=992.76
- risk_adjusted_momentum | 10D | IC mean=-0.0013, IC std=0.1075, IR=-0.0122, valid dates=1893, avg assets=992.49
- risk_adjusted_momentum | 20D | IC mean=0.0053, IC std=0.1090, IR=0.0484, valid dates=1883, avg assets=991.98
- roe | 1D | IC mean=0.0226, IC std=0.5482, IR=0.0412, valid dates=1623, avg assets=6.33
- roe | 5D | IC mean=0.0538, IC std=0.5504, IR=0.0977, valid dates=1622, avg assets=6.31
- roe | 10D | IC mean=0.1009, IC std=0.5383, IR=0.1874, valid dates=1616, avg assets=6.31
- roe | 20D | IC mean=0.1514, IC std=0.5324, IR=0.2843, valid dates=1606, avg assets=6.28
- rsi_z | 1D | IC mean=-0.0282, IC std=0.1013, IR=-0.2785, valid dates=1825, avg assets=994.45
- rsi_z | 5D | IC mean=-0.0239, IC std=0.1001, IR=-0.2386, valid dates=1821, avg assets=994.07
- rsi_z | 10D | IC mean=-0.0145, IC std=0.0963, IR=-0.1502, valid dates=1816, avg assets=993.80
- rsi_z | 20D | IC mean=-0.0116, IC std=0.0976, IR=-0.1188, valid dates=1806, avg assets=993.34
- volume_ratio_5d_60d | 1D | IC mean=-0.0109, IC std=0.0842, IR=-0.1291, valid dates=1883, avg assets=888.92
- volume_ratio_5d_60d | 5D | IC mean=-0.0055, IC std=0.0820, IR=-0.0668, valid dates=1879, avg assets=888.37
- volume_ratio_5d_60d | 10D | IC mean=0.0015, IC std=0.0831, IR=0.0177, valid dates=1874, avg assets=887.77
- volume_ratio_5d_60d | 20D | IC mean=0.0018, IC std=0.0805, IR=0.0218, valid dates=1864, avg assets=886.65

## Quantile Returns

- dividend_yield | 1D | Q1=-0.0001, Q2=0.0002, Q3=0.0000, Q4=0.0003, Q5=-0.0002 | top-bottom=-0.0001
- dividend_yield | 5D | Q1=-0.0008, Q2=0.0009, Q3=0.0008, Q4=0.0011, Q5=-0.0004 | top-bottom=0.0004
- dividend_yield | 10D | Q1=-0.0014, Q2=0.0023, Q3=0.0024, Q4=0.0015, Q5=0.0004 | top-bottom=0.0018
- dividend_yield | 20D | Q1=-0.0012, Q2=0.0067, Q3=0.0053, Q4=0.0027, Q5=0.0033 | top-bottom=0.0046
- eps | 1D | Q1=0.0002, Q2=-0.0005, Q3=-0.0007, Q4=0.0005, Q5=0.0003 | top-bottom=0.0001
- eps | 5D | Q1=0.0014, Q2=-0.0033, Q3=-0.0036, Q4=0.0033, Q5=0.0017 | top-bottom=0.0003
- eps | 10D | Q1=0.0017, Q2=-0.0068, Q3=-0.0055, Q4=0.0084, Q5=0.0037 | top-bottom=0.0020
- eps | 20D | Q1=0.0061, Q2=-0.0169, Q3=-0.0066, Q4=0.0159, Q5=0.0080 | top-bottom=0.0019
- historical_price_volume | 1D | Q1=0.0004, Q2=0.0004, Q3=0.0004, Q4=0.0006, Q5=0.0013 | top-bottom=0.0009
- historical_price_volume | 5D | Q1=0.0021, Q2=0.0023, Q3=0.0028, Q4=0.0033, Q5=0.0048 | top-bottom=0.0027
- historical_price_volume | 10D | Q1=0.0042, Q2=0.0048, Q3=0.0059, Q4=0.0070, Q5=0.0093 | top-bottom=0.0050
- historical_price_volume | 20D | Q1=0.0088, Q2=0.0103, Q3=0.0123, Q4=0.0140, Q5=0.0171 | top-bottom=0.0083
- low_volatility_20d | 1D | Q1=0.0007, Q2=0.0006, Q3=0.0006, Q4=0.0008, Q5=0.0005 | top-bottom=-0.0002
- low_volatility_20d | 5D | Q1=0.0040, Q2=0.0030, Q3=0.0028, Q4=0.0028, Q5=0.0027 | top-bottom=-0.0013
- low_volatility_20d | 10D | Q1=0.0079, Q2=0.0062, Q3=0.0057, Q4=0.0056, Q5=0.0056 | top-bottom=-0.0023
- low_volatility_20d | 20D | Q1=0.0151, Q2=0.0129, Q3=0.0115, Q4=0.0111, Q5=0.0118 | top-bottom=-0.0034
- momentum_40d | 1D | Q1=0.0005, Q2=0.0004, Q3=0.0007, Q4=0.0006, Q5=0.0009 | top-bottom=0.0004
- momentum_40d | 5D | Q1=0.0025, Q2=0.0023, Q3=0.0030, Q4=0.0030, Q5=0.0047 | top-bottom=0.0021
- momentum_40d | 10D | Q1=0.0046, Q2=0.0049, Q3=0.0055, Q4=0.0064, Q5=0.0092 | top-bottom=0.0047
- momentum_40d | 20D | Q1=0.0097, Q2=0.0104, Q3=0.0115, Q4=0.0128, Q5=0.0175 | top-bottom=0.0078
- momentum_60d | 1D | Q1=0.0005, Q2=0.0004, Q3=0.0008, Q4=0.0005, Q5=0.0010 | top-bottom=0.0004
- momentum_60d | 5D | Q1=0.0026, Q2=0.0021, Q3=0.0029, Q4=0.0033, Q5=0.0045 | top-bottom=0.0019
- momentum_60d | 10D | Q1=0.0046, Q2=0.0045, Q3=0.0057, Q4=0.0069, Q5=0.0089 | top-bottom=0.0043
- momentum_60d | 20D | Q1=0.0089, Q2=0.0094, Q3=0.0120, Q4=0.0146, Q5=0.0174 | top-bottom=0.0085
- natr_14 | 1D | Q1=0.0007, Q2=0.0006, Q3=0.0005, Q4=0.0005, Q5=0.0008 | top-bottom=0.0001
- natr_14 | 5D | Q1=0.0039, Q2=0.0030, Q3=0.0029, Q4=0.0028, Q5=0.0026 | top-bottom=-0.0013
- natr_14 | 10D | Q1=0.0077, Q2=0.0061, Q3=0.0058, Q4=0.0054, Q5=0.0054 | top-bottom=-0.0023
- natr_14 | 20D | Q1=0.0149, Q2=0.0129, Q3=0.0118, Q4=0.0109, Q5=0.0115 | top-bottom=-0.0034
- obv_z90 | 1D | Q1=0.0003, Q2=0.0008, Q3=0.0005, Q4=0.0006, Q5=0.0010 | top-bottom=0.0007
- obv_z90 | 5D | Q1=0.0023, Q2=0.0023, Q3=0.0030, Q4=0.0033, Q5=0.0045 | top-bottom=0.0022
- obv_z90 | 10D | Q1=0.0038, Q2=0.0046, Q3=0.0065, Q4=0.0069, Q5=0.0088 | top-bottom=0.0050
- obv_z90 | 20D | Q1=0.0084, Q2=0.0095, Q3=0.0128, Q4=0.0141, Q5=0.0162 | top-bottom=0.0078
- pb | 1D | Q1=0.0001, Q2=0.0001, Q3=-0.0003, Q4=0.0002, Q5=0.0002 | top-bottom=0.0001
- pb | 5D | Q1=0.0004, Q2=0.0007, Q3=-0.0010, Q4=0.0008, Q5=0.0008 | top-bottom=0.0004
- pb | 10D | Q1=0.0012, Q2=0.0013, Q3=-0.0013, Q4=0.0012, Q5=0.0027 | top-bottom=0.0014
- pb | 20D | Q1=0.0045, Q2=0.0026, Q3=-0.0019, Q4=0.0046, Q5=0.0068 | top-bottom=0.0023
- pe | 1D | Q1=0.0001, Q2=-0.0005, Q3=0.0002, Q4=0.0007, Q5=-0.0003 | top-bottom=-0.0004
- pe | 5D | Q1=0.0008, Q2=-0.0018, Q3=0.0006, Q4=0.0035, Q5=-0.0017 | top-bottom=-0.0026
- pe | 10D | Q1=0.0016, Q2=-0.0029, Q3=0.0016, Q4=0.0066, Q5=-0.0021 | top-bottom=-0.0036
- pe | 20D | Q1=0.0038, Q2=-0.0024, Q3=0.0047, Q4=0.0119, Q5=-0.0013 | top-bottom=-0.0051
- revenue_yoy | 1D | Q1=-0.0001, Q2=-0.0002, Q3=0.0003, Q4=0.0001, Q5=0.0004 | top-bottom=0.0005
- revenue_yoy | 5D | Q1=-0.0008, Q2=-0.0006, Q3=0.0020, Q4=0.0013, Q5=0.0010 | top-bottom=0.0018
- revenue_yoy | 10D | Q1=-0.0027, Q2=-0.0004, Q3=0.0042, Q4=0.0032, Q5=0.0022 | top-bottom=0.0049
- revenue_yoy | 20D | Q1=-0.0038, Q2=0.0008, Q3=0.0086, Q4=0.0069, Q5=0.0071 | top-bottom=0.0109
- risk_adjusted_momentum | 1D | Q1=0.0005, Q2=0.0004, Q3=0.0007, Q4=0.0005, Q5=0.0010 | top-bottom=0.0005
- risk_adjusted_momentum | 5D | Q1=0.0022, Q2=0.0021, Q3=0.0030, Q4=0.0031, Q5=0.0051 | top-bottom=0.0029
- risk_adjusted_momentum | 10D | Q1=0.0040, Q2=0.0043, Q3=0.0056, Q4=0.0067, Q5=0.0100 | top-bottom=0.0060
- risk_adjusted_momentum | 20D | Q1=0.0084, Q2=0.0094, Q3=0.0120, Q4=0.0130, Q5=0.0189 | top-bottom=0.0105
- roe | 1D | Q1=-0.0002, Q2=-0.0003, Q3=-0.0001, Q4=0.0002, Q5=0.0004 | top-bottom=0.0006
- roe | 5D | Q1=-0.0007, Q2=-0.0025, Q3=0.0004, Q4=0.0001, Q5=0.0036 | top-bottom=0.0043
- roe | 10D | Q1=-0.0025, Q2=-0.0034, Q3=0.0015, Q4=0.0002, Q5=0.0088 | top-bottom=0.0113
- roe | 20D | Q1=-0.0050, Q2=-0.0029, Q3=0.0032, Q4=0.0020, Q5=0.0168 | top-bottom=0.0218
- rsi_z | 1D | Q1=0.0005, Q2=0.0006, Q3=0.0005, Q4=0.0008, Q5=0.0008 | top-bottom=0.0002
- rsi_z | 5D | Q1=0.0022, Q2=0.0032, Q3=0.0031, Q4=0.0037, Q5=0.0038 | top-bottom=0.0016
- rsi_z | 10D | Q1=0.0042, Q2=0.0062, Q3=0.0062, Q4=0.0069, Q5=0.0080 | top-bottom=0.0038
- rsi_z | 20D | Q1=0.0098, Q2=0.0123, Q3=0.0130, Q4=0.0135, Q5=0.0144 | top-bottom=0.0046
- volume_ratio_5d_60d | 1D | Q1=0.0002, Q2=0.0004, Q3=0.0006, Q4=0.0010, Q5=0.0009 | top-bottom=0.0007
- volume_ratio_5d_60d | 5D | Q1=0.0012, Q2=0.0025, Q3=0.0031, Q4=0.0035, Q5=0.0045 | top-bottom=0.0033
- volume_ratio_5d_60d | 10D | Q1=0.0031, Q2=0.0050, Q3=0.0062, Q4=0.0070, Q5=0.0085 | top-bottom=0.0054
- volume_ratio_5d_60d | 20D | Q1=0.0081, Q2=0.0113, Q3=0.0127, Q4=0.0135, Q5=0.0152 | top-bottom=0.0071

## Factor Turnover

- dividend_yield | avg top-quantile turnover=0.0516 | dates=1923
- eps | avg top-quantile turnover=0.0118 | dates=1276
- historical_price_volume | avg top-quantile turnover=0.1369 | dates=1923
- low_volatility_20d | avg top-quantile turnover=0.0418 | dates=1923
- momentum_40d | avg top-quantile turnover=0.1099 | dates=1903
- momentum_60d | avg top-quantile turnover=0.0901 | dates=1883
- natr_14 | avg top-quantile turnover=0.0439 | dates=1929
- obv_z90 | avg top-quantile turnover=0.1179 | dates=1854
- pb | avg top-quantile turnover=0.0414 | dates=1923
- pe | avg top-quantile turnover=0.0479 | dates=1923
- revenue_yoy | avg top-quantile turnover=0.0638 | dates=1691
- risk_adjusted_momentum | avg top-quantile turnover=0.1330 | dates=1903
- roe | avg top-quantile turnover=0.0220 | dates=1276
- rsi_z | avg top-quantile turnover=0.1792 | dates=1826
- volume_ratio_5d_60d | avg top-quantile turnover=0.1549 | dates=1884

## Monotonicity Check

- dividend_yield | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- dividend_yield | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- dividend_yield | 10D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- dividend_yield | 20D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- eps | 1D | monotonicity_pass=false | score=0.2500 | notes=non-monotonic quantile-return ordering
- eps | 5D | monotonicity_pass=false | score=0.2500 | notes=non-monotonic quantile-return ordering
- eps | 10D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- eps | 20D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- historical_price_volume | 1D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- historical_price_volume | 5D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- historical_price_volume | 10D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- historical_price_volume | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- low_volatility_20d | 1D | monotonicity_pass=false | score=0.2500 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 5D | monotonicity_pass=false | score=0.2500 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 10D | monotonicity_pass=false | score=0.0000 | notes=non-monotonic quantile-return ordering
- low_volatility_20d | 20D | monotonicity_pass=false | score=0.2500 | notes=non-monotonic quantile-return ordering
- momentum_40d | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- momentum_40d | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- momentum_40d | 10D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- momentum_40d | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- momentum_60d | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- momentum_60d | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- momentum_60d | 10D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- momentum_60d | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- natr_14 | 1D | monotonicity_pass=false | score=0.2500 | notes=non-monotonic quantile-return ordering
- natr_14 | 5D | monotonicity_pass=false | score=0.0000 | notes=non-monotonic quantile-return ordering
- natr_14 | 10D | monotonicity_pass=false | score=0.0000 | notes=non-monotonic quantile-return ordering
- natr_14 | 20D | monotonicity_pass=false | score=0.2500 | notes=non-monotonic quantile-return ordering
- obv_z90 | 1D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- obv_z90 | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- obv_z90 | 10D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- obv_z90 | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- pb | 1D | monotonicity_pass=false | score=0.2500 | notes=non-monotonic quantile-return ordering
- pb | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- pb | 10D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- pb | 20D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- pe | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- pe | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- pe | 10D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- pe | 20D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- revenue_yoy | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- revenue_yoy | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- revenue_yoy | 10D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- revenue_yoy | 20D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- risk_adjusted_momentum | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- risk_adjusted_momentum | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- risk_adjusted_momentum | 10D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- risk_adjusted_momentum | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- roe | 1D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- roe | 5D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- roe | 10D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- roe | 20D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- rsi_z | 1D | monotonicity_pass=false | score=0.5000 | notes=non-monotonic quantile-return ordering
- rsi_z | 5D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- rsi_z | 10D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- rsi_z | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- volume_ratio_5d_60d | 1D | monotonicity_pass=false | score=0.7500 | notes=non-monotonic quantile-return ordering
- volume_ratio_5d_60d | 5D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- volume_ratio_5d_60d | 10D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5
- volume_ratio_5d_60d | 20D | monotonicity_pass=true | score=1.0000 | notes=monotonic increasing from Q1 to Q5

## Interpretation

- Factors with consistently positive IC, positive top-bottom spreads, and monotonic quantile structure appear more promising.
- Weak or unstable signals should be treated as research findings, not pipeline failures.
- Current OHLCV coverage is limited to 1092 tickers, so results are subset-level rather than full-market conclusions.

## Limitations

- only 1092 tickers are covered by the current OHLCV subset
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
- factor_ic_daily: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_ic_daily.parquet
- factor_correlation: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_correlation.parquet
- factor_scoreboard: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_scoreboard.parquet
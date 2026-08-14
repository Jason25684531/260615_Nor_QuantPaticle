# Factor Quality Summary

Generated at: 2026-08-14T09:49:10.340980+00:00

## Input Parquet Paths

- universe: D:\01_Project\260615_Nor_QuantPaticle\data\processed\universe.parquet
- valuation: D:\01_Project\260615_Nor_QuantPaticle\data\processed\valuation.parquet
- ohlcv: D:\01_Project\260615_Nor_QuantPaticle\data\processed\ohlcv.parquet

## Artifact Shapes

- close_matrix: (1943, 1092)
- high_matrix: (1943, 1092)
- low_matrix: (1943, 1092)
- volume_matrix: (1943, 1092)
- factors_price_volume: (2121756, 16)
- factors_valuation_snapshot: (1083, 7)
- factors_composite: (2122848, 5)

## Factor Coverage

- universe_ticker_count: 1095
- price_volume_date_range: 2018-01-02 to 2025-12-31
- valuation_ticker_count: 1083
- price_volume_missing_ratio: 0.1191
- valuation_missing_ratio: 0.2020
- composite_missing_ratio: 0.0191

## Composite Row Breakdown

- factors_composite_total_rows: 2122848
- historical_price_volume_rows: 2121756
- latest_snapshot_mixed_rows: 1092

## Composite Semantics

historical_price_volume_composite:
rows: 2121756
date_range: 2018-01-02 to 2025-12-31
is_snapshot_component_used: false
historical_backtest_ready: true

latest_snapshot_mixed_composite:
rows: 1092
as_of_date: 2026-08-14 to 2026-08-14
is_snapshot_component_used: true
historical_backtest_ready: false

## Price-Volume Factor Readiness

momentum_60d: ready
low_volatility_20d: ready
volume_ratio_5d_60d: ready
historical_price_volume_composite: ready
low_volatility_method: atr_20d_over_close

## Alphalens readiness by factor type

momentum_60d: ready
low_volatility_20d: ready
volume_ratio_5d_60d: ready
historical_price_volume_composite: ready
pb_inverse: snapshot_only_not_historical_ready
pe_inverse: snapshot_only_not_historical_ready
dividend_yield: snapshot_only_not_historical_ready
latest_snapshot_mixed_composite: not_historical_ready

## Validation

- alignment_is_aligned: True
- alignment_missing_ratio: 0.0855
- alphalens_ready: True

## Valuation Snapshot Limitation

valuation inputs are latest snapshot data because the valuation date column is empty; they are not historical point-in-time factors.

## Warnings

- latest_snapshot_mixed cannot be used for historical backtest.

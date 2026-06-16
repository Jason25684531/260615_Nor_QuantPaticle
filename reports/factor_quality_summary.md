# Factor Quality Summary

Generated at: 2026-06-16T12:45:02.667086+00:00

## Input Parquet Paths

- universe: D:\01_Project\260615_Nor_QuantPaticle\data\processed\universe.parquet
- valuation: D:\01_Project\260615_Nor_QuantPaticle\data\processed\valuation.parquet
- ohlcv: D:\01_Project\260615_Nor_QuantPaticle\data\processed\ohlcv.parquet

## Artifact Shapes

- close_matrix: (1942, 100)
- high_matrix: (1942, 100)
- low_matrix: (1942, 100)
- volume_matrix: (1942, 100)
- factors_price_volume: (194200, 5)
- factors_valuation_snapshot: (1078, 7)
- factors_composite: (194300, 5)

## Factor Coverage

- universe_ticker_count: 1090
- price_volume_date_range: 2018-01-02 to 2025-12-30
- valuation_ticker_count: 1078
- price_volume_missing_ratio: 0.0166
- valuation_missing_ratio: 0.2042
- composite_missing_ratio: 0.0027

## Composite Row Breakdown

- factors_composite_total_rows: 194300
- historical_price_volume_rows: 194200
- latest_snapshot_mixed_rows: 100

## Composite Semantics

historical_price_volume_composite:
rows: 194200
date_range: 2018-01-02 to 2025-12-30
is_snapshot_component_used: false
historical_backtest_ready: true

latest_snapshot_mixed_composite:
rows: 100
as_of_date: 2026-06-16 to 2026-06-16
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
- alignment_missing_ratio: 0.0039
- alphalens_ready: True

## Valuation Snapshot Limitation

valuation inputs are latest snapshot data because the valuation date column is empty; they are not historical point-in-time factors.

## Warnings

- latest_snapshot_mixed cannot be used for historical backtest.

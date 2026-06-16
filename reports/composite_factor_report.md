# Composite Factor Report

## Run Metadata

- generated_at: 2026-06-16T12:46:09.131057+00:00
- config_path: config\strategy.yaml
- pipeline_name: run_backtest.py

## Factor Scoreboard Summary

- dividend_yield: horizon=<NA>, ir=<NA>, candidate=False, notes=excluded: snapshot-only valuation semantics; limited OHLCV coverage: 100 tickers
- historical_price_volume: horizon=20, ir=0.19209641575416847, candidate=True, notes=weak monotonicity warning; limited OHLCV coverage: 100 tickers
- latest_snapshot_mixed: horizon=<NA>, ir=<NA>, candidate=False, notes=excluded: snapshot-only valuation semantics; limited OHLCV coverage: 100 tickers
- low_volatility_20d: horizon=1, ir=-0.20550134302751608, candidate=True, notes=weak monotonicity warning; limited OHLCV coverage: 100 tickers
- momentum_60d: horizon=20, ir=0.05712210548431289, candidate=True, notes=limited OHLCV coverage: 100 tickers
- pb_inverse: horizon=<NA>, ir=<NA>, candidate=False, notes=excluded: snapshot-only valuation semantics; limited OHLCV coverage: 100 tickers
- pe_inverse: horizon=<NA>, ir=<NA>, candidate=False, notes=excluded: snapshot-only valuation semantics; limited OHLCV coverage: 100 tickers
- volume_ratio_5d_60d: horizon=1, ir=-0.09119237330572674, candidate=True, notes=weak monotonicity warning; limited OHLCV coverage: 100 tickers

## Selected Factor

- selected_factor: historical_price_volume

## Selection Rationale

- historical_price_volume is the Week 3 default because it is an eligible historical composite and the strongest current baseline candidate from Week 2.5 analysis.

## Snapshot Factor Exclusion

- pb_inverse, pe_inverse, dividend_yield, and latest_snapshot_mixed are excluded because valuation data is latest snapshot only.

## IC / IR Summary For Selected Factor

- IC mean=0.03316760387709174, IC std=0.17266123236540404, IR=0.19209641575416847, best_horizon=20

## Quantile Return Summary

- top_bottom_spread: 0.004098668497910179

## Turnover Summary

- avg_turnover: 0.15588920244390256

## Monotonicity Warning

- monotonicity_pass: False

## Universe / OHLCV Coverage

- actual_ohlcv_ticker_count: 100

## Limitations

- This is a research factor selection report, not production readiness.
- OHLCV coverage may be below the full TWSE universe.
- Factor monotonicity checks are imperfect.

## Generated Artifacts

- factor_scoreboard: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_scoreboard.parquet
- selected_factor_scores: D:\01_Project\260615_Nor_QuantPaticle\data\processed\selected_factor_scores.parquet
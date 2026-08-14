# Composite Factor Report

## Run Metadata

- generated_at: 2026-08-13T07:12:01.534477+00:00
- config_path: config\strategy.yaml
- pipeline_name: run_backtest.py

## Factor Scoreboard Summary

- dividend_yield: horizon=<NA>, ir=<NA>, candidate=False, notes=excluded: snapshot-only valuation semantics; limited OHLCV coverage: 100 tickers
- historical_price_volume: horizon=20, ir=0.26885932017353775, candidate=True, notes=weak monotonicity warning; limited OHLCV coverage: 100 tickers
- latest_snapshot_mixed: horizon=<NA>, ir=<NA>, candidate=False, notes=excluded: snapshot-only valuation semantics; limited OHLCV coverage: 100 tickers
- low_volatility_20d: horizon=1, ir=-0.20937648593976985, candidate=True, notes=weak monotonicity warning; limited OHLCV coverage: 100 tickers
- momentum_60d: horizon=20, ir=0.08672693893451405, candidate=True, notes=limited OHLCV coverage: 100 tickers
- pb_inverse: horizon=<NA>, ir=<NA>, candidate=False, notes=excluded: snapshot-only valuation semantics; limited OHLCV coverage: 100 tickers
- pe_inverse: horizon=<NA>, ir=<NA>, candidate=False, notes=excluded: snapshot-only valuation semantics; limited OHLCV coverage: 100 tickers
- volume_ratio_5d_60d: horizon=10, ir=-0.06662259090791946, candidate=True, notes=weak monotonicity warning; limited OHLCV coverage: 100 tickers

## Selected Factor

- selected_factor: historical_price_volume

## Selection Rationale

- historical_price_volume is the Week 3 default because it is an eligible historical composite and the strongest current baseline candidate from Week 2.5 analysis.

## Snapshot Factor Exclusion

- pb_inverse, pe_inverse, dividend_yield, and latest_snapshot_mixed are excluded because valuation data is latest snapshot only.

## IC / IR Summary For Selected Factor

- IC mean=0.046707776397098784, IC std=0.17372571040851706, IR=0.26885932017353775, best_horizon=20

## Quantile Return Summary

- top_bottom_spread: 0.006119533133617923

## Turnover Summary

- avg_turnover: 0.15391998466509668

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
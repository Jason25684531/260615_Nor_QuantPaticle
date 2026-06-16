# Backtest Report

## Run Metadata

- generated_at: 2026-06-16T12:46:09.133734+00:00
- config_path: config\strategy.yaml
- pipeline_name: run_backtest.py

## Scope

- research backtest only
- not live trading
- not investment advice

## Backtest Readiness

- actual_ohlcv_ticker_count: 100
- readiness_status: passed

## Inputs

- close_matrix: D:\01_Project\260615_Nor_QuantPaticle\data\processed\close_matrix.parquet
- factors_composite: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factors_composite.parquet
- topn_positions: D:\01_Project\260615_Nor_QuantPaticle\data\processed\topn_positions.parquet
- portfolio_weights: D:\01_Project\260615_Nor_QuantPaticle\data\processed\portfolio_weights.parquet

## Selected Factor

- selected_factor: historical_price_volume

## Portfolio Construction

- selected_rows: 38440
- top_n: 20
- weighting_rule: equal weight

## Execution Lag

- execution_lag_days: 1
- factor at date T trades on execution_date T+1

## Cost Model

- cost_model_summary: {"buy_cost_rate": 0.002425, "buy_fee_rate": 0.001425, "sell_cost_rate": 0.005425, "sell_fee_rate": 0.001425, "slippage_rate": 0.001, "transaction_tax_rate": 0.003}

## Backtest Engine

- engine: fallback_weight_engine

## Backtest Metrics

- total_return: -0.979296861133108
- annualized_return: -0.39537850256071616
- annualized_volatility: 0.12897879674173846
- sharpe: -3.065453489633687
- max_drawdown: -0.979296861133108
- win_rate: 0.32286302780638515
- turnover: 0.5968074150360453
- avg_exposure: 0.770854788877446
- start_date: 2018-01-02
- end_date: 2025-12-30
- ticker_count: 100
- top_n: 20
- cost_model_summary: {"buy_cost_rate": 0.002425, "buy_fee_rate": 0.001425, "sell_cost_rate": 0.005425, "sell_fee_rate": 0.001425, "slippage_rate": 0.001, "transaction_tax_rate": 0.003}
- engine: fallback_weight_engine

## Equity Curve / Drawdown Summary

- total_return: -0.979296861133108
- max_drawdown: -0.979296861133108

## Limitations

- OHLCV ticker coverage and coverage ratio remain research limitations.
- yfinance fallback data may differ from official TWSE data.
- valuation snapshot factors excluded.
- latest_snapshot_mixed excluded.
- factor monotonicity not fully passed.
- transaction costs are simplified assumptions.

## Generated Artifacts

- factor_scoreboard: D:\01_Project\260615_Nor_QuantPaticle\data\processed\factor_scoreboard.parquet
- selected_factor_scores: D:\01_Project\260615_Nor_QuantPaticle\data\processed\selected_factor_scores.parquet
- topn_positions: D:\01_Project\260615_Nor_QuantPaticle\data\processed\topn_positions.parquet
- portfolio_weights: D:\01_Project\260615_Nor_QuantPaticle\data\processed\portfolio_weights.parquet
- backtest_results: D:\01_Project\260615_Nor_QuantPaticle\data\processed\backtest_results.parquet
- backtest_metrics: D:\01_Project\260615_Nor_QuantPaticle\data\processed\backtest_metrics.parquet
- composite_factor_report: D:\01_Project\260615_Nor_QuantPaticle\reports\composite_factor_report.md
- backtest_report: D:\01_Project\260615_Nor_QuantPaticle\reports\backtest_report.md
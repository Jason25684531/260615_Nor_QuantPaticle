# Backtest Realism Report

---

> **研究聲明 Research Disclaimer**
> - research backtest only · not investment advice · not production-ready
> - OHLCV coverage: 100 tickers (bounded yfinance fallback)
> - valuation snapshot factors excluded (pb_inverse / pe_inverse / dividend_yield / latest_snapshot_mixed)

---

## 1. Run Metadata

- generated_at: 2026-08-11T03:41:50.065594+00:00
- config_path: config/strategy.yaml
- pipeline: run_backtest_diagnostics.py
- factor_name: historical_price_volume
- top_n: 20
- ohlcv_ticker_count: 100

## 2. Purpose

Current baseline: total_return=-0.2432, sharpe=-0.2502.
This report diagnoses performance through:
1. Factor alpha weakness (signal itself has no predictive power)
2. Transaction cost drag (daily rebalance × high costs)
3. Excessive turnover overwhelming any gross return
4. Combination of the above

## 3. Baseline Strategy Recap

- factor: historical_price_volume
- selection: Top 20 by composite score
- weighting: equal weight
- execution: T+1 (signal at date T, trade at T+1)
- baseline result: total_return=-0.2432, annualized_return=-0.0355, sharpe=-0.2502, max_drawdown=-0.3977, turnover=0.1406

## 4. Universe Coverage

- actual OHLCV tickers: 100
- data source: yfinance fallback (TWSE OpenAPI OHLCV not fully available)
- survivorship bias caveat: delisted tickers may be underrepresented

## 5. Selected Factor

- factor_name: historical_price_volume
- eligible: yes (historical price-volume, no snapshot valuation component)
- excluded factors: pb_inverse, pe_inverse, dividend_yield, latest_snapshot_mixed

## 6. Rebalance Frequency Results

| frequency | rebalance_count | total_return | annualized_return | sharpe | max_drawdown | turnover |
|---|---|---|---|---|---|---|
| daily | 1942 | -0.2432 | -0.0355 | -0.2502 | -0.3977 | 0.1406 |
| weekly | 413 | 0.2820 | 0.0327 | 0.2316 | -0.2808 | 0.0970 |
| monthly | 96 | 0.5948 | 0.0624 | 0.4507 | -0.2246 | 0.0422 |

**Turnover by frequency:**
  - daily: turnover=0.1406
  - weekly: turnover=0.0970
  - monthly: turnover=0.0422

## 7. Cost Sensitivity Results

| scenario | total_return | annualized_return | sharpe | max_drawdown | turnover | cost_drag |
|---|---|---|---|---|---|---|
| no_cost | 1.2081 | 0.1082 | 0.7630 | -0.2671 | 0.1406 | 0.0000 |
| half_cost | 0.2928 | 0.0339 | 0.2389 | -0.2858 | 0.1406 | 0.9153 |
| base_cost | -0.2432 | -0.0355 | -0.2502 | -0.3977 | 0.1406 | 1.4513 |
| high_cost | -0.5080 | -0.0879 | -0.6187 | -0.5316 | 0.1406 | 1.7161 |

**Analysis**: no_cost=1.2081, base_cost=-0.2432 → yes, cost removal materially improved return

## 8. Top N Sensitivity Results

| top_n | total_return | annualized_return | sharpe | max_drawdown | turnover | notes |
|---|---|---|---|---|---|---|
| 10 | -0.0915 | -0.0124 | -0.0841 | -0.3244 | 0.1071 |  |
| 20 | -0.2432 | -0.0355 | -0.2502 | -0.3977 | 0.1406 |  |
| 30 | -0.6500 | -0.1273 | -0.9524 | -0.6505 | 0.2388 |  |

## 9. Turnover Diagnostics

- average_daily_turnover: 0.1421
- median_daily_turnover: 0.1000
- max_daily_turnover: 1.0000
- annualized_turnover_estimate: 35.8202
- avg_holdings: 20.0000
- avg_buys_per_rebalance: 0.0918
- avg_sells_per_rebalance: 0.0911
- estimated_cost_drag: 0.1406

## 10. Buffer Rule Impact

- hold_until_drop: True
- drop_rank_buffer: 30
- Status: buffer rule enabled in config. Compare turnover vs non-buffer run for full impact analysis.

## 11. Engine Status / Cross-Check

- vectorbt_status: unavailable

| engine | total_return | sharpe | max_drawdown | turnover | notes |
|---|---|---|---|---|---|
| fallback_weight_engine | -0.2432 | -0.2502 | -0.3977 | 0.1406 | deterministic fallback engine |
| vectorbt | NaN | NaN | NaN | NaN | vectorbt unavailable |

## 12. Interpretation

### Q1: Did no-cost performance improve materially?
- no_cost=1.2081, base_cost=-0.2432 → yes, cost removal materially improved return

### Q2: Did weekly/monthly rebalance reduce turnover?
- daily=0.1406, weekly=0.0970, monthly=0.0422
- Turnover reduction is present but modest, or data insufficient.

### Q3: Did lower turnover improve Sharpe or drawdown?
- Sharpe by frequency: daily=-0.2502, weekly=0.2316, monthly=0.4507

### Q4: Is the strategy failing due to factor weakness or cost/turnover?
- Cost and turnover are the dominant contributors to the negative result. The gross alpha signal may exist but is overwhelmed by transaction costs at daily rebalance frequency.

### Q5: Should this factor proceed to Week 4 research?
- Recommend proceeding to Week 4 if no_cost Sharpe > 0 or if weekly/monthly rebalance shows materially improved performance. Otherwise, consider alternative factor selection first.

## 13. Limitations

- OHLCV input is limited to 100 tickers; see data quality report.
- yfinance fallback data may differ from official TWSE closing prices
- Valuation snapshot factors (pb_inverse, pe_inverse, dividend_yield) excluded — composite is price-volume only
- Backtest period and survivorship bias not fully controlled
- Buffer rule reduces turnover but introduces path-dependency (results depend on history)
- All cost assumptions are simplified (no market impact model)
- vectorbt engine status: unavailable

## 14. Recommended Next Step

1. If no_cost Sharpe > 0: proceed to Week 4 with weekly/monthly rebalance and buffer rule
2. If no_cost Sharpe < 0: investigate alternative factor construction or factor selection criteria
3. Consider expanding OHLCV universe beyond 100 tickers for more robust results
4. Add sub-period analysis (2018–2020 vs 2021–2024) to test time-period stability

## 15. Generated Artifacts

- rebalance_calendar: D:\01_Project\260615_Nor_QuantPaticle\data\processed\rebalance_calendar.parquet
- backtest_scenarios: D:\01_Project\260615_Nor_QuantPaticle\data\processed\backtest_scenarios.parquet
- topn_sensitivity: D:\01_Project\260615_Nor_QuantPaticle\data\processed\topn_sensitivity.parquet
- rebalance_sensitivity: D:\01_Project\260615_Nor_QuantPaticle\data\processed\rebalance_sensitivity.parquet
- backtest_turnover_diagnostics: D:\01_Project\260615_Nor_QuantPaticle\data\processed\backtest_turnover_diagnostics.parquet
- backtest_engine_comparison: D:\01_Project\260615_Nor_QuantPaticle\data\processed\backtest_engine_comparison.parquet
- backtest_realism_report: D:\01_Project\260615_Nor_QuantPaticle\reports\backtest_realism_report.md
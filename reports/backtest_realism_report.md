# Backtest Realism Report

---

> **研究聲明 Research Disclaimer**
> - research backtest only · not investment advice · not production-ready
> - OHLCV coverage: 100 / 1090 tickers (yfinance fallback)
> - valuation snapshot factors excluded (pb_inverse / pe_inverse / dividend_yield / latest_snapshot_mixed)

---

## 1. Run Metadata

- generated_at: 2026-06-17T07:51:02.660093+00:00
- config_path: config/strategy.yaml
- pipeline: run_backtest_diagnostics.py
- factor_name: historical_price_volume
- top_n: 20
- ohlcv_ticker_count: 100

## 2. Purpose

Week 3 baseline backtest produced total_return = -0.979297, Sharpe = -3.065453.
This report diagnoses whether the poor result is caused by:
1. Factor alpha weakness (signal itself has no predictive power)
2. Transaction cost drag (daily rebalance × high costs)
3. Excessive turnover overwhelming any gross return
4. Combination of the above

## 3. Baseline Strategy Recap

- factor: historical_price_volume (equal-weight composite of momentum_60d, low_volatility_20d, volume_ratio_5d_60d)
- selection: Top 20 by composite score
- weighting: equal weight
- execution: T+1 (signal at date T, trade at T+1)
- baseline result: total_return=-0.979297, annualized_return=-0.395379, sharpe=-3.065453, max_drawdown=-0.979297, turnover=0.596807

## 4. Universe Coverage

- actual OHLCV tickers: 100 / 1090 TWSE listed
- data source: yfinance fallback (TWSE OpenAPI OHLCV not fully available)
- survivorship bias caveat: delisted tickers may be underrepresented

## 5. Selected Factor

- factor_name: historical_price_volume
- eligible: yes (historical price-volume, no snapshot valuation component)
- excluded factors: pb_inverse, pe_inverse, dividend_yield, latest_snapshot_mixed

## 6. Rebalance Frequency Results

| frequency | rebalance_count | total_return | annualized_return | sharpe | max_drawdown | turnover |
|---|---|---|---|---|---|---|
| daily | 1941 | -0.9623 | -0.3464 | -2.6439 | -0.9623 | 0.5059 |
| weekly | 413 | -0.9394 | -0.3050 | -4.3673 | -0.9394 | 0.4145 |
| monthly | 96 | -0.5296 | -0.0932 | -1.8877 | -0.5296 | 0.0844 |

**Turnover by frequency:**
  - daily: turnover=0.5059
  - weekly: turnover=0.4145
  - monthly: turnover=0.0844

## 7. Cost Sensitivity Results

| scenario | total_return | annualized_return | sharpe | max_drawdown | turnover | cost_drag |
|---|---|---|---|---|---|---|
| no_cost | 0.7923 | 0.0787 | 0.6142 | -0.2647 | 0.5059 | 0.0000 |
| half_cost | -0.7394 | -0.1601 | -1.2452 | -0.7425 | 0.5059 | 1.5317 |
| base_cost | -0.9623 | -0.3464 | -2.6439 | -0.9623 | 0.5059 | 1.7546 |
| high_cost | -0.9920 | -0.4657 | -3.4863 | -0.9920 | 0.5059 | 1.7843 |

**Analysis**: no_cost=0.7923, base_cost=-0.9623 → yes, cost removal materially improved return

## 8. Top N Sensitivity Results

| top_n | total_return | annualized_return | sharpe | max_drawdown | turnover | notes |
|---|---|---|---|---|---|---|
| 10 | -0.9582 | -0.3377 | -2.4850 | -0.9582 | 0.4858 |  |
| 20 | -0.9623 | -0.3464 | -2.6439 | -0.9623 | 0.5059 |  |
| 30 | -0.9739 | -0.3769 | -3.0449 | -0.9739 | 0.5603 |  |

## 9. Turnover Diagnostics

- average_daily_turnover: 0.3099
- median_daily_turnover: 0.3000
- max_daily_turnover: 1.0000
- annualized_turnover_estimate: 78.0912
- avg_holdings: 20.0000
- avg_buys_per_rebalance: 0.1568
- avg_sells_per_rebalance: 0.1563
- estimated_cost_drag: 0.3065

## 10. Buffer Rule Impact

- hold_until_drop: True
- drop_rank_buffer: 30
- Status: buffer rule enabled in config. Compare turnover vs non-buffer run for full impact analysis.

## 11. Engine Status / Cross-Check

- vectorbt_status: unavailable

| engine | total_return | sharpe | max_drawdown | turnover | notes |
|---|---|---|---|---|---|
| fallback_weight_engine | -0.9793 | -3.0655 | -0.9793 | 0.5968 | deterministic fallback engine |
| vectorbt | NaN | NaN | NaN | NaN | vectorbt unavailable |

## 12. Interpretation

### Q1: Did no-cost performance improve materially?
- no_cost=0.7923, base_cost=-0.9623 → yes, cost removal materially improved return

### Q2: Did weekly/monthly rebalance reduce turnover?
- daily=0.5059, weekly=0.4145, monthly=0.0844
- Turnover reduction is present but modest, or data insufficient.

### Q3: Did lower turnover improve Sharpe or drawdown?
- Sharpe by frequency: daily=-2.6439, weekly=-4.3673, monthly=-1.8877

### Q4: Is the strategy failing due to factor weakness or cost/turnover?
- Cost and turnover are the dominant contributors to the negative result. The gross alpha signal may exist but is overwhelmed by transaction costs at daily rebalance frequency.

### Q5: Should this factor proceed to Week 4 research?
- Recommend proceeding to Week 4 if no_cost Sharpe > 0 or if weekly/monthly rebalance shows materially improved performance. Otherwise, consider alternative factor selection first.

## 13. Limitations

- OHLCV coverage is 100 / 1090 (9.2%) — small universe introduces concentration risk
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
# Migration Notes

`PythonQuantTrading` is reference-only material for this project. The repository can
inform examples and migration decisions, but `twse-factor-lab` does not fork its
architecture and does not depend on FinLab.

## Reference Mapping

| PythonQuantTrading area | Week 1 direction |
| --- | --- |
| yfinance examples | Reworked as `YFinanceClient` OHLCV fallback |
| FinLab data examples | Replaced by `TWSEClient` and `YFinanceClient` |
| Backtrader examples | Future vectorbt migration reference |
| Pyfolio reports | Future local metrics and Markdown reports |
| Alpha191 / TA-Lib ideas | Future factor engineering reference |
| ML / Shioaji examples | Outside MVP scope |

## FinLab Replacement

- `finlab.data.get()` is replaced by `TWSEClient` for official TWSE data and
  `YFinanceClient` for OHLCV fallback.
- `finlab.backtest.sim()` is not implemented in Week 1. Later changes will move
  the backtest layer toward vectorbt.
- Backtrader and Pyfolio examples are useful conceptually, but Week 1 only builds
  the data foundation needed by later factor analysis and backtesting work.

## Week 1 Boundary

Week 1 creates the project foundation, official data access, yfinance fallback,
Parquet cache, and data quality report. It does not implement Alphalens analysis,
vectorbt backtests, portfolio construction, cost models, live trading, or ML.

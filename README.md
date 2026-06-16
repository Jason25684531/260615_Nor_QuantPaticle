# TWSE Factor Lab

`twse-factor-lab` is a Taiwan equity factor research foundation. Week 1 focuses on
the official data layer, yfinance OHLCV fallback, Parquet research cache, and data
quality reporting. Week 2 extends those existing Parquet outputs into research-ready
matrices, factors, validation helpers, and Alphalens formatter readiness. Week 3
adds config-driven OHLCV universe expansion readiness, a factor scoreboard, Top N
selection, T+1 equal-weight portfolio construction, and a research-only backtest.

## Week 1 Scope

Included:

- Python project structure with `src/` package layout
- TWSE OpenAPI client and endpoint registry
- yfinance OHLCV fallback for Taiwan tickers
- Normalized universe, valuation, and OHLCV schemas
- Parquet processed-data cache
- Data quality summary
- Unit tests that do not require live network access

Not included in Week 1:

- Factor calculation
- Alphalens analysis
- vectorbt backtesting
- Portfolio construction
- Trading cost model
- Shioaji live trading
- ML / DL workflows

## Data Sources

| Data | Primary source | Fallback | Output |
| --- | --- | --- | --- |
| Listed-company universe | TWSE OpenAPI | Manual CSV later | `data/processed/universe.parquet` |
| PE / PB / dividend yield | TWSE OpenAPI | None | `data/processed/valuation.parquet` |
| OHLCV | yfinance | TWSE close/volume later | `data/processed/ohlcv.parquet` |
| Data quality | Local pipeline | None | `reports/data_quality_summary.md` |

## Week 2 Scope

Included:

- OHLCV matrix artifacts for close, high, low, and volume
- Price-volume factors, snapshot-safe valuation factors, and composite outputs
- Factor alignment checks and no-lookahead guard helpers
- Alphalens-ready formatter inputs and readiness validation
- Factor quality summary and artifact manifest

Not included in Week 2:

- vectorbt, backtesting, or portfolio construction
- transaction cost modeling
- historical reconstruction of valuation factors from empty `valuation.date`
- full Alphalens tear sheet generation

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Run Data Pipeline

```bash
.venv\Scripts\python run_data_pipeline.py --config config/strategy.yaml
```

The pipeline uses TWSE OpenAPI for official data and yfinance for a bounded OHLCV
fallback subset configured in `config/strategy.yaml`.

For Week 3, the default OHLCV setting is `data.ohlcv.ticker_limit: 100`, with
batching, retry, failed ticker logging, and coverage metrics in
`reports/data_quality_summary.md`.

## Run Factor Pipeline

```bash
.venv\Scripts\python run_factor_pipeline.py --config config/strategy.yaml
```

The Week 2 factor pipeline reads the Week 1 Parquet outputs and writes:

- `data/processed/close_matrix.parquet`
- `data/processed/high_matrix.parquet`
- `data/processed/low_matrix.parquet`
- `data/processed/volume_matrix.parquet`
- `data/processed/factors_price_volume.parquet`
- `data/processed/factors_valuation_snapshot.parquet`
- `data/processed/factors_composite.parquet`
- `data/processed/_manifest.json`
- `reports/factor_quality_summary.md`

Valuation factors are treated as latest snapshot data because `valuation.parquet`
currently has an empty `date` column. The Week 2 pipeline reports that limitation
explicitly and does not pretend those factors are historical point-in-time data.

## Run Factor Analysis

```bash
.venv\Scripts\python run_factor_analysis.py --config config/strategy.yaml
```

The Week 2.5 analysis pipeline writes forward returns, IC/IR summaries, quantile
returns, turnover, monotonicity checks, and `reports/factor_analysis_report.md`.

## Run Backtest Pipeline

```bash
.venv\Scripts\python run_backtest.py --config config/strategy.yaml
```

The Week 3 backtest pipeline consumes the latest data, factor, and factor-analysis
artifacts. It checks `backtest.min_ticker_count` before producing a normal baseline
report, defaults to `historical_price_volume`, applies Top N selection, T+1 equal
weights, and a fee/tax/slippage cost model.

Snapshot valuation factors remain excluded from historical backtests:

- `pb_inverse`
- `pe_inverse`
- `dividend_yield`
- `latest_snapshot_mixed`

Week 3 remains research-only. It is not live trading, broker integration, or
investment advice.

## Run Tests and Checks

```bash
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m black .
```

## Roadmap

- Week 1: data layer, Parquet cache, quality report
- Week 2: factor engineering and Alphalens formatter readiness
- Week 2.5: historical factor analysis
- Week 3: OHLCV expansion readiness, factor scoreboard, Top N portfolio, and `vectorbt` or fallback backtesting
- Week 4: robustness checks, charts, and Markdown reports

`alphalens-reloaded` and `vectorbt` are roadmap dependencies. They are not required
to run Week 1 tests, the Week 1 data pipeline, or the Week 2 factor pipeline.

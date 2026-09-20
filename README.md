# TWSE Factor Lab

## Fundamental runtime status

- Research: COMPLETE
- Strategy: FROZEN (`fundamental_g2g3_top5_reb60_score_weighted_v1`)
- Canonical Runtime Provider: PASS
- Real Three-Way Parity: PASS
- Production Engineering: COMPLETE
- Fresh OOS: PENDING FUTURE DATA
- Production Ready: NO; Production Enable: FALSE; Broker: DISABLED
- Final Status: ENGINEERING_COMPLETE_OOS_PENDING

## Project Overview

An offline-first Taiwan equity factor research platform. It produces reproducible
Parquet evidence and research-only backtests; it is not a live trading system or
investment advice.

The repository has two layers. **Legacy RC1** is a frozen, closed research
cycle — its evidence, hashes, and verdicts are immutable and are never
rewritten. The **New Research Validation Platform** is the active
governance/validation harness used for all research cycles after RC1; it
never modifies Legacy RC1 evidence. See the sections below for each layer.

## Quick Start / Installation

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Configuration

The canonical configuration is `config/strategy.yaml`. It controls data sources,
factor windows, PIT rules, portfolio construction, execution lag, costs, and
diagnostic grids. Do not edit frozen values for an RC1 reproduction.

## Testing

```bash
python -m pytest
python -m ruff check .
openspec validate --all --strict
```

## Repository architecture and command lifecycle

Reusable research behavior lives under `src/twse_factor_lab/` by domain:
`data`, `factors`, `portfolio`, `backtest`, `analysis`, `governance`,
`acceptance`, and `reporting`. Thin orchestration owners live under
`src/twse_factor_lab/application/commands/`; temporary legacy or frozen replay
adapters live under `application/compatibility/` when needed. Operational
one-off jobs remain under `jobs/`.

`docs/architecture/runner_inventory.json` classifies every root runner and job
as supported, compatibility, diagnostic, operational-job, or retired. The
current supported research-cycle default is v3. The inventory and cleanup
ledger are validated by `tests/test_architecture_boundaries.py`.

The root `run_research_report.py` is now a compatibility adapter to the
application command. Other root runners remain intentionally classified until
their callers and retention obligations are migrated. In particular,
`run_backtest.py` is retained because tests import it and it can overwrite
canonical composite artifacts; use the canonical composite runner for the
supported path.

CI is offline/deterministic and runs install, Ruff, pytest, and OpenSpec strict
validation. It does not require Yahoo, TWSE, MOPS, API keys, or private data.

## Command lifecycle and safe entry points

Use the application command as the supported interface when one exists. Root
`run_*.py` files are retained as compatibility adapters, frozen replay entry
points, or explicitly pending migration cohorts. The runner inventory is the
source of truth for lifecycle, owner, output namespace, and retirement status;
do not infer support from a filename alone.

| Lifecycle | Current command index | Output policy |
| --- | --- | --- |
| Supported | `run_data_pipeline.py`, `run_fundamental_pipeline.py`, `run_factor_pipeline.py`, `run_factor_analysis.py`, `run_factor_tearsheet.py`, `run_composite_strategy.py`, `run_final_acceptance.py`, `run_performance_report.py`, `run_historical_research_cycle.py`, `run_research_cycle_v3.py`, `run_robustness.py` | Existing legacy-canonical paths remain until a cohort migration proves equivalent package output. |
| Compatibility | `run_research_report.py`, `run_backtest.py`, `run_research_cycle_v2.py`, `run_rc1_offline_e2e.py` | Preserve documented arguments and callers; RC1 replay is read-only. `run_research_report.py` delegates to `twse_factor_lab.application.commands.research_report`. |
| Diagnostic | `run_backtest_diagnostics.py`, `run_engine_parity_audit_v1.py`, `run_composite_factor_admission_v1.py`, `run_composite_strategy_lab_and_pyfolio_v1.py`, `run_controlled_factor_discovery_v4.py`, `run_final_strategy_validation_v1.py`, `run_final_strategy_validation_v2.py`, `run_final_strategy_validation_v3.py`, `run_fresh_oos_validation_v1.py`, `run_fundamental_pit_coverage.py` | Non-canonical evidence only; diagnostic results never promote a strategy automatically. |
| Operational | `run_update_market_data.py`, `run_shadow_daily.py`, `run_shadow_runtime_s3.py`, `run_official_migration.py`, `run_project_closure.py`, `run_daily_fundamental_production.py`, `jobs/run_fundamental_pit_expansion.py`, `jobs/backfill_operating_income_pit.py` | Runtime/operation writes are gated and fail closed. |

## Artifact ownership

```text
repository root              immutable RC1 handoffs and manifests only
data/processed/              legacy RC1 canonical artifacts; no new active writes
data/research/<cycle>/       reproducible active research evidence
data/diagnostics/<run>/      non-canonical diagnostic evidence
data/runtime/<run>/          shadow and operational state
reports/{final,rc1}/         frozen human-readable evidence
reports/research/<cycle>/    active research reports
outputs/                     explicitly transient, ignored output only
```

`docs/architecture/artifact_inventory.json` records effective ownership and
retention. Before moving or deleting a file, check
`docs/architecture/cleanup_ledger.json`: unknown and frozen items stay in
place, and every relocation needs references, hash impact, rollback, and
verification evidence. `docs/architecture/runner_cohort_matrix.json` records
why a runner is pending migration; empty application command modules are not
created just to satisfy a planned target.

## Legacy RC1 (Frozen)

### Current Status

Research Platform RC1: **PASS / CLOSED**  
Strategy: **REJECTED**

### Architecture

```text
TWSE / yfinance inputs → D2 integrity + universe → D2.5 PIT
→ D3 factors → D3.5 composite/breadth → D1 execution calendar
→ Custom + Vectorbt parity → Pyfolio → D4 robustness → D5 acceptance → RC1 governance
```

Signals are formed on T and executed on the next trading day T+1. No same-day
execution or future-price fill is part of the canonical semantics.

### Research Governance

D1-D5 evidence is frozen. Selected factors are `risk_adjusted_momentum` and
`historical_price_volume` with weights 0.5/0.5; Top-N=20, buffer=30, daily
rebalance, breadth threshold=0.40, exposure 1.0/0.5, and cost assumptions are
immutable in RC1. D4 is `REJECT`; D5 Strategy Acceptance is `REJECTED`.

### Canonical Pipeline

Run the supported stages in this order:

```text
run_data_pipeline.py
run_fundamental_pipeline.py
run_factor_pipeline.py
run_factor_analysis.py
run_factor_tearsheet.py
run_composite_strategy.py
run_performance_report.py
run_robustness.py
run_final_acceptance.py
```

Use `--config config/strategy.yaml` only on runners that expose that option.
The legacy `run_backtest.py` can overwrite canonical composite artifacts; do not
run it after the canonical D3.5 strategy runner.

### D1-D5 Stages

- D1: trading calendar and T+1 execution semantics.
- D2: adjusted OHLCV, source integrity, eligibility, and PARTIAL TWSE universe.
- D2.5: point-in-time fundamental alignment.
- D3/D3.5: factor research, composite scoring, buffer, and breadth.
- D4: IS/OOS robustness; verdict `REJECT`.
- D5: frozen OOS acceptance; Strategy Acceptance `REJECTED`.

### Custom / Vectorbt Economic Parity

The Custom engine is the reference path. Vectorbt is cross-checked on the same
weights and cost model in `data/processed/backtest_engine_comparison.parquet`.

### Pyfolio

Pyfolio reporting is available for performance analysis. Transaction data is a
known limitation and is not used to replace canonical trade-excursion evidence.

### Artifacts

Root machine-readable frozen evidence includes the D4/D5 handoffs, trial
inventory, statistical acceptance, trade excursions, and processed research
artifacts. Human-readable canonical reports are under `reports/final/`.
`final_artifact_inventory.json` records existence, schema/date checks, hashes,
and canonical-location decisions. Large generated Parquet is not committed
merely to satisfy RC1.

### Reproducibility

`reproducibility_manifest.json` records environment, git state, configuration
hash, freeze ID, D4/D5 hashes, frozen parameters, verdicts, archive identifiers,
and artifact hashes. `reports/rc1/` contains the clean-install and offline E2E
reports.

### Known Limitations

The platform retains PARTIAL TWSE coverage, survivorship bias, external data
gaps, limited fundamental coverage, transaction-cost and execution-price
assumptions, PARTIAL-universe breadth, benchmark/attribution limits, missing
Pyfolio transactions, and close-to-close MAE/MFE approximation.

### Final Verdict

Final status: **Research Platform RC1: PASS / CLOSED / Strategy: REJECTED**.

## New Research Validation Platform

Everything below `src/twse_factor_lab/` outside the legacy D1-D5 modules is
the active research validation platform used for research cycles after RC1.
It reuses the canonical Custom engine, cost model, and T→T+1 execution
semantics; it does not run a second backtester.

```text
PIT Data
  -> Research Governance      (governance/: manifests, experiment registry, isolation/freeze)
  -> Factor Library           (factors/registry.py)
  -> Factor Gate              (analysis/factor_gate.py: pre-declared primary_horizon verdict)
  -> Multi-Factor Strategy Lab(strategy/lab.py, strategy/handoff.py)
  -> Custom / Vectorbt / Backtrader (backtest/*_engine.py: tri-engine parity)
  -> Canonical + Pyfolio      (analysis/performance.py: definition-aware cross-check)
  -> Barra-style Attribution  (analysis/attribution.py)
  -> Robustness               (analysis/research_robustness.py: diagnostic-only)
  -> Fresh-state OOS          (acceptance/research_cycle.py: fresh cash/positions re-execution)
  -> PSR / DSR                (acceptance/psr.py: strategy-only trial population)
  -> Acceptance                (acceptance/research_cycle.py: acceptance matrix)
  -> Freeze                    (acceptance/research_cycle.py: hash + inventory + code revision)
```

Key correctness properties enforced by this layer:

- Factor Gate verdicts use a pre-declared `primary_horizon`; other horizons are
  diagnostics only, never a post-hoc "best horizon" pick.
- OOS evaluation re-executes from fresh cash and empty positions at the OOS
  start date; pre-OOS history is warmup-only and cannot leak P&L or inherited
  positions into OOS results.
- DSR/PSR trial populations only include terminal, selection-relevant
  `strategy_backtest` experiments; factor-selection and diagnostic experiments
  (Pyfolio, Backtrader, Barra, robustness, ablation) never inflate the
  strategy Sharpe trial count, and missing Sharpe values are never imputed.
- Once a research cycle is frozen (`freeze/research_freeze_manifest.json`
  exists), every write under its namespace fails fast; freeze metadata
  records git commit SHA, git-dirty status, Python/project version, and the
  installed numpy/pandas/scipy/statsmodels/vectorbt/pyfolio/backtrader
  versions, plus a complete relative-path inventory so any unexpected file
  added after freezing is detected on verification.

This platform never rewrites Legacy RC1 evidence, `reports/final/`,
`data/processed/` canonical artifacts, or the RC1 verdict above.

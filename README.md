# TWSE Factor Lab

## Current Status

- Research engineering: complete
- Current strategy: frozen Fundamental G2/G3 Top5, score-weighted, REB60
- Fingerprint: `cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f`
- Runtime parity: PASS across Research, Shadow, and Production evidence
- Fresh OOS: `INSUFFICIENT_DATA` (37 observed trading days; 180 required)
- Production eligibility: BLOCKED; enable flag false; broker submission disabled
- Current branch/HEAD: `main` / `7688671aaffef6baccb873acbf3bd0afc0e4ea7f`

This is an offline-first Taiwan equity research and shadow-runtime repository.
It is not a live trading system or investment advice.

## Frozen Strategy

The source of truth is `FundamentalStrategySpec` in
`src/twse_factor_lab/production/fundamental.py`:

- Factors: `G2_OPERATING_INCOME_YOY` and `G3_EPS_YOY`
- Factor weighting: equal (50/50)
- Selection: Top5
- Portfolio weighting: score-weighted
- Rebalance: 60D with carry-forward between due dates
- Research cutoff: `2026-07-28`
- Fresh OOS minimum: 180 trading days / 9 months / 3 rebalances

Do not change strategy identity, ranking, selection, weights, REB60, costs,
benchmark binding, cutoff, or Fresh OOS rules during cleanup.

## Architecture

```text
data sources → PIT canonical stores → G2/G3 factors
             → score/rank/Top5 → score-weighted REB60 targets
             → Research → Shadow → Production parity
```

Reusable code lives under `src/twse_factor_lab/`; application owners live in
`src/twse_factor_lab/application/commands/`; operational jobs live in `jobs/`.
The authoritative runner lifecycle is
`docs/architecture/runner_inventory.json`. Root runners are supported,
compatibility, diagnostic, operational, or frozen replay entrypoints; do not
infer lifecycle from filenames. The architecture check is:

```powershell
python -m twse_factor_lab.application.commands.architecture_check
```

The current Fundamental daily adapter is `run_daily_fundamental_production.py`.
The research-report adapter is `run_research_report.py`. Frozen RC1 replay and
hash-bound validation runners remain at their recorded root paths.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
openspec validate --all --strict
```

Use `config/strategy.yaml` for pipeline paths and data settings. Use the
application owner or the documented runner inventory entry for each workflow;
do not invoke diagnostic runners as promotion commands.

## Daily Shadow / Fresh OOS

```powershell
python run_daily_fundamental_production.py --validate-only
python run_daily_fundamental_production.py --dry-run
```

The safe flow is scheduler → market/fundamental refresh → PIT → freshness →
factor health → historical parity → shadow → Fresh OOS ledger → availability →
monitoring. Fresh OOS observations must be legally untouched; contaminated
periods must never be relabeled or appended as Fresh OOS.

Production writes are fail-closed until the formal Fresh OOS gate passes and a
human explicitly enables production. Broker orders are disabled.

## Research Evidence

Canonical current evidence is under
`data/research/fundamental-production-final-v1/`. It includes historical
factor/PIT evidence, benchmark comparisons, runtime parity, Fresh OOS status,
risk-overlay rejection, survivorship limitation, and the production gate.

Derivative publication figures/tables are generated under
`docs/research-publication/` by
`twse_factor_lab.reporting.publication_evidence`; they do not replace
canonical accounting or factor evidence.

Legacy RC1 evidence remains immutable at its recorded root, `data/processed/`,
`reports/final/`, and `reports/rc1/` locations. Adverse overlays, blocked
survivorship evidence, benchmark limitations, and insufficient Fresh OOS are
formal provenance and are not deleted.

RC1: **PASS / CLOSED**
Strategy: **REJECTED**

## Repository Structure

| Path | Purpose |
|---|---|
| `src/twse_factor_lab/` | Reusable domain, runtime, governance, and reporting code |
| `jobs/` | Operational PIT/data jobs |
| `config/` | Active pipeline and historical-evidence configuration |
| `tests/` | Unit, research, runtime, integration, and governance coverage |
| `data/processed/` | Retained canonical/RC1 artifacts |
| `data/research/` | Research-cycle and Fundamental evidence |
| `data/runtime/` | Shadow and operational state |
| `data/project-closure/` | Engineering-closure evidence |
| `docs/handoff/` | Five core handoff docs plus inventory and cleanup records |
| `docs/archive/` | Evidence/archive navigation; original paths remain canonical |
| `docs/architecture/` | Runner, artifact, cohort, and cleanup governance |
| `openspec/` | Current specs and archived change decisions |

## Documentation

Start with the five handoff documents:

1. [Research overview](docs/handoff/01_RESEARCH_OVERVIEW.md)
2. [Frozen strategy specification](docs/handoff/02_STRATEGY_SPEC.md)
3. [Data and PIT](docs/handoff/03_DATA_AND_PIT.md)
4. [Validation results](docs/handoff/04_VALIDATION_RESULTS.md)
5. [Operations handoff](docs/handoff/05_OPERATIONS_HANDOFF.md)

Cleanup governance:

- [Repository inventory](docs/handoff/repository_inventory.md)
- [Dependency map](docs/handoff/dependency_map.md)
- [Cleanup plan](docs/handoff/cleanup_plan.md)
- [Cleanup summary](docs/handoff/cleanup_summary.md)
- [Archive index](docs/archive/index.md)
- [OpenSpec index](docs/handoff/openspec_index.md)

Before moving or deleting anything, read
`docs/architecture/cleanup_ledger.json` and the handoff dependency map.
Unknown, frozen, hash-bound, or reproduction-bound paths stay in place.

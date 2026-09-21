# Repository architecture

The repository has four explicit ownership zones:

```text
src/twse_factor_lab/
  data/ factors/ portfolio/ backtest/ analysis/ governance/ acceptance/
      reusable domain and infrastructure behavior
  reporting/                         presentation and report assembly
  application/commands/              supported orchestration owners
  application/compatibility/         temporary legacy/frozen adapters
jobs/                                 operational one-off jobs
tests/                                behavior and architecture contracts
data/, reports/                      generated research namespaces
openspec/                             requirements and change artifacts
```

Reusable domain packages must not import application commands, root runners, or
report renderers. Commands parse configuration, compose domain services, invoke
the workflow, and report results. A protocol or abstract base is introduced
only when the application genuinely selects between interchangeable
implementations; otherwise a function or value object is preferred.

The migration is incremental. `research_report` is the current application
command pilot and `run_research_report.py` is its tested compatibility
adapter. `run_daily_fundamental_production.py` is the operational pilot and
delegates to `application.commands.fundamental_final`. The remaining root runners are not assumed to have package owners
merely because an inventory migration target was once planned: their actual
cohort, blockers, and frozen/hash-bound exclusions are recorded in
`runner_cohort_matrix.json`. Empty command modules are not created to make the
inventory look complete.

## Runner lifecycle

`runner_inventory.json` is the source of truth for root `run_*.py` files and
operational jobs. The current default research cycle is v3. Root runners that
are still imported or needed for frozen replay remain compatibility paths until
their documented retention period expires. Diagnostic and experimental runners
are not part of the supported canonical path.

`artifact_inventory.json` is the source of truth for logical artifact
ownership. Existing `data/processed/`, `reports/final/`, and `reports/rc1/`
content remains in place when it is canonical or frozen. New active output is
classified before it is routed to research, diagnostics, runtime, or transient
namespaces. `cleanup_ledger.json` records evidence, rollback, and verification
for every move/delete candidate; unknown candidates are left untouched.
Tracked ignored material under `data/` and `reports/` is covered by the same
artifact inventory rather than treated as disposable. `.tokensave/` is local,
ignored tool state and is intentionally not versioned. `outputs/` is transient,
but `outputs/fundamental_data/` is retained while current PIT workflows name it.

For atomic material writes, reuse `data.parquet_store.ParquetStore.save` for
Parquet frames and the existing runtime/governance atomic JSON helpers where
their namespace contract applies. The cleanup does not add a generic writer
base class and does not merge helpers whose validation or freeze semantics
differ.

## Cleanup policy

`cleanup_ledger.json` records the evidence and owner for each cleanup candidate.
`unknown` blocks deletion. RC1/frozen evidence is always `retain` and must stay
at its canonical location. Ignored logs and caches can be removed only when no
active workflow or retained diagnostic depends on them.

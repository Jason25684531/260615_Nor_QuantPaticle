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

## Runner lifecycle

`runner_inventory.json` is the source of truth for root `run_*.py` files and
operational jobs. The current default research cycle is v3. Root runners that
are still imported or needed for frozen replay remain compatibility paths until
their documented retention period expires. Diagnostic and experimental runners
are not part of the supported canonical path.

## Cleanup policy

`cleanup_ledger.json` records the evidence and owner for each cleanup candidate.
`unknown` blocks deletion. RC1/frozen evidence is always `retain` and must stay
at its canonical location. Ignored logs and caches can be removed only when no
active workflow or retained diagnostic depends on them.

# Cleanup baseline

Captured during `refactor-architecture-cleanup` implementation.

## Current shape

- 22 root `run_*.py` files and one operational job under `jobs/`.
- Reusable package areas: `data`, `factors`, `portfolio`, `backtest`,
  `analysis`, `governance`, `acceptance`, `reporting`, and `strategy`.
- Application orchestration now starts at
  `src/twse_factor_lab/application/`; `research_report` is the first migrated
  owner and `run_research_report.py` is its compatibility adapter.
- Version-controlled root evidence includes RC1/D4/D5 handoffs, manifests,
  statistical/trial Parquet, trade excursions, and the strategy freeze
  manifest. These are listed as `retain` in `cleanup_ledger.json`.

## Reference audit

The inventory was checked against:

- Python imports and public test imports (`rg` over `*.py`).
- README, `docs/`, OpenSpec artifacts, and `.github/workflows/ci.yml` command
  references.
- Frozen/research manifests and the read-only `run_rc1_offline_e2e.py` replay.
- Ignored transient output references, including the two `v4-run.*.log` files
  removed during this change.

The contract is exercised by `tests/test_architecture_boundaries.py` and the
exported `run_architecture_checks(root)` function.

## Lifecycle decision

The supported research-cycle default is v3. v2 remains a compatibility path
for reproducibility, RC1 replay remains a frozen compatibility path, and
diagnostic/experimental runners remain available but are not part of the
canonical supported chain. Compatibility adapters are retained for at least
one release after their application replacement is documented and tested.

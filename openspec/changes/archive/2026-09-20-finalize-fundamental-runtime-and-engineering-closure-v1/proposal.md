## Why

The frozen Fundamental strategy has a valid PIT store and safe runtime seams, but its checked-in canonical input does not contain the frozen G2 operating-income series.  The resulting provider correctly fails closed, leaving real Research/Shadow/Production parity and the daily E2E unavailable.

## What Changes

- Complete the canonical Fundamental runtime from the repository's PIT records, security universe, and trading calendar without changing frozen strategy semantics.
- Produce non-empty, independently invoked Research, Shadow, and Production snapshots from predeclared historical dates and require zero-mismatch parity.
- Exercise the safe daily recommendation path, persistence, Web/LINE serializers, and fail-closed gates in an isolated validation namespace.
- Emit immutable closure evidence and documentation while retaining Fresh OOS as an external, future-data promotion gate; production and broker submission remain disabled.

## Capabilities

### New Capabilities

- `canonical-runtime-provider`: Builds auditable, PIT-safe frozen G2/G3 runtime snapshots directly from canonical records.
- `real-three-way-parity`: Validates independently invoked Research, Shadow, and Production snapshots against fixed historical fixtures.
- `daily-production-runtime`: Defines the safe daily execution, persistence, and consumer-adapter closure contract.
- `engineering-closure`: Produces the final machine-readable status, OOS gate, documentation, and archive evidence.

### Modified Capabilities

- None.

## Impact

Affected areas are `src/twse_factor_lab/production/`, the Fundamental PIT ingestion/runtime data contract, `run_daily_fundamental_production.py`, regression tests, closure evidence under `data/project-closure/`, and project documentation.  No new dependency, live order path, automatic production enablement, or strategy parameter is introduced.

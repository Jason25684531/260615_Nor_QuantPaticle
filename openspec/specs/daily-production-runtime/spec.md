# Daily Production Runtime

## Purpose

Define the safe Fundamental recommendation execution contract.

## Requirements

### Requirement: Daily validation uses the canonical runtime safely
The daily command SHALL resolve a trading session, run freshness, provider, PIT, factor-health, parity, ranking, Top5, REB60, target, recommendation, persistence, Web, LINE, and health stages in order. It SHALL provide validate-only, dry-run, and explicit historical write modes with deterministic exit behavior and disabled broker submission.

#### Scenario: Valid historical rebalance fixture
- **WHEN** an eligible historical rebalance fixture is validated in an isolated namespace
- **THEN** non-empty recommendations are atomically persisted and Web and LINE fixtures derive from the same recommendation records

#### Scenario: A safety gate fails
- **WHEN** stale data, future PIT data, factor health, provider, parity, or eligibility fails
- **THEN** no active recommendation is written and broker submission remains disabled

### Requirement: Recommendation persistence is idempotent
The recommendation store SHALL use the natural key formed by date, strategy fingerprint, input SHA, and ticker, write atomically, and make ten identical reruns stable with zero duplicate records and stable content and identifiers.

#### Scenario: Same input is rerun
- **WHEN** an identical historical recommendation run is persisted ten times
- **THEN** its row count, IDs, and contents remain unchanged

# Daily Production Runtime

## Purpose

Define the safe Fundamental recommendation execution contract.
## Requirements
### Requirement: Daily validation uses the canonical runtime safely
The daily command SHALL default to a scheduler-safe execution against the latest completed trading session in canonical market data, SHALL allow an explicit as-of date, and SHALL run freshness, provider, PIT, factor-health, parity, ranking, Top5, REB60, target, recommendation, persistence, Web, LINE, and health stages in order. Historical closure validation SHALL require an explicit mode. The command SHALL provide validate-only, dry-run, isolated observation, and explicit historical write modes with deterministic exit behavior and disabled broker submission.

#### Scenario: Default daily invocation
- **WHEN** the daily command is invoked without an as-of date or historical-validation mode
- **THEN** it resolves the latest completed canonical trading session and does not regenerate historical closure evidence

#### Scenario: Valid historical rebalance fixture
- **WHEN** an eligible historical rebalance fixture is validated in an isolated namespace
- **THEN** non-empty recommendations are atomically persisted and Web and LINE fixtures derive from the same recommendation records

#### Scenario: A safety gate fails
- **WHEN** stale data, future PIT data, factor health, provider, parity, or eligibility fails
- **THEN** no active production recommendation is written and broker submission remains disabled

#### Scenario: Explicit historical validation
- **WHEN** historical closure validation is explicitly requested
- **THEN** the historical validator runs without invoking or mutating the daily recommendation store

### Requirement: Recommendation persistence is idempotent
The recommendation store SHALL use the natural key formed by date, strategy fingerprint, input SHA, and ticker, write atomically, and make ten identical reruns stable with zero duplicate records and stable content and identifiers. Before each daily run it SHALL restore the latest selected targets and last persisted rebalance date from the selected namespace, and it SHALL fail closed rather than silently reset state when persisted state cannot be read.

#### Scenario: Same input is rerun
- **WHEN** an identical historical recommendation run is persisted ten times
- **THEN** its row count, IDs, and contents remain unchanged

#### Scenario: Process restarts between sessions
- **WHEN** a new daily process uses a store containing prior targets and a prior rebalance
- **THEN** it carries those targets and continues the same REB60 session count

#### Scenario: REB60 boundary
- **WHEN** the prior rebalance is session zero
- **THEN** session 59 does not rebalance, session 60 rebalances, and a persisted session-60 rebalance becomes the next anchor

### Requirement: Observation execution is isolated from production
The daily runtime SHALL support an observation mode that uses the frozen canonical strategy while production eligibility is blocked. Observation results MUST be labeled `OBSERVATION`, MUST persist only in the observation namespace when writing is requested, MUST NOT modify production recommendations or historical closure evidence, and MUST keep broker submission disabled. Dry-run and validate-only execution MUST NOT mutate any recommendation namespace.

#### Scenario: Blocked strategy runs in observation mode
- **WHEN** current evidence is not production eligible and observation mode is requested
- **THEN** the canonical daily result is produced with `OBSERVATION` status and no production approval is asserted

#### Scenario: Observation write is repeated
- **WHEN** the same observation input is written ten times
- **THEN** the observation ledger remains idempotent and production and closure paths remain unchanged

#### Scenario: Observation dry-run
- **WHEN** observation is combined with dry-run or validate-only
- **THEN** the result is computed without mutating observation, production, or closure artifacts

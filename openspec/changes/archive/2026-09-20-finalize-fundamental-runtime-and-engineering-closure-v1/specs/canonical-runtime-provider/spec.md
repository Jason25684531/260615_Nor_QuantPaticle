## ADDED Requirements

### Requirement: Canonical provider derives the frozen factor inputs from PIT records
The system SHALL expose `CanonicalFundamentalRuntimeProvider` that constructs each requested snapshot directly from canonical PIT fundamental records, the security universe, and the trading calendar.  It SHALL use G2 Operating Income YoY and G3 EPS YoY exactly as frozen, retain period-end and available-date lineage, and SHALL fail closed when either required source metric is absent.

#### Scenario: PIT-valid source data is available
- **WHEN** an as-of session has canonical operating-income and EPS records with `available_date <= as_of_date`
- **THEN** the provider emits non-empty eligible rows with raw and normalized G2/G3, score, rank, selection, target weight, REB60 status, source, input SHA, and provider SHA

#### Scenario: Future or missing source data is encountered
- **WHEN** a required record is future-dated or no eligible operating-income/EPS lineage exists
- **THEN** the provider rejects the snapshot and emits no recommendation

### Requirement: Provider preserves frozen strategy semantics
The provider SHALL not change the frozen strategy ID, fingerprint, G2/G3 definitions, equal factor weights, Top5, score-weighted target logic, PIT semantics, or 60-session rebalance semantics.

#### Scenario: Frozen contract is checked
- **WHEN** a runtime validation runs
- **THEN** immutable identity and strategy parameter checks report no strategy, factor, or selection change

# Real Three-Way Parity

## Purpose

Ensure the three named Fundamental runtime consumers agree over canonical data.

## Requirements

### Requirement: Independent real runtime snapshots are compared
The system SHALL predeclare at least twelve historical fixture sessions spanning multiple years, factor-update boundaries, rebalance and non-rebalance dates, and universe changes. Research, Shadow, and Production SHALL invoke separate adapters and SHALL not consume another path's persisted snapshot output.

#### Scenario: All three paths have source rows
- **WHEN** every fixed fixture session is evaluated against the canonical runtime input
- **THEN** research, shadow, and production snapshot files contain non-empty rows with the required factor, ranking, target, and REB60 fields

#### Scenario: Any path is empty or unavailable
- **WHEN** a source runtime has no rows or cannot be invoked
- **THEN** real snapshot parity is BLOCKED or FAIL and cannot be reported as PASS

### Requirement: Parity requires zero decision divergence
The system SHALL compare universe eligibility, G2/G3 raw and normalized values, score, rank, selection, target weight, and REB60 using the existing frozen tolerance. PASS SHALL require zero selection, target, and REB60 mismatches.

#### Scenario: Outputs agree
- **WHEN** all compared records agree within the frozen tolerance
- **THEN** the report records zero selection, target, and REB60 mismatches and marks real parity PASS

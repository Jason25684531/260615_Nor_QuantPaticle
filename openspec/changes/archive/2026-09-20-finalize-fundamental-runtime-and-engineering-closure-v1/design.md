## Context

The frozen `fundamental_g2g3_top5_reb60_score_weighted_v1` contract requires G2 Operating Income YoY and G3 EPS YoY.  The existing provider validates an input-only factor-output schema and correctly fails closed because the checked-in PIT records contain no `operating_income` metric.  Existing synthetic tests prove seam behavior but cannot prove real repository-data parity.

## Goals / Non-Goals

**Goals:**

- Materialize deterministic, PIT-safe G2/G3 rows only from canonical records, the registered universe, and market sessions.
- Invoke named Research, Shadow, and Production adapters independently while sharing the same factor, normalization, ranking, Top5, target, and REB60 logic.
- Persist isolated validation evidence and retain a blocked Fresh OOS promotion state.

**Non-Goals:**

- Changing factor formulas, weights, Top5, score-weighted targets, REB60 semantics, selection behavior, the frozen identity, or historical expected selections.
- Filling missing operating income with a proxy, inferring it from net income, accepting future data, enabling production, or submitting broker orders.

## Decisions

### Canonical records are the sole factor input

The provider SHALL compute G2 and G3 from versioned PIT records selected by `available_date <= as_of_date`, retaining each selected period/publication/availability lineage.  A missing G2 source is a hard error.  This is preferred to using the existing precomputed score artifacts because those artifacts lack enough filing lineage to be a canonical PIT runtime input.

### One shared pure runtime path

G2/G3 calculation, cross-sectional normalization, score, deterministic rank/tie-break, Top5, score weighting, and session-based REB60 are one shared implementation.  The three paths have separate invocations and output writers; they do not read one another's snapshot file.  This avoids three strategy implementations while preserving a meaningful adapter parity boundary.

### Fixed evidence before evaluation

The twelve fixture dates are declared before any output is generated.  The manifest binds dates and canonical input hashes.  Real parity requires all three non-empty outputs and zero field mismatches under the existing numeric tolerance.

### Safe closure is evidence-driven

The daily runner uses an isolated store for historical validation, has no broker adapter, and treats failed freshness/PIT/factor/provider/parity/eligibility gates as no-recommendation results.  Fresh OOS remains a promotion prerequisite, never an engineering workaround.

## Risks / Trade-offs

- [Canonical G2 records are absent] → Fail closed and report the exact missing metric; do not create a proxy or claim parity.
- [Historical dates have incomplete universe/market coverage] → Select dates from the canonical calendar only and reject missing snapshots.
- [Writing validation evidence changes active state] → Use a dedicated closure namespace and atomically replace only its deterministic artifact files.
- [Frozen output differs] → Diagnose the first shared-logic stage and fix an implementation error; never revise fixtures or tolerances to pass.

## Migration Plan

1. Add the missing official operating-income PIT records with source, period end, publication date, and available date, then validate their lineage.
2. Run the full closure validation in its isolated namespace and inspect all zero-mismatch gates.
3. Sync specifications and archive only after every engineering gate passes.
4. Roll back by restoring the previous code and deleting only the newly created isolated validation namespace; never alter active production state.

## Open Questions

- The current committed canonical store has no operating-income records.  A source-complete, independently auditable historical MOPS/TWSE operating-income PIT dataset is required before this Change can truthfully reach the requested PASS state.

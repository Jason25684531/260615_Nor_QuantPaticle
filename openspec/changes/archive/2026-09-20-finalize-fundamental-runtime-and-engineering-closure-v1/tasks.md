## 1. Canonical PIT runtime inputs

- [x] 1.1 Add source-complete official `operating_income` PIT records for every historical fixture, with period-end, publication-date, available-date, source, and revision lineage.
- [x] 1.2 Validate the canonical store contains PIT-valid G2 and G3 source records without proxying, changing factor definitions, or admitting future data.
- [x] 1.3 Implement and test the shared canonical G2/G3-to-target runtime directly from those records, universe, and trading calendar.

## 2. Real snapshot parity

- [x] 2.1 Lock a non-empty twelve-date fixture manifest and generate independent Research, Shadow, and Production parquet snapshots.
- [x] 2.2 Compare full factor, score, rank, selection, target, and REB60 output; repair the first shared root cause until all mismatch counts are zero.
- [x] 2.3 Verify historical frozen selection drift is zero and strategy immutability evidence is unchanged.

## 3. Safe daily validation

- [x] 3.1 Execute validate-only, dry-run, and an isolated historical write-path E2E; verify fail-closed behavior for all required gates.
- [x] 3.2 Run ten identical persistence writes and verify atomicity, natural-key uniqueness, stable rows, IDs, and content.
- [x] 3.3 Generate real Web and LINE fixtures from the E2E recommendations while retaining broker disabled and production enablement false.

## 4. Closure and verification

- [x] 4.1 Create final machine status, closure report, future OOS gate, and required documentation consistent with actual validation evidence.
- [x] 4.2 Run pytest, ruff, strict Change/all-spec validation, RC1, frozen-hash verification, real parity, daily E2E, independent review, and Ponytail review.
- [x] 4.3 Sync the four specs, archive this only active Change, revalidate all specs, then create and verify the requested commit and release tag if all gates pass.

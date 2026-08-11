# Fundamental Coverage Report

## Live Pipeline Status

- status: FAILED
- source: MOPS `ajax_t163sb05` (balance sheet)
- reason: official-source read timeout during the 2330 / ROC 112 acceptance run
- artifact action: no fundamental parquet was generated or overwritten

## Known Limitations

- The 2013–2025 backfill has not completed, so coverage metrics are unavailable.
- Financial statements require doc.twse publication-date coverage.
- Monthly revenue is PERIOD_ONLY and uses the statutory next-month-10th policy.
- Valuation snapshots remain SNAPSHOT_ONLY and are excluded from historical matrices.

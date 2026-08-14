# Fundamental Coverage Report

## Live Pipeline Status

- status: PASS
- raw_cache_hits: 5748
- raw_cache_misses: 43

## Coverage

- financial_statement_coverage: 0.4348 (30/69)
- publication_date_coverage: 0.4348 (30/69)
- monthly_revenue_coverage: 0.9855 (68/69)
- valuation_pit_coverage: 1.0000 (69/69)
- pit_ready_ratio: 1.0000 (69/69)

## PIT Status

- FULL_PIT: 5277500
- PERIOD_ONLY: 166975
- PUBLICATION_DATE_AWARE: 1261

## 2013–2015 Publication-Date Coverage

- 2013: financial_statement_count=3280, publication_date_found=52, publication_date_missing=3228, coverage_ratio=0.0159, downgraded_count=0, excluded_count=3228
- 2014: financial_statement_count=3368, publication_date_found=41, publication_date_missing=3327, coverage_ratio=0.0122, downgraded_count=0, excluded_count=3327
- 2015: financial_statement_count=3486, publication_date_found=25, publication_date_missing=3461, coverage_ratio=0.0072, downgraded_count=0, excluded_count=3461

## Historical Acceptance: 2023-05-15 / 2330

- D2 eligibility: false (2330 has no canonical OHLCV row on this date), so it is correctly excluded from `fundamental_matrix`; this acceptance reads its source-backed PIT records directly.
- 2023 Q1 financial Revenue: 508632973; EPS: 7.98; ROE: 0.079356.
- Financial period_end: 2023-03-31; doc.twse publication_date: 2023-05-12; available_date: 2023-05-15; source: `mops+doc.twse`; pit_status: `PUBLICATION_DATE_AWARE`.
- TWSE BWIBBU_d on 2023-05-15: PE 12.59; PB 4.17; dividend yield 2.22; referenced report period 112/1; pit_status: `FULL_PIT`.

## Failed Requests

- None

## Known Limitations

- Monthly revenue is PERIOD_ONLY, using the statutory next-month-10th policy.
- Financial statements require doc.twse publication-date coverage and may be excluded when absent.
- 2013–2015 publication-date completeness and financial-industry mappings require coverage review.
- Valuation snapshots remain SNAPSHOT_ONLY and are excluded from historical matrices.
- Records whose publication_date precedes the OHLCV trading calendar's earliest date have no real next-trading-day to derive available_date from; they are excluded from fundamental_pit rather than fabricated (see Publication-Date Coverage above for raw statement/publication counts in that range).

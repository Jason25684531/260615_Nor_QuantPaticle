# Fresh OOS Promotion

Fresh OOS is not an engineering gate. It requires untouched data collected
after the strategy freeze date, sufficient historical sessions, months, and
rebalances, and the frozen validation command. Until those gates pass,
`FRESH_OOS_STATUS=INSUFFICIENT_DATA`, production eligibility is blocked for
`INSUFFICIENT_OOS`, production enablement remains FALSE, and broker submission
remains DISABLED. No factor research or strategy re-selection is permitted.

The executable contract is
`data/project-closure/fundamental-runtime-engineering-complete-v1/future_oos_gate.json`:
it fixes the 2026-07-28 strategy freeze, the 2026-08-31 contamination boundary,
minimum 180 sessions, 9 months, and 3 rebalances. Once those conditions exist,
run `python run_fresh_oos_validation_v1.py --download`; only a PASS followed by
explicit human approval can progress from `PRODUCTION_VALIDATED` to
`PRODUCTION_ENABLED`.

# Production Runbook

Validate the frozen runtime without active recommendations:

```powershell
python run_daily_fundamental_production.py --validate-only
python run_daily_fundamental_production.py --dry-run
```

With no `--as-of-date`, the command uses the latest completed session present
in canonical OHLCV data. Historical closure evidence is regenerated only by:

```powershell
python run_daily_fundamental_production.py --historical-validation
```

Inspect a historical date without a write:

```powershell
python run_daily_fundamental_production.py --as-of-date 2025-06-30
```

Run or persist a broker-free observation while production eligibility remains
blocked. Observation state is isolated under
`data/observation/fundamental-recommendations/`:

```powershell
python run_daily_fundamental_production.py --observation --dry-run
python run_daily_fundamental_production.py --observation --write-recommendations
```

The write command is intentionally fail-closed until Fresh OOS passes and a
human explicitly enables production:

```powershell
python run_daily_fundamental_production.py --as-of-date YYYY-MM-DD --write-recommendations
```

Final runtime evidence is under `data/research/fundamental-production-final-v1/`.
The safe historical E2E store is `isolated-validation/daily_recommendations.parquet`;
its health, parity, Web, and LINE fixtures are in the same namespace. The final
machine status and OOS gate are under
`data/project-closure/fundamental-runtime-engineering-complete-v1/`.

Stale data, PIT failure, factor-health failure, provider failure, parity failure,
or ineligible production evidence writes no recommendation. Investigate the
reported gate, repair canonical data or runtime code, then rerun validate-only;
do not bypass a gate. Roll back by keeping `PRODUCTION_ENABLE_FLAG=FALSE` and
removing only an isolated validation namespace. Fresh OOS promotion requires the
recorded future-data gate to pass, followed by explicit human approval; it never
reopens factor research or sends broker orders.

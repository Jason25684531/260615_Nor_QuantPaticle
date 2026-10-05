"""Audit historical universe survivorship without fabricating delisted members."""

from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from twse_factor_lab.reporting.performance_adapter import load_frozen_identity

NAMESPACE = Path("data/research/fundamental-production-final-v1")
OUTPUT = NAMESPACE / "survivorship-bias"
START = pd.Timestamp("2021-01-04")
END = pd.Timestamp("2025-12-31")
EVALUATION_ID = "survivorship_corrected_historical_replay_v1"
UNIVERSE_ID = "twse_point_in_time_historical_membership_v1"
POLICY_VERSION = "historical_membership_source_policy_v1"


def membership_eligible(
    trade_date: Any, effective_from: Any, effective_to: Any = None
) -> bool:
    """Apply the frozen inclusive effective-date membership semantics."""

    date = pd.Timestamp(trade_date).normalize()
    start = pd.Timestamp(effective_from).normalize()
    if date < start:
        return False
    return effective_to is None or pd.isna(effective_to) or date <= pd.Timestamp(effective_to).normalize()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _paths(root: Path) -> dict[str, Path]:
    return {
        "current_membership": root / "data/processed/universe.parquet",
        "research_universe": root / "data/processed/research_universe.parquet",
        "security_master": root / "data/runtime/shadow-s3-v1/market_security_master.parquet",
        "security_status": root / "data/research/canonical-hybrid-revalidation-v1/hybrid_security_status.parquet",
        "ohlcv": root / "data/processed/ohlcv.parquet",
        "close_matrix": root / "data/processed/close_matrix.parquet",
        "fundamental_pit": root / "data/processed/fundamental_pit.parquet",
        "fundamental_v2": root / "data/processed/fundamental_pit_v2/fundamental_records.parquet",
        "backtest_returns": root / NAMESPACE / "historical-evidence/canonical_backtest_returns.parquet",
    }


def _source_inventory(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    existing = {name: path.exists() for name, path in paths.items()}
    inventory = {
        "current_membership_source": "data/processed/universe.parquet",
        "listing_date_source": "data/processed/universe.parquet",
        "delisting_source": None,
        "historical_membership_source": None,
        "security_status_reference": "data/research/canonical-hybrid-revalidation-v1/hybrid_security_status.parquet",
        "price_sources": ["data/processed/ohlcv.parquet", "data/processed/close_matrix.parquet"],
        "fundamental_sources": ["data/processed/fundamental_pit.parquet", "data/processed/fundamental_pit_v2/fundamental_records.parquet"],
        "available_delisted_tickers": [],
        "source_exists": existing,
        "limitations": [
            "No authoritative historical TWSE/TPEX delisting membership interval source is present in the repository.",
            "market_security_master and hybrid_security_status are current/status reconciliation references, not historical membership history.",
            "Current-listed-only universe cannot be backcast into delisted membership.",
            "Delisting execution semantics and ticker identity transitions are not available for a corrected replay.",
        ],
    }
    return inventory


def _known_membership(universe: pd.DataFrame) -> pd.DataFrame:
    frame = universe.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["effective_from"] = pd.to_datetime(frame["listed_date"], errors="coerce").dt.normalize()
    frame["effective_to"] = pd.NaT
    frame["membership_status"] = "CURRENT_LISTED_ONLY_REFERENCE"
    frame["source"] = "data/processed/universe.parquet"
    frame["source_record_id"] = frame["ticker"]
    frame["source_asof"] = pd.NaT
    frame["reason"] = "Current-listed-only source; effective_to unknown; not corrected historical membership"
    return frame[["ticker", "market", "effective_from", "effective_to", "membership_status", "source", "source_record_id", "source_asof", "reason"]].sort_values("ticker").reset_index(drop=True)


def _coverage(root: Path, members: pd.DataFrame, paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = pd.to_datetime(pd.read_parquet(paths["backtest_returns"])["date"]).dt.normalize()
    close = pd.read_parquet(paths["close_matrix"])
    close.index = pd.to_datetime(close.index).normalize()
    close = close.reindex(dates)
    price_tickers = set(close.columns.astype(str))
    v2 = pd.read_parquet(paths["fundamental_v2"], columns=["ticker", "metric", "available_date"])
    v2["ticker"] = v2["ticker"].astype(str)
    v2["available_date"] = pd.to_datetime(v2["available_date"], errors="coerce").dt.normalize()
    available = {
        metric: v2.loc[v2["metric"].eq(metric)].groupby("ticker")["available_date"].min()
        for metric in ("operating_income", "eps")
    }
    rows: list[dict[str, Any]] = []
    for date in dates:
        known = members.loc[members["effective_from"].le(date)]
        tickers = set(known["ticker"])
        price_valid = tickers.intersection(price_tickers)
        if date in close.index:
            price_valid = {ticker for ticker in price_valid if pd.notna(close.loc[date, ticker])}
        g2 = {ticker for ticker, first in available["operating_income"].items() if pd.notna(first) and first <= date}
        g3 = {ticker for ticker, first in available["eps"].items() if pd.notna(first) and first <= date}
        fundamental_valid = tickers.intersection(g2).intersection(g3)
        replayable = price_valid.intersection(fundamental_valid)
        rows.append({
            "trade_date": date,
            "historical_member_count": len(known),
            "replayable_member_count": len(replayable),
            "eligible_member_count": len(known),
            "missing_price_count": len(tickers - price_valid),
            "missing_fundamental_count": len(tickers - fundamental_valid),
            "coverage_ratio": len(replayable) / len(known) if len(known) else 0.0,
        })
    coverage = pd.DataFrame(rows)
    price_ratio = float(coverage["historical_member_count"].sub(coverage["missing_price_count"]).sum() / coverage["historical_member_count"].sum())
    fundamental_ratio = float(coverage["historical_member_count"].sub(coverage["missing_fundamental_count"]).sum() / coverage["historical_member_count"].sum())
    report = {
        "historical_member_count": int(len(members)),
        "current_survivor_count": int(len(members)),
        "historically_delisted_member_count": 0,
        "membership_dates_covered": {"start": str(members["effective_from"].min().date()), "end": str(END.date())},
        "unknown_effective_from_count": int(members["effective_from"].isna().sum()),
        "unknown_effective_to_count": int(members["effective_to"].isna().sum()),
        "membership_source_coverage": 0.0,
        "ohlcv_coverage": price_ratio,
        "g2_coverage": fundamental_ratio,
        "g3_coverage": fundamental_ratio,
        "fully_replayable_ticker_count": 0,
        "non_replayable_ticker_count": int(len(members)),
        "coverage_by_date_start": str(dates.min().date()),
        "coverage_by_date_end": str(dates.max().date()),
        "resolution_status": "BLOCKED",
    }
    return coverage, report


def _comparison(root: Path) -> pd.DataFrame:
    summary = json.loads((root / NAMESPACE / "performance/performance_summary.json").read_text(encoding="utf-8"))
    metrics = summary["periods"]["backtest"]["metrics"]
    rows = []
    for name in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "annualized_volatility", "turnover", "transaction_cost"):
        rows.append({"metric": name, "original_current_listed_only": metrics.get(name, {}).get("value"), "survivorship_corrected": None, "difference": None, "status": "BLOCKED"})
    returns = pd.read_parquet(root / NAMESPACE / "historical-evidence/canonical_backtest_returns.parquet")
    returns["date"] = pd.to_datetime(returns["date"])
    for year, values in returns.groupby(returns["date"].dt.year)["daily_return"]:
        rows.append({"metric": f"{year}_return", "original_current_listed_only": float((1 + values).prod() - 1), "survivorship_corrected": None, "difference": None, "status": "BLOCKED"})
    return pd.DataFrame(rows)


def _update_final_report(root: Path, coverage: dict[str, Any]) -> None:
    path = root / NAMESPACE / "final_strategy_validation_report.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Final Strategy Validation Report\n"
    section = """## Survivorship Bias Evaluation

Original Universe = `current-listed-only`

Historical Membership Status = BLOCKED

Delisted Membership Coverage = 0.0 (no authoritative historical delisting source)

Corrected Replay Status = BLOCKED

Survivorship Bias Status = SURVIVORSHIP_BIAS_NOT_FULLY_RESOLVED

Return Impact = NOT_AVAILABLE

CAGR Impact = NOT_AVAILABLE

MDD Impact = NOT_AVAILABLE

Historical Impact Classification = NOT_PREDECLARED

Project Historical Research Closed = YES

No Further 2021-2025 Parameter Tuning = YES

"""
    if "## Survivorship Bias Evaluation" in text:
        text = text.split("## Survivorship Bias Evaluation", 1)[0].rstrip() + "\n\n"
    marker = "## Production Gate"
    if marker in text:
        text = text.replace(marker, section + marker, 1)
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def generate_survivorship_bias_diagnostic(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    identity = load_frozen_identity(root)
    if identity.strategy_id != "fundamental_g2g3_top5_reb60_score_weighted_v1" or identity.strategy_fingerprint != "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f":
        raise ValueError("FROZEN_STRATEGY_CHANGED")
    paths = _paths(root)
    OUTPUT_PATH = root / OUTPUT
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    inventory = _source_inventory(root, paths)
    policy = {
        "policy_version": POLICY_VERSION,
        "source_priority": ["repository_authoritative_historical_membership", "official_twse_listing_delisting_history", "official_exchange_security_status_history", "other_reproducible_authoritative_source"],
        "selected_source": None,
        "no_fabrication": ["no_current_list_backcasting", "no_guessed_effective_to", "no_last_price_as_delisting_date", "no_synthetic_members"],
        "effective_from_semantics": "inclusive first legally tradable membership date",
        "effective_to_semantics": "inclusive last legal membership date or authoritative delisting effective date; null only when known ongoing",
        "resolution_gate": ["authoritative_membership_intervals", "OHLCV_for_all_replay_candidates", "PIT_G2_and_G3_for_all_replay_candidates", "defined_delisting_execution_semantics", "unambiguous_security_identity"],
        "classification_policy": "NOT_PREDECLARED when no approved materiality threshold exists",
    }
    policy["policy_hash"] = _sha_payload(policy)
    _write_json(OUTPUT_PATH / "historical_universe_source_inventory.json", inventory)
    _write_json(OUTPUT_PATH / "historical_membership_source_policy.json", policy)
    universe = pd.read_parquet(paths["current_membership"])
    members = _known_membership(universe)
    members.to_parquet(OUTPUT_PATH / "historical_universe_membership.parquet", index=False)
    coverage_by_date, coverage = _coverage(root, members, paths)
    coverage_by_date.to_parquet(OUTPUT_PATH / "historical_membership_coverage_by_date.parquet", index=False)
    _write_json(OUTPUT_PATH / "historical_membership_coverage_report.json", coverage)
    spec = {
        "universe_id": UNIVERSE_ID,
        "evaluation_identity": EVALUATION_ID,
        "membership_source": None,
        "membership_basis": "current_listed_only_reference",
        "effective_date_semantics": policy["effective_to_semantics"],
        "eligibility_rules": "existing listing-age, OHLCV, liquidity, and PIT factor rules; unchanged",
        "coverage_status": "BLOCKED",
        "source_hashes": {key: _sha_file(path) for key, path in paths.items() if path.exists()},
        "coverage": coverage,
        "universe_fingerprint": _sha_payload({"universe_id": UNIVERSE_ID, "source_hashes": {key: _sha_file(path) for key, path in paths.items() if path.exists()}, "policy_hash": policy["policy_hash"]}),
    }
    _write_json(OUTPUT_PATH / "historical_universe_spec.json", spec)
    selection_columns = ["rebalance_date", "original_top5", "corrected_top5", "overlap_count", "added_historical_tickers", "removed_original_tickers"]
    pd.DataFrame(columns=selection_columns).to_parquet(OUTPUT_PATH / "survivorship_selection_audit.parquet", index=False)
    comparison = _comparison(root)
    comparison.to_csv(OUTPUT_PATH / "survivorship_bias_comparison.csv", index=False)
    validation = {
        "status": "PASS",
        "resolution_status": "BLOCKED",
        "decision": "SURVIVORSHIP_BIAS_RESOLUTION_BLOCKED",
        "reason": "AUTHORITATIVE_HISTORICAL_MEMBERSHIP_DATA_UNAVAILABLE",
        "evaluation_identity": EVALUATION_ID,
        "universe_id": UNIVERSE_ID,
        "strategy_id": identity.strategy_id,
        "strategy_fingerprint": identity.strategy_fingerprint,
        "historical_window": {"start": str(START.date()), "end": str(END.date())},
        "historical_impact_classification": "NOT_PREDECLARED",
        "original_historical_evidence": "HISTORICALLY_SUPPORTIVE",
        "survivorship_corrected_evidence": "BLOCKED",
        "benchmark": {"benchmark_id": "twii_taiex_broad_market_price_index_v1", "status": "AVAILABLE", "return_convention": "DEFINITION_DIFFERENCE"},
        "risk_overlay_route": "CLOSED_NOT_ADOPTED",
        "fresh_oos": "INSUFFICIENT_DATA",
        "production_gate": "BLOCKED",
        "production_ready": False,
        "remaining_blockers": ["AUTHORITATIVE_HISTORICAL_MEMBERSHIP_DATA", "FRESH_OOS"],
    }
    _write_json(OUTPUT_PATH / "survivorship_bias_validation.json", validation)
    report = f"""# Survivorship Bias Validation Report

## 1. Existing Limitation

Current Universe = `current-listed-only`

## 2. Historical Membership Source

Source = UNAVAILABLE
Authority = No authoritative historical TWSE/TPEX delisting interval source in repository
Coverage = BLOCKED

## 3. Membership Model

effective_from = inclusive first legally tradable membership date
effective_to = inclusive authoritative last membership date; unknown is not inferred
eligibility rules = existing listing, price, liquidity, and PIT G2/G3 rules unchanged

## 4. Data Coverage

Historical Members = {coverage['historical_member_count']}
Delisted Members = 0 known (source unavailable)
OHLCV Coverage = {coverage['ohlcv_coverage']:.6f}
G2 Coverage = {coverage['g2_coverage']:.6f}
G3 Coverage = {coverage['g3_coverage']:.6f}

## 5. Resolution Status

BLOCKED

## 6. Original vs Corrected

Corrected replay is not emitted because authoritative membership data is unavailable. See `survivorship_bias_comparison.csv` for original values and explicit blocked nulls.

## 7. Selection Impact

Affected Rebalances = NOT_AVAILABLE
Average Top5 Overlap = NOT_AVAILABLE
Delisted Names Selected = NOT_AVAILABLE

## 8. Benchmark Context

Canonical benchmark remains TAIEX (`twii_taiex_broad_market_price_index_v1`); no corrected-vs-benchmark metrics are computed.

## 9. Survivorship Bias Impact

Return Difference = NOT_AVAILABLE
CAGR Difference = NOT_AVAILABLE
Sharpe Difference = NOT_AVAILABLE
MDD Difference = NOT_AVAILABLE

## 10. Limitations

- No authoritative historical membership/delisting intervals.
- Current-listed-only membership cannot be backcast.
- Delisted OHLCV, PIT G2/G3 coverage, ticker identity mapping, and delisting execution semantics cannot be validated for omitted names.
- Historical impact materiality threshold is not predeclared (`NOT_PREDECLARED`).

## 11. Governance

Base Fingerprint = UNCHANGED
Original Historical Evidence = HISTORICALLY_SUPPORTIVE
Fresh OOS = INSUFFICIENT_DATA
Production = BLOCKED
Risk Overlay Route = CLOSED_NOT_ADOPTED

## 12. Final Decision

SURVIVORSHIP_BIAS_RESOLUTION_BLOCKED

Project historical research is closed. No further 2021-2025 parameter tuning will be performed.
"""
    (OUTPUT_PATH / "survivorship_bias_validation_report.md").write_text(report, encoding="utf-8")
    artifact_paths = [path for path in OUTPUT_PATH.iterdir() if path.is_file() and path.name != "historical_universe_manifest.json"]
    manifest = {
        "manifest_version": "historical-universe-survivorship-v1",
        "status": "BLOCKED",
        "resolution_status": "BLOCKED",
        "evaluation_identity": EVALUATION_ID,
        "universe_id": UNIVERSE_ID,
        "strategy_id": identity.strategy_id,
        "strategy_fingerprint": identity.strategy_fingerprint,
        "policy_hash": policy["policy_hash"],
        "universe_fingerprint": spec["universe_fingerprint"],
        "corrected_replay_generated": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_sha256": {path.name: _sha_file(path) for path in artifact_paths},
    }
    _write_json(OUTPUT_PATH / "historical_universe_manifest.json", manifest)
    _update_final_report(root, coverage)
    return validation


__all__ = ["generate_survivorship_bias_diagnostic", "membership_eligible"]

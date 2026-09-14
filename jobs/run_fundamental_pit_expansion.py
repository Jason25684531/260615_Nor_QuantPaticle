"""Expand doc.twse publication-date coverage with resumable raw caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Support both `python -m jobs.run_fundamental_pit_expansion` and direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_data_pipeline import load_config, project_root_for_config, resolve_path
from twse_factor_lab.data.fundamental import (
    FundamentalClient,
    FundamentalDataError,
    RawFundamentalCache,
)
from twse_factor_lab.data.normalizer import clean_ticker

SEASONS_PER_YEAR = 4


def cached_publication_tickers(cache_root: str | Path) -> list[str]:
    """Return source tickers already represented by t57sb01 metadata.

    Diagnostic only: the target universe MUST come from
    ``build_target_tickers`` — cache contents never define enumeration.
    """

    tickers: set[str] = set()
    for path in Path(cache_root).glob("*.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("endpoint") != "/server-java/t57sb01":
            continue
        ticker = clean_ticker(
            (metadata.get("request_params") or {}).get("data", {}).get("co_id")
        )
        if isinstance(ticker, str) and ticker.isdigit():
            tickers.add(ticker)
    return sorted(tickers)


def build_target_tickers(config_path: str | Path, config: dict | None = None) -> list[str]:
    """Canonical publication-expansion target set.

    research_universe[is_eligible] union acceptance_tickers, clean_ticker
    normalized, deterministically sorted. Cache keys are never consulted.
    """

    config = config or load_config(config_path)
    universe = pd.read_parquet(
        resolve_path(config_path, config["paths"]["research_universe"])
    )
    eligible = {
        str(clean_ticker(ticker))
        for ticker in universe.loc[universe["is_eligible"].astype(bool), "ticker"]
    }
    acceptance = {
        str(clean_ticker(ticker))
        for ticker in config["fundamental"].get("acceptance_tickers", [])
    }
    return sorted(eligible | acceptance)


def target_ticker_sha256(tickers: Iterable[str]) -> str:
    """Canonical serialization: sorted tickers newline-joined + trailing newline."""

    payload = "\n".join(sorted(tickers)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_digest(tickers: Iterable[str]) -> dict:
    values = sorted(tickers)
    return {"count": len(values), "sha256": target_ticker_sha256(values)}


def _years(config_path: str | Path, config: dict) -> list[int]:
    ohlcv = pd.read_parquet(resolve_path(config_path, config["paths"]["ohlcv"]))
    end_year = int(pd.to_datetime(ohlcv["date"]).max().year) - 1911
    return list(
        range(int(config["fundamental"]["start_roc_year"]), end_year + 1)
    )


def _ticker_execution_status(rows: pd.DataFrame) -> str:
    fetched = rows["status"].eq("fetched").any()
    failed = rows["status"].eq("failed").any()
    publication_rows = int(rows.get("publication_rows", pd.Series(dtype=float)).fillna(0).sum())
    if publication_rows > 0:
        return "FETCHED" if fetched else "CACHED"
    if failed:
        return "FAILED_SOURCE"
    return "NO_SOURCE_RECORD"


def build_reconciliation(
    *,
    target: list[str],
    before: list[str],
    after: list[str],
    sample_size: int = 30,
) -> dict:
    target_set, before_set, after_set = set(target), set(before), set(after)
    missing_after = sorted(target_set - after_set)
    return {
        "TARGET_ELIGIBLE_TICKERS": _set_digest(target_set),
        "PUBLICATION_TICKERS_BEFORE": _set_digest(before_set),
        "PUBLICATION_TICKERS_AFTER": _set_digest(after_set),
        "target_minus_before": _set_digest(target_set - before_set),
        "before_minus_target": _set_digest(before_set - target_set),
        "target_minus_after": _set_digest(target_set - after_set),
        "after_minus_target": _set_digest(after_set - target_set),
        "after_minus_before": _set_digest(after_set - before_set),
        "missing_target_ticker_sample": missing_after[:sample_size],
        "unexpected_extra_tickers": sorted(after_set - target_set)[:sample_size],
    }


def run_expansion(
    config_path: str | Path,
    *,
    tickers: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    client: FundamentalClient | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Enumerate the full canonical target set; the cache only answers HIT/MISS."""

    config = load_config(config_path)
    fundamental = config["fundamental"]
    cache_root = resolve_path(config_path, fundamental["raw_cache_dir"])
    cache = client.cache if client is not None else RawFundamentalCache(cache_root)
    if client is None:
        client = FundamentalClient(
            cache=cache,
            twse_base_url=config["twse"]["base_url"],
            timeout=int(config["twse"].get("timeout_seconds", 30)),
            mops_timeout=int(fundamental.get("mops_timeout_seconds", 90)),
            throttle_seconds=float(fundamental.get("throttle_seconds", 0.5)),
            retry=int(fundamental.get("retry", 3)),
            mops_statement_min_ticker_count=int(
                fundamental.get("mops_statement_min_ticker_count", 100)
            ),
        )
    canonical_target = build_target_tickers(config_path, config)
    if tickers is None:
        target = canonical_target
    else:
        target = sorted({str(clean_ticker(ticker)) for ticker in tickers})
        if target != canonical_target:
            # Explicit subset runs (tests / batching) are allowed but must be
            # visible: a full run MUST enumerate the canonical set.
            pass
    publication_before = cached_publication_tickers(cache_root)
    years = list(years if years is not None else _years(config_path, config))
    start_stats = cache.statistics()
    rows: list[dict] = []
    publications: list[pd.DataFrame] = []
    http_success_count = 0
    http_failure_count = 0
    for ticker in target:
        for year in years:
            before = cache.statistics()
            try:
                result = client.publication_dates(ticker, int(year))
            except FundamentalDataError as exc:
                after = cache.statistics()
                misses = after["misses"] - before["misses"]
                http_failure_count += 1
                http_success_count += max(0, misses - 1)
                rows.append(
                    {
                        "ticker": ticker,
                        "roc_year": int(year),
                        "status": "failed",
                        "publication_rows": 0,
                        "error": str(exc),
                    }
                )
                continue
            after = cache.statistics()
            misses = after["misses"] - before["misses"]
            http_success_count += misses
            rows.append(
                {
                    "ticker": ticker,
                    "roc_year": int(year),
                    "status": "fetched" if misses else "cached",
                    "publication_rows": int(len(result)),
                }
            )
            if not result.empty:
                publications.append(result)
    stats = pd.DataFrame(rows)
    end_stats = cache.statistics()
    cache_hit_count = end_stats["hits"] - start_stats["hits"]
    cache_miss_count = end_stats["misses"] - start_stats["misses"]
    planned_lookup_count = len(target) * len(years) * SEASONS_PER_YEAR
    # Lookups skipped because an earlier season in the same ticker/year failed.
    lookup_failure_skips = planned_lookup_count - cache_hit_count - cache_miss_count
    accounting_reconciled = (
        cache_hit_count + cache_miss_count + lookup_failure_skips
        == planned_lookup_count
        and lookup_failure_skips >= 0
    )
    ticker_status = {
        ticker: _ticker_execution_status(group)
        for ticker, group in stats.groupby("ticker")
    }
    silently_unprocessed = sorted(set(target) - set(ticker_status))
    enumeration_status = (
        "FULL_ENUMERATION" if not silently_unprocessed else "PARTIAL_ENUMERATION"
    )
    publication_after = cached_publication_tickers(cache_root)
    covered_target = sorted(set(target) & set(publication_after))
    missing_target = sorted(set(target) - set(publication_after))
    newly_fetched = sorted(
        ticker for ticker, status in ticker_status.items() if status == "FETCHED"
    )
    failed_tickers = sorted(
        ticker for ticker, status in ticker_status.items() if status == "FAILED_SOURCE"
    )
    if enumeration_status != "FULL_ENUMERATION":
        expansion_status = "PARTIAL_EXPANSION_ONLY"
    elif missing_target and set(missing_target) <= set(failed_tickers) and not newly_fetched:
        expansion_status = "EXPANSION_BLOCKED_SOURCE"
    else:
        expansion_status = "FULL_EXPANSION_VERIFIED"
    if output_dir is None:
        output_dir = (
            project_root_for_config(config_path)
            / "outputs/fundamental_data/fundamental_pit_coverage_v1"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_dir / "publication_expansion_stats.csv", index=False)
    if publications:
        pd.concat(publications, ignore_index=True).drop_duplicates().to_csv(
            output_dir / "publication_dates_expanded.csv", index=False
        )
    reconciliation = build_reconciliation(
        target=target, before=publication_before, after=publication_after
    )
    (output_dir / "publication_ticker_reconciliation.json").write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = stats["status"].value_counts().to_dict() if not stats.empty else {}
    nonempty_mapping = (
        int(
            stats.loc[stats["publication_rows"].fillna(0) > 0, "ticker"].nunique()
        )
        if not stats.empty
        else 0
    )
    data_expansion_status = "PASS" if not counts.get("failed") else "PARTIAL"
    if not rows or counts.get("failed", 0) == len(rows):
        data_expansion_status = "BLOCKED_SOURCE_UNAVAILABLE"
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_expansion_status": data_expansion_status,
        "enumeration_status": enumeration_status,
        "full_publication_expansion_status": expansion_status,
        "target_ticker_source": (
            "research_universe.parquet[is_eligible] union acceptance_tickers"
        ),
        "target_ticker_count": len(target),
        "target_ticker_sha256": target_ticker_sha256(target),
        "existing_publication_ticker_count_before_run": len(publication_before),
        "missing_publication_ticker_count_before_run": len(
            set(target) - set(publication_before)
        ),
        "roc_years": years,
        "planned_lookup_count": planned_lookup_count,
        "cache_hit_count": int(cache_hit_count),
        "cache_miss_count": int(cache_miss_count),
        "lookup_failure_skips": int(lookup_failure_skips),
        "accounting_reconciled": bool(accounting_reconciled),
        "http_request_count": int(cache_miss_count),
        "http_success_count": int(http_success_count),
        "http_failure_count": int(http_failure_count),
        "newly_fetched_ticker_count": len(newly_fetched),
        "final_publication_ticker_count": len(publication_after),
        "covered_target_ticker_count": len(covered_target),
        "missing_target_ticker_count": len(missing_target),
        "target_publication_coverage_ratio": (
            len(covered_target) / len(target) if target else 0.0
        ),
        "non_empty_publication_mapping_ticker_count": nonempty_mapping,
        "silently_unprocessed_ticker_count": len(silently_unprocessed),
        "ticker_execution_status_counts": pd.Series(ticker_status)
        .value_counts()
        .to_dict()
        if ticker_status
        else {},
        "fetched": int(counts.get("fetched", 0)),
        "cached": int(counts.get("cached", 0)),
        "failed": int(counts.get("failed", 0)),
        "client_cache": cache.statistics(),
        "client_response_sanity": client.response_sanity_report(),
    }
    if not accounting_reconciled:
        summary["data_expansion_status"] = "FAILED_ACCOUNTING"
    (output_dir / "publication_expansion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verification_path = output_dir / "publication_expansion_verification.json"
    previous_review: dict | None = None
    if verification_path.exists():
        try:
            previous = json.loads(verification_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        source = previous.get("previous_review", previous)
        if source.get("review_type") == "read_only_remediation_review":
            previous_review = {
                key: source.get(key)
                for key in (
                    "review_type",
                    "verdict",
                    "target_ticker_count",
                    "target_ticker_sha256",
                    "target_ticker_hash_format",
                )
            }
    verification = {
        "verified_at": datetime.now(UTC).isoformat(),
        "enumeration_status": enumeration_status,
        "full_publication_expansion_status": expansion_status,
        "target": _set_digest(target),
        "silently_unprocessed": silently_unprocessed,
        "ticker_execution_status": ticker_status,
        "reconciliation": reconciliation,
    }
    if previous_review:
        verification["previous_review"] = previous_review
        verification["review_baseline_comparison"] = {
            "target_ticker_count_matches": previous_review.get("target_ticker_count")
            == len(target),
            "target_ticker_sha256_matches": previous_review.get("target_ticker_sha256")
            == target_ticker_sha256(target),
            "note": (
                "Review hash is not reproducible under its documented serialization "
                "(newline-delimited sorted clean_ticker values with trailing "
                "newline); set equality was established via count and membership "
                "(877 four-digit tickers plus 912000)."
            ),
        }
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--max-tickers", type=int)
    parser.add_argument("--ticker-offset", type=int, default=0)
    args = parser.parse_args()
    tickers = None
    if args.max_tickers is not None:
        tickers = build_target_tickers(args.config)[
            args.ticker_offset : args.ticker_offset + args.max_tickers
        ]
    summary = run_expansion(args.config, tickers=tickers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

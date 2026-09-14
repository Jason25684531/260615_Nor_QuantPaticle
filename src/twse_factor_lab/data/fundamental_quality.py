"""Quality, lineage, coverage, and readiness helpers for PIT fundamentals."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.analysis.factor_gate import _coverage, _paired_observations
from twse_factor_lab.data.fundamental import PIT_COLUMNS, validate_pit_records
from twse_factor_lab.data.normalizer import clean_ticker

FUNDAMENTAL_METRICS = ("eps", "roe")
LINEAGE_COLUMNS = [
    "source_record_id",
    "revision_status",
    "source_components",
    "formula_version",
    "metric_semantics",
]


def _record_id(row: pd.Series, *, extra: str = "") -> str:
    values = [
        str(row.get(column, ""))
        for column in ("ticker", "metric", "period_end", "publication_date", "value")
    ]
    return hashlib.sha256("|".join(values + [extra]).encode()).hexdigest()


def build_ticker_mapping_report(
    source: pd.DataFrame | Iterable[object], *, ticker_column: str = "ticker"
) -> pd.DataFrame:
    """Map source identifiers once, quarantining invalid or ambiguous mappings."""

    if isinstance(source, pd.DataFrame):
        if "source_ticker" in source:
            values = source["source_ticker"]
        elif ticker_column in source:
            values = source[ticker_column]
        else:
            raise KeyError(f"Missing ticker column: {ticker_column}")
    else:
        values = pd.Series(list(source), dtype="object")
    report = pd.DataFrame({"source_ticker": values.drop_duplicates()})
    report["canonical_ticker"] = report["source_ticker"].map(clean_ticker)
    valid = report["canonical_ticker"].map(
        lambda value: isinstance(value, str) and value.isdigit() and len(value) == 4
    )
    report["mapping_status"] = np.where(valid, "MAPPED", "REJECTED")
    report["reason"] = np.where(valid, "clean_ticker", "invalid_ticker")
    duplicate = report.loc[valid].duplicated("canonical_ticker", keep=False)
    report.loc[duplicate.index[duplicate], "mapping_status"] = "QUARANTINED"
    report.loc[duplicate.index[duplicate], "reason"] = "duplicate_mapping"
    return report.reset_index(drop=True)


def normalize_fundamental_records(
    pit: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("eps", "net_income", "equity"),
) -> pd.DataFrame:
    """Create deterministic financial records and derived ROE lineage."""

    required = set(PIT_COLUMNS)
    missing = required - set(pit.columns)
    if missing:
        raise KeyError(f"Missing PIT columns: {sorted(missing)}")
    frame = pit.loc[pit["metric"].isin(metrics)].copy()
    if frame.empty:
        return pd.DataFrame(columns=[*PIT_COLUMNS, *LINEAGE_COLUMNS])
    frame["ticker"] = frame["ticker"].map(clean_ticker)
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["publication_date"] = pd.to_datetime(
        frame["publication_date"], errors="coerce"
    )
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "period_end", "publication_date", "value"])
    frame = frame.drop_duplicates(
        ["ticker", "metric", "period_end", "publication_date"], keep="last"
    ).copy()
    frame["source_record_id"] = frame.apply(_record_id, axis=1)
    frame["source_components"] = pd.NA
    frame["formula_version"] = pd.NA
    frame["metric_semantics"] = np.where(
        frame["metric"].eq("eps"), "cumulative-as-reported", pd.NA
    )

    income = frame[frame["metric"].eq("net_income")]
    equity = frame[frame["metric"].eq("equity")]
    if not income.empty and not equity.empty:
        keys = ["ticker", "period_end", "publication_date", "available_date"]
        roe = income.merge(
            equity,
            on=keys,
            suffixes=("_income", "_equity"),
            how="inner",
        )
        roe["value"] = pd.to_numeric(roe["value_income"], errors="coerce").div(
            pd.to_numeric(roe["value_equity"], errors="coerce").replace(0, np.nan)
        )
        roe = roe.dropna(subset=["value"])
        if not roe.empty:
            derived = pd.DataFrame(
                {
                    "ticker": roe["ticker"],
                    "metric": "roe",
                    "period_end": roe["period_end"],
                    "publication_date": roe["publication_date"],
                    "available_date": roe["available_date"],
                    "value": roe["value"],
                    "source": "derived_net_income_over_equity",
                    "pit_status": roe["pit_status_income"],
                    "source_components": [
                        json.dumps(
                            {
                                "net_income": income_id,
                                "equity": equity_id,
                            },
                            sort_keys=True,
                        )
                        for income_id, equity_id in zip(
                            roe["source_record_id_income"],
                            roe["source_record_id_equity"],
                            strict=True,
                        )
                    ],
                    "formula_version": "net_income_over_equity_v1",
                    "metric_semantics": "derived",
                }
            )
            derived["source_record_id"] = derived.apply(_record_id, axis=1)
            frame = pd.concat([frame, derived[frame.columns]], ignore_index=True)

    frame = frame.sort_values(
        ["ticker", "metric", "period_end", "publication_date"]
    ).reset_index(drop=True)
    version_key = ["ticker", "metric", "period_end"]
    frame["revision_status"] = np.where(
        frame.groupby(version_key, sort=False).cumcount().eq(0), "ORIGINAL", "REVISION"
    )
    return frame[[*PIT_COLUMNS, *LINEAGE_COLUMNS]]


def build_publication_alignment_report(
    records: pd.DataFrame, *, trading_days: Iterable[object] | None = None
) -> pd.DataFrame:
    """Return one auditable row per record; anomalies are never hidden."""

    columns = [
        "ticker",
        "metric",
        "period_end",
        "publication_date",
        "available_date",
        "period_end_to_publication_days",
        "publication_to_available_days",
        "anomaly",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)
    frame = records.copy()
    for column in ("period_end", "publication_date", "available_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["period_end_to_publication_days"] = (
        frame["publication_date"] - frame["period_end"]
    ).dt.days
    frame["publication_to_available_days"] = (
        frame["available_date"] - frame["publication_date"]
    ).dt.days
    frame["anomaly"] = ""
    checks = [
        (frame["publication_date"] < frame["period_end"], "PUBLICATION_BEFORE_PERIOD_END"),
        (frame["available_date"] < frame["publication_date"], "AVAILABLE_BEFORE_PUBLICATION"),
    ]
    if trading_days is not None:
        days = set(pd.DatetimeIndex(pd.to_datetime(list(trading_days))))
        checks.append(
            (~frame["available_date"].isin(days), "UNMAPPED_TRADING_DAY")
        )
    duplicate = frame.duplicated(
        ["ticker", "metric", "period_end", "publication_date"], keep=False
    )
    checks.append((duplicate, "DUPLICATE_ACTIVE_RECORD"))
    for mask, reason in checks:
        frame.loc[mask & frame["anomaly"].eq(""), "anomaly"] = reason
    return frame[columns].sort_values(
        ["period_end", "ticker", "metric", "publication_date"]
    ).reset_index(drop=True)


def publication_alignment_summary(report: pd.DataFrame) -> dict[str, Any]:
    """Summarize alignment intervals and anomaly counts for JSON reports."""

    def stats(series: pd.Series) -> dict[str, float | None]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return {name: None for name in ("mean", "median", "min", "max", "p95")}
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p95": float(values.quantile(0.95)),
        }

    return {
        "record_count": int(len(report)),
        "period_end_to_publication_days": stats(
            report.get("period_end_to_publication_days", pd.Series(dtype=float))
        ),
        "publication_to_available_days": stats(
            report.get("publication_to_available_days", pd.Series(dtype=float))
        ),
        "anomalies": report.get("anomaly", pd.Series(dtype=str))
        .loc[lambda values: values.ne("")]
        .value_counts()
        .to_dict(),
    }


def build_fundamental_coverage_reports(
    matrix: pd.DataFrame,
    research_universe: pd.DataFrame,
    *,
    years: Iterable[int] = range(2019, 2026),
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Build daily, yearly, and ticker coverage from the eligible universe."""

    universe = research_universe.loc[
        research_universe["is_eligible"].astype(bool), ["date", "ticker"]
    ].copy()
    universe["date"] = pd.to_datetime(universe["date"])
    universe["ticker"] = universe["ticker"].astype(str)
    universe = universe.drop_duplicates()
    dates = pd.DataFrame({"date": sorted(universe["date"].unique())})
    daily = dates.merge(
        universe.groupby("date")["ticker"].nunique().rename("liquid_universe_count"),
        on="date",
        how="left",
    )
    valid = matrix.copy()
    valid["date"] = pd.to_datetime(valid["date"])
    valid["ticker"] = valid["ticker"].astype(str)
    valid = valid.loc[valid["metric"].isin(FUNDAMENTAL_METRICS)]
    valid = valid.dropna(subset=["value"]).drop_duplicates(
        ["date", "ticker", "metric"]
    )
    valid = valid.merge(universe, on=["date", "ticker"], how="inner")
    for metric in FUNDAMENTAL_METRICS:
        counts = (
            valid.loc[valid["metric"].eq(metric)]
            .groupby("date")["ticker"]
            .nunique()
            .rename(f"{metric}_valid_count")
        )
        daily = daily.merge(counts, on="date", how="left")
    daily = daily.fillna(0)
    joint = valid.pivot_table(
        index=["date", "ticker"], columns="metric", values="value", aggfunc="last"
    ).reindex(columns=FUNDAMENTAL_METRICS)
    joint_counts = (
        joint.dropna(subset=list(FUNDAMENTAL_METRICS))
        .reset_index()
        .groupby("date")["ticker"]
        .nunique()
        .rename("joint_valid_count")
    )
    daily = daily.merge(joint_counts, on="date", how="left").fillna(0)
    daily["liquid_universe_count"] = daily["liquid_universe_count"].astype(int)
    for column in ["eps_valid_count", "roe_valid_count", "joint_valid_count"]:
        daily[column] = daily[column].astype(int)
    for metric in FUNDAMENTAL_METRICS:
        daily[f"{metric}_coverage"] = daily[f"{metric}_valid_count"].div(
            daily["liquid_universe_count"].replace(0, np.nan)
        ).fillna(0.0)
    daily["joint_coverage"] = daily["joint_valid_count"].div(
        daily["liquid_universe_count"].replace(0, np.nan)
    ).fillna(0.0)
    daily["year"] = daily["date"].dt.year

    yearly = (
        daily.groupby("year")
        .agg(
            date_count=("date", "nunique"),
            liquid_universe_count=("liquid_universe_count", "mean"),
            eps_valid_count=("eps_valid_count", "mean"),
            roe_valid_count=("roe_valid_count", "mean"),
            joint_valid_count=("joint_valid_count", "mean"),
            eps_coverage=("eps_coverage", "mean"),
            roe_coverage=("roe_coverage", "mean"),
            joint_coverage=("joint_coverage", "mean"),
        )
        .reset_index()
    )
    yearly = yearly.set_index("year").reindex(list(years)).reset_index()
    yearly["year"] = yearly["year"].astype(int)

    ticker = universe.groupby("ticker").agg(
        eligible_date_count=("date", "nunique"),
        first_eligible_date=("date", "min"),
        last_eligible_date=("date", "max"),
    )
    valid_ticker = valid.assign(is_valid=True)
    for metric in FUNDAMENTAL_METRICS:
        rows = valid_ticker[valid_ticker["metric"].eq(metric)]
        grouped = rows.groupby("ticker").agg(
            **{
                f"{metric}_observation_count": ("date", "nunique"),
                f"{metric}_first_available_date": ("date", "min"),
                f"{metric}_last_available_date": ("date", "max"),
            }
        )
        ticker = ticker.join(grouped)
    ticker = ticker.fillna({
        "eps_observation_count": 0,
        "roe_observation_count": 0,
    })
    ticker["eps_coverage_ratio"] = ticker["eps_observation_count"].div(
        ticker["eligible_date_count"]
    )
    ticker["roe_coverage_ratio"] = ticker["roe_observation_count"].div(
        ticker["eligible_date_count"]
    )
    ticker = ticker.reset_index()

    distribution: dict[str, Any] = {}
    for metric in ("eps_coverage", "roe_coverage", "joint_coverage"):
        values = daily[metric]
        distribution[metric] = {
            "mean": float(values.mean()),
            "median": float(values.quantile(0.5)),
            "min": float(values.min()),
            "max": float(values.max()),
            "p10": float(values.quantile(0.1)),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.9)),
        }
    return {
        "daily": daily,
        "yearly": yearly,
        "by_ticker": ticker,
        "distribution": distribution,
    }


def build_fundamental_missingness_report(
    matrix: pd.DataFrame,
    research_universe: pd.DataFrame,
    *,
    records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reconcile every eligible date/ticker/metric cell that is missing."""

    universe = research_universe.loc[
        research_universe["is_eligible"].astype(bool), ["date", "ticker"]
    ].copy()
    universe["date"] = pd.to_datetime(universe["date"])
    universe["ticker"] = universe["ticker"].astype(str)
    cells = universe.assign(_key=1).merge(
        pd.DataFrame({"metric": list(FUNDAMENTAL_METRICS), "_key": 1}), on="_key"
    ).drop(columns="_key")
    valid = matrix.loc[matrix["metric"].isin(FUNDAMENTAL_METRICS)].copy()
    valid["date"] = pd.to_datetime(valid["date"])
    valid["ticker"] = valid["ticker"].astype(str)
    valid = valid.dropna(subset=["value"]).drop_duplicates(
        ["date", "ticker", "metric"]
    )[["date", "ticker", "metric", "value"]]
    report = cells.merge(valid, on=["date", "ticker", "metric"], how="left", indicator=True)
    report = report.loc[report["_merge"].eq("left_only")].drop(columns="_merge")
    if report.empty:
        return pd.DataFrame(columns=["date", "ticker", "metric", "missing_reason"])
    if records is None:
        records = pd.DataFrame(columns=["ticker", "metric", "available_date", "value"])
    source = records.loc[records["metric"].isin(FUNDAMENTAL_METRICS)].copy()
    source["ticker"] = source["ticker"].astype(str)
    source["available_date"] = pd.to_datetime(source["available_date"], errors="coerce")
    source_counts = source.groupby(["ticker", "metric"]).size().rename("source_count")
    latest = source.groupby(["ticker", "metric"])["available_date"].min().rename("first_available_date")
    report = report.join(source_counts, on=["ticker", "metric"])
    report = report.join(latest, on=["ticker", "metric"])
    report["missing_reason"] = np.select(
        [
            report["source_count"].isna(),
            report["first_available_date"].gt(report["date"]),
        ],
        ["NO_SOURCE_RECORD", "NOT_YET_PUBLISHED"],
        default="INVALID_RECORD",
    )
    return report[
        ["date", "ticker", "metric", "missing_reason", "source_count", "first_available_date"]
    ].sort_values(["date", "ticker", "metric"]).reset_index(drop=True)


def factor_gate_compatible_coverage(
    factor_matrix: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    horizon: int = 20,
) -> float:
    """Use the Factor Gate's exact h20 coverage semantics without IC/verdicts."""

    pairs = _paired_observations(factor_matrix, forward_returns, horizon)
    return _coverage(pairs, factor_matrix, forward_returns, horizon)


def evaluate_data_readiness(
    records: pd.DataFrame,
    *,
    alignment_report: pd.DataFrame,
    coverage: dict[str, Any] | None = None,
    factor_gate_coverages: dict[str, float] | None = None,
    threshold: float = 0.20,
    source_status: str = "AVAILABLE",
) -> dict[str, Any]:
    """Evaluate BLOCKED/PARTIAL/READY without running factor IC diagnostics."""

    required = set(PIT_COLUMNS)
    schema_pass = required.issubset(records.columns)
    pit_pass = False
    if schema_pass:
        try:
            validate_pit_records(records)
        except (KeyError, ValueError):
            pit_pass = False
        else:
            pit_pass = True
    alignment_pass = not alignment_report.get("anomaly", pd.Series(dtype=str)).ne("").any()
    duplicate_columns = [
        column
        for column in ("ticker", "metric", "period_end", "publication_date")
        if column in records
    ]
    duplicate_revision_pass = bool(duplicate_columns) and not records.duplicated(
        duplicate_columns
    ).any()
    integrity = {
        "pit_integrity": "PASS" if pit_pass else "FAIL",
        "schema_integrity": "PASS" if schema_pass else "FAIL",
        "publication_alignment": "PASS" if alignment_pass else "FAIL",
        "duplicate_revision_integrity": "PASS" if duplicate_revision_pass else "FAIL",
    }
    if factor_gate_coverages is None:
        yearly = (coverage or {}).get("daily", pd.DataFrame())
        factor_gate_coverages = {
            metric: float(yearly[f"{metric}_coverage"].median())
            if f"{metric}_coverage" in yearly
            else 0.0
            for metric in ("eps", "roe")
        }
        if "joint_coverage" in yearly:
            factor_gate_coverages["joint"] = float(yearly["joint_coverage"].median())
    factor_gate_coverages = {key: float(value) for key, value in factor_gate_coverages.items()}
    eps_ready = factor_gate_coverages.get("eps", 0.0) >= threshold
    roe_ready = factor_gate_coverages.get("roe", 0.0) >= threshold
    joint_ready = factor_gate_coverages.get("joint", 0.0) >= threshold
    integrity_pass = all(value == "PASS" for value in integrity.values())
    if not integrity_pass or source_status == "UNAVAILABLE":
        overall = "BLOCKED"
    elif eps_ready and roe_ready and joint_ready:
        overall = "READY"
    else:
        overall = "PARTIAL"
    return {
        "integrity_gates": integrity,
        "integrity_pass": integrity_pass,
        "factor_gate_compatible_coverage": factor_gate_coverages,
        "coverage_threshold": threshold,
        "eps_research_ready": eps_ready,
        "roe_research_ready": roe_ready,
        "joint_composite_ready": joint_ready,
        "joint_status": "READY" if joint_ready else "PARTIAL_FOR_COMPOSITE",
        "overall_data_readiness": overall,
        "READY_FOR_RESEARCH_CYCLE_V3": "YES" if overall == "READY" else "NO",
        "source_status": source_status,
    }


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks for manifests without loading large raw files."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_raw_inventory(
    cache_root: str | Path, *, code_revision: str = "unknown"
) -> pd.DataFrame:
    """Index raw-cache metadata and content hashes without parsing source files."""

    rows: list[dict[str, Any]] = []
    root = Path(cache_root)
    for metadata_path in sorted(root.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_path = metadata_path.with_suffix(".raw")
        if not raw_path.exists():
            continue
        request = metadata.get("request_params") or metadata.get("params") or {}
        rows.append(
            {
                "raw_path": str(raw_path),
                "metadata_path": str(metadata_path),
                "source": metadata.get("source", ""),
                "endpoint": metadata.get("endpoint", ""),
                "retrieval_mode": "cache",
                "retrieved_at": metadata.get("fetched_at", ""),
                "raw_hash": metadata.get("content_hash") or sha256_file(raw_path),
                "code_revision": code_revision,
                "request_params": json.dumps(request, ensure_ascii=False, sort_keys=True),
                "classification": metadata.get("classification", ""),
            }
        )
    return pd.DataFrame(rows)


def build_source_audit(
    baseline_records: pd.DataFrame,
    baseline_matrix: pd.DataFrame,
    research_universe: pd.DataFrame,
    *,
    raw_inventory: pd.DataFrame,
    source_checks: dict[str, Any],
) -> dict[str, Any]:
    """Explain the baseline 25/30 gap and list available expansion sources."""

    baseline = baseline_records.loc[baseline_records["metric"].eq("eps")].copy()
    matrix = baseline_matrix.loc[baseline_matrix["metric"].eq("eps")].copy()
    baseline_tickers = sorted(baseline["ticker"].astype(str).unique())
    matrix_tickers = set(matrix["ticker"].astype(str).unique())
    universe = research_universe.copy()
    universe["ticker"] = universe["ticker"].astype(str)
    universe["date"] = pd.to_datetime(universe["date"])
    root_causes = []
    for ticker in baseline_tickers:
        records = baseline.loc[baseline["ticker"].astype(str).eq(ticker)]
        eligible = universe.loc[
            universe["ticker"].eq(ticker) & universe["is_eligible"].astype(bool),
            "date",
        ]
        if ticker in matrix_tickers:
            reason = "covered_by_baseline_matrix"
        elif eligible.empty:
            reason = "NO_LIQUID_PIT_SAFE_UNIVERSE_OVERLAP"
        else:
            record_start = pd.to_datetime(records["available_date"]).min()
            record_end = pd.to_datetime(records["available_date"]).max()
            reason = (
                "NO_LIQUID_PIT_SAFE_UNIVERSE_OVERLAP"
                if record_end < eligible.min() or record_start > eligible.max()
                else "MATRIX_BUILD_OR_PIT_VALIDATION_GAP"
            )
        root_causes.append(
            {
                "ticker": ticker,
                "baseline_eps_records": int(len(records)),
                "baseline_matrix_rows": int(
                    matrix.loc[matrix["ticker"].astype(str).eq(ticker)].shape[0]
                ),
                "eligible_date_count": int(len(eligible)),
                "root_cause": reason,
            }
        )
    endpoint_counts = raw_inventory["endpoint"].value_counts().to_dict()
    unavailable = source_checks.get("unavailable", [])
    return {
        "DATA_EXPANSION_STATUS": (
            "BLOCKED_SOURCE_UNAVAILABLE" if unavailable else "SOURCE_AVAILABLE"
        ),
        "baseline": {
            "eps_pit_ticker_count": len(baseline_tickers),
            "eps_matrix_ticker_count": len(matrix_tickers),
            "joint_coverage_ticker_count": len(
                set(baseline["ticker"].astype(str).unique())
                & set(
                    baseline_records.loc[baseline_records["metric"].eq("roe"), "ticker"]
                    .astype(str)
                    .unique()
                )
                & matrix_tickers
            ),
        },
        "root_causes": root_causes,
        "coverage_root_cause": root_causes,
        "raw_cache_inventory": {
            "metadata_count": int(len(raw_inventory)),
            "raw_count": int(raw_inventory["raw_path"].nunique())
            if not raw_inventory.empty
            else 0,
            "endpoint_counts": {str(key): int(value) for key, value in endpoint_counts.items()},
        },
        "expandable_sources": [
            {
                "source": "doc.twse",
                "endpoint": "t57sb01",
                "retrieval_mode": "per_ticker_year_season",
            },
            {
                "source": "mopsov",
                "endpoint": "ajax_t163sb04/ajax_t163sb05",
                "retrieval_mode": "bulk_year_season",
            },
        ],
        "source_checks": source_checks,
        "data_source_audit": source_checks,
        "external_blockers": unavailable,
        "required_external_source": (
            "doc.twse t57sb01 + mopsov ajax_t163sb04/ajax_t163sb05"
            if unavailable
            else None
        ),
    }


__all__ = [
    "FUNDAMENTAL_METRICS",
    "LINEAGE_COLUMNS",
    "build_fundamental_coverage_reports",
    "build_fundamental_missingness_report",
    "build_raw_inventory",
    "build_publication_alignment_report",
    "build_source_audit",
    "build_ticker_mapping_report",
    "evaluate_data_readiness",
    "factor_gate_compatible_coverage",
    "normalize_fundamental_records",
    "publication_alignment_summary",
    "sha256_file",
]

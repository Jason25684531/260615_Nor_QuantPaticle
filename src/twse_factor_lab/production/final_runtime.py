"""Final Fundamental production runtime contracts.

This module is intentionally input-only: factor and ranking calculations stay
in the canonical research/runtime provider.  Missing canonical G2/G3 inputs
are a hard stop, never an invitation to infer a replacement factor.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from twse_factor_lab.production.fundamental import (
    STRATEGY_FINGERPRINT,
    STRATEGY_ID,
    FundamentalStrategySpec,
    ProductionEligibilityGate,
    ProductionIntegrationError,
    build_current_promotion_evidence,
    normalize_recommendations,
)

FINAL_NAMESPACE = "fundamental-production-final-v1"
BROKER_ORDER_SUBMISSION = "DISABLED"
FACTOR_OUTPUT_COLUMNS = (
    "ticker",
    "as_of_date",
    "universe_eligible",
    "g2_raw",
    "g3_raw",
    "normalized_g2",
    "normalized_g3",
    "composite_score",
    "rank",
    "selected",
    "target_weight",
    "fundamental_period_end",
    "fundamental_available_date",
    "publication_date",
    "data_source",
    "input_sha",
    "provider_sha",
    "rebalance_flag",
    "reason",
)
RECOMMENDATION_COLUMNS = (
    "run_id",
    "as_of_date",
    "strategy_id",
    "strategy_fingerprint",
    "ticker",
    "score",
    "rank",
    "selected",
    "target_weight",
    "action",
    "rebalance_due",
    "reason",
    "input_sha",
    "provider_sha",
    "created_at",
)
PARITY_NUMERIC = (
    "g2_raw",
    "g3_raw",
    "normalized_g2",
    "normalized_g3",
    "composite_score",
    "target_weight",
)
PARITY_EXACT = ("ticker", "rank", "selected", "rebalance_flag")


class FinalRuntimeError(ProductionIntegrationError):
    """Raised when the final runtime contract cannot be satisfied."""


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _date(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _source_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha(
        [
            {key: row.get(key) for key in sorted(row) if key not in {"provider_sha"}}
            for row in rows
        ]
    )


class CanonicalFundamentalRuntimeProvider:
    """Expose already-calculated canonical PIT-safe Fundamental rows.

    ``rows`` are supplied by the canonical factor runtime.  This class validates
    and fingerprints them; it does not implement G2/G3 or ranking.  Repository
    construction deliberately fails when the checked-in PIT layer lacks the
    frozen factor outputs.
    """

    def __init__(
        self,
        rows: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
        *,
        factor_runtime: Callable[[str], Iterable[Mapping[str, Any]]] | None = None,
        data_source: str = "canonical_fundamental_pit",
        market_sessions: Sequence[str | date] | None = None,
        security_master: Iterable[str] | None = None,
    ) -> None:
        self._rows = rows
        self._factor_runtime = factor_runtime
        self.data_source = data_source
        self._market_sessions = {_date(value) for value in (market_sessions or ())}
        self._security_master = {str(value) for value in (security_master or ())}

    @classmethod
    def from_repository(cls, root: str | Path) -> CanonicalFundamentalRuntimeProvider:
        path = (
            Path(root) / "data/processed/fundamental_pit_v2/fundamental_records.parquet"
        )
        if not path.exists():
            raise FinalRuntimeError("CANONICAL_PIT_DATASET_MISSING")
        frame = pd.read_parquet(path)
        required_metrics = {"operating_income", "eps"}
        present = set(frame.get("metric", pd.Series(dtype=str)).astype(str))
        missing = sorted(required_metrics - present)
        if missing:
            raise FinalRuntimeError(
                "CANONICAL_G2_G3_INPUT_UNAVAILABLE:" + ",".join(missing)
            )
        return cls(frame, data_source=str(path))

    def _raw_rows(self, as_of_date: str) -> list[Mapping[str, Any]]:
        if self._factor_runtime is not None:
            return list(self._factor_runtime(as_of_date))
        if self._rows is None:
            raise FinalRuntimeError("CANONICAL_G2_G3_INPUT_UNAVAILABLE")
        frame = (
            self._rows
            if isinstance(self._rows, pd.DataFrame)
            else pd.DataFrame(self._rows)
        )
        required = set(FACTOR_OUTPUT_COLUMNS) - {
            "input_sha",
            "provider_sha",
            "data_source",
        }
        if not required.issubset(frame.columns):
            if "metric" in frame.columns:
                missing = sorted(
                    {"operating_income", "eps"} - set(frame["metric"].astype(str))
                )
                raise FinalRuntimeError(
                    "CANONICAL_G2_G3_INPUT_UNAVAILABLE:"
                    + (",".join(missing) or "factor_output_columns")
                )
            missing = sorted(required - set(frame.columns))
            raise FinalRuntimeError(
                "CANONICAL_FACTOR_OUTPUT_SCHEMA_INVALID:" + ",".join(missing)
            )
        return frame.to_dict("records")

    def snapshot(
        self, as_of_date: str, *, rebalance_flag: bool | None = None
    ) -> list[dict[str, Any]]:
        rows = self._raw_rows(as_of_date)
        if not rows:
            raise FinalRuntimeError("CANONICAL_FACTOR_SNAPSHOT_EMPTY")
        frame = pd.DataFrame(rows).copy()
        frame["as_of_date"] = frame["as_of_date"].map(_date)
        as_of = _date(as_of_date)
        if self._market_sessions and as_of not in self._market_sessions:
            raise FinalRuntimeError("AS_OF_DATE_NOT_IN_CANONICAL_MARKET_SESSION")
        if set(frame["as_of_date"]) != {as_of}:
            raise FinalRuntimeError("SNAPSHOT_AS_OF_DATE_MISMATCH")
        frame["fundamental_available_date"] = frame["fundamental_available_date"].map(
            _date
        )
        frame["publication_date"] = frame["publication_date"].map(_date)
        frame["fundamental_period_end"] = frame["fundamental_period_end"].map(_date)
        if (
            frame["publication_date"].isna().any()
            or frame["fundamental_available_date"].isna().any()
            or frame["fundamental_period_end"].isna().any()
            or (frame["fundamental_period_end"] > frame["publication_date"]).any()
            or (frame["publication_date"] > frame["fundamental_available_date"]).any()
            or (frame["fundamental_available_date"] > as_of).any()
        ):
            raise FinalRuntimeError("PIT_INTEGRITY_FAIL")
        if frame["ticker"].astype(str).duplicated().any():
            raise FinalRuntimeError("DUPLICATE_TICKER_SNAPSHOT")
        if self._security_master and not set(frame["ticker"].astype(str)).issubset(
            self._security_master
        ):
            raise FinalRuntimeError("SECURITY_MASTER_MISMATCH")
        for column in FACTOR_OUTPUT_COLUMNS:
            if column not in frame:
                if column in {"input_sha", "provider_sha"}:
                    continue
                if column == "data_source":
                    frame[column] = self.data_source
                elif column == "rebalance_flag":
                    frame[column] = (
                        bool(rebalance_flag) if rebalance_flag is not None else False
                    )
                elif column == "reason":
                    frame[column] = "CANONICAL_G2_G3_SCORE_WEIGHTED"
                else:
                    raise FinalRuntimeError(
                        "CANONICAL_FACTOR_OUTPUT_SCHEMA_INVALID:" + column
                    )
        frame["ticker"] = frame["ticker"].astype(str)
        frame["universe_eligible"] = frame["universe_eligible"].map(_as_bool)
        frame["selected"] = frame["selected"].map(_as_bool)
        frame["rebalance_flag"] = frame["rebalance_flag"].map(_as_bool)
        for column in PARITY_NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[list(PARITY_NUMERIC)].isna().any().any():
            raise FinalRuntimeError("CANONICAL_FACTOR_OUTPUT_NON_NUMERIC")
        raw = frame.to_dict("records")
        input_sha = _source_sha(raw)
        provider_rows = []
        for row in raw:
            item = dict(row)
            item["score"] = float(item["composite_score"])
            item["input_sha"] = str(item.get("input_sha") or input_sha)
            item["data_source"] = str(item.get("data_source") or self.data_source)
            provider_rows.append(item)
        provider_sha = _sha(
            [
                {
                    key: row.get(key)
                    for key in FACTOR_OUTPUT_COLUMNS
                    if key != "provider_sha"
                }
                for row in provider_rows
            ]
        )
        for row in provider_rows:
            row["provider_sha"] = provider_sha
        return sorted(provider_rows, key=lambda row: (int(row["rank"]), row["ticker"]))


def snapshot_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(FACTOR_OUTPUT_COLUMNS))
    if not frame.empty:
        frame = frame.sort_values(["as_of_date", "rank", "ticker"]).reset_index(
            drop=True
        )
    return frame


def build_real_snapshot_parity(
    research: Sequence[Mapping[str, Any]],
    shadow: Sequence[Mapping[str, Any]],
    production: Sequence[Mapping[str, Any]],
    *,
    dates: Sequence[str] = (),
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compare real rows; empty or unavailable runtimes can never pass."""

    sources = {
        "research": list(research),
        "shadow": list(shadow),
        "production": list(production),
    }
    report: dict[str, Any] = {
        "snapshot_count": len(dates),
        "date_list": list(dates),
        "research_rows": len(research),
        "shadow_rows": len(shadow),
        "production_rows": len(production),
        "source_runtime_available": all(bool(rows) for rows in sources.values()),
        "empty_output_parity_accepted": False,
        "factor_mismatches": 0,
        "score_mismatches": 0,
        "rank_mismatches": 0,
        "selection_mismatches": 0,
        "target_mismatches": 0,
        "rebalance_mismatches": 0,
        "max_numerical_diff": 0.0,
        "status": "BLOCKED",
        "reason": "SOURCE_RUNTIME_UNAVAILABLE",
    }
    if not report["source_runtime_available"]:
        return report

    def key(row: Mapping[str, Any]) -> tuple[str, str]:
        return (str(row.get("as_of_date", "")), str(row["ticker"]))

    base = {key(row): row for row in research}
    for name in ("shadow", "production"):
        other = {key(row): row for row in sources[name]}
        if set(other) != set(base):
            report["selection_mismatches"] += len(set(other) ^ set(base))
            continue
        for row_key, expected in base.items():
            actual = other[row_key]
            for field in PARITY_NUMERIC:
                diff = abs(float(expected[field]) - float(actual[field]))
                report["max_numerical_diff"] = max(report["max_numerical_diff"], diff)
                if diff > atol:
                    if field in {"g2_raw", "g3_raw", "normalized_g2", "normalized_g3"}:
                        report["factor_mismatches"] += 1
                    elif field == "composite_score":
                        report["score_mismatches"] += 1
                    else:
                        report["target_mismatches"] += 1
            for field in PARITY_EXACT:
                if str(expected[field]) != str(actual[field]):
                    if field == "rank":
                        report["rank_mismatches"] += 1
                    elif field == "selected":
                        report["selection_mismatches"] += 1
                    elif field == "rebalance_flag":
                        report["rebalance_mismatches"] += 1
    report["status"] = (
        "PASS"
        if not any(
            report[key]
            for key in (
                "factor_mismatches",
                "score_mismatches",
                "rank_mismatches",
                "selection_mismatches",
                "target_mismatches",
                "rebalance_mismatches",
            )
        )
        else "FAIL"
    )
    report["reason"] = (
        "ZERO_MISMATCH" if report["status"] == "PASS" else "SNAPSHOT_MISMATCH"
    )
    return report


def build_three_runtime_snapshots(
    provider: CanonicalFundamentalRuntimeProvider,
    dates: Sequence[str],
    *,
    research_runtime: Callable[[str], Iterable[Mapping[str, Any]]] | None = None,
    shadow_runtime: Callable[[str], Iterable[Mapping[str, Any]]] | None = None,
    production_runtime: Callable[[str], Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run three named runtime paths over the same dates and input contract."""

    if not all((research_runtime, shadow_runtime, production_runtime)):
        raise FinalRuntimeError("REAL_RUNTIME_PATH_REQUIRED")
    if len({id(research_runtime), id(shadow_runtime), id(production_runtime)}) != 3:
        raise FinalRuntimeError("DISTINCT_RUNTIME_PATHS_REQUIRED")
    runtimes = {
        "research": research_runtime,
        "shadow": shadow_runtime,
        "production": production_runtime,
    }
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in runtimes}
    for as_of_date in dates:
        for name, runtime in runtimes.items():
            rows = list(runtime(as_of_date))
            if not rows:
                raise FinalRuntimeError(f"{name.upper()}_RUNTIME_EMPTY:{as_of_date}")
            result[name].extend(rows)
    return result


def predeclared_snapshot_manifest() -> dict[str, Any]:
    dates = [
        "2019-03-29",
        "2019-06-28",
        "2020-03-31",
        "2020-09-30",
        "2021-03-31",
        "2021-12-30",
        "2022-06-30",
        "2022-12-30",
        "2023-06-30",
        "2023-12-29",
        "2024-06-28",
        "2025-06-30",
    ]
    return {
        "rule": (
            "fixed quarter-end/last-session dates declared before "
            "performance evaluation"
        ),
        "snapshot_count": len(dates),
        "dates": dates,
        "cases": {
            "rebalance": dates[::2],
            "non_rebalance": dates[1::2],
            "factor_update_before_after": [dates[2], dates[3]],
            "universe_membership_change": [dates[6], dates[7]],
        },
        "input_sha": None,
        "status": "PREDECLARED_RUNTIME_INPUT_PENDING",
    }


@dataclass(frozen=True)
class DailyRecommendation:
    run_id: str
    as_of_date: str
    strategy_id: str
    strategy_fingerprint: str
    ticker: str
    score: float
    rank: int
    selected: bool
    target_weight: float
    action: str
    rebalance_due: bool
    reason: str
    input_sha: str
    provider_sha: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def trading_session_rebalance_due(
    sessions: Sequence[str | date],
    as_of_date: str,
    last_rebalance_date: str | None = None,
) -> bool:
    ordered = sorted({_date(item) for item in sessions})
    current = _date(as_of_date)
    if current not in ordered:
        raise FinalRuntimeError("AS_OF_DATE_NOT_IN_MARKET_CALENDAR")
    if last_rebalance_date is None:
        return True
    last = _date(last_rebalance_date)
    prior = [item for item in ordered if item <= last]
    if not prior:
        return True
    return ordered.index(current) - ordered.index(prior[-1]) >= 60


def build_daily_recommendations(
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of_date: str,
    rebalance_due: bool,
    prior_targets: Sequence[Mapping[str, Any]] = (),
) -> list[DailyRecommendation]:
    canonical_rows = [
        {
            **row,
            "score": row.get("score", row.get("composite_score")),
        }
        for row in rows
    ]
    canonical_prior = [
        {
            **row,
            "score": row.get("score", row.get("composite_score", 0.0)),
            "stock_id": row.get("stock_id", row.get("ticker")),
        }
        for row in prior_targets
    ]
    normalized = normalize_recommendations(
        canonical_rows,
        strategy_id=STRATEGY_ID,
        asof_date=as_of_date,
        rebalance_flag=rebalance_due,
        prior_targets=canonical_prior,
    )
    selected = [row for row in normalized if _as_bool(row.get("selected", True))]
    prior = {
        str(row.get("ticker", row.get("stock_id"))): row for row in canonical_prior
    }
    input_sha = str(canonical_rows[0].get("input_sha", "")) if canonical_rows else ""
    provider_sha = (
        str(canonical_rows[0].get("provider_sha", "")) if canonical_rows else ""
    )
    run_id = _sha([as_of_date, STRATEGY_FINGERPRINT, input_sha, rebalance_due])
    created = f"{_date(as_of_date).date().isoformat()}T00:00:00+00:00"
    result: list[DailyRecommendation] = []
    for row in selected:
        ticker = str(row.get("ticker", row.get("stock_id")))
        action = (
            "NO_REBALANCE"
            if not rebalance_due
            else ("HOLD" if ticker in prior else "ENTER")
        )
        result.append(
            DailyRecommendation(
                run_id=run_id,
                as_of_date=as_of_date,
                strategy_id=STRATEGY_ID,
                strategy_fingerprint=STRATEGY_FINGERPRINT,
                ticker=ticker,
                score=float(row["score"]),
                rank=int(row["rank"]),
                selected=True,
                target_weight=float(row["target_weight"]),
                action=action,
                rebalance_due=bool(rebalance_due),
                reason=str(row.get("reason", "CANONICAL_G2_G3_SCORE_WEIGHTED")),
                input_sha=input_sha,
                provider_sha=provider_sha,
                created_at=created,
            )
        )
    if rebalance_due:
        current = {item.ticker for item in result}
        for ticker, row in sorted(prior.items()):
            if ticker not in current:
                result.append(
                    DailyRecommendation(
                        run_id=run_id,
                        as_of_date=as_of_date,
                        strategy_id=STRATEGY_ID,
                        strategy_fingerprint=STRATEGY_FINGERPRINT,
                        ticker=ticker,
                        score=float(row.get("score", 0.0)),
                        rank=int(row.get("rank", 0)),
                        selected=False,
                        target_weight=0.0,
                        action="EXIT",
                        rebalance_due=True,
                        reason="REB60_EXIT",
                        input_sha=input_sha,
                        provider_sha=provider_sha,
                        created_at=created,
                    )
                )
    return sorted(result, key=lambda item: (item.rank, item.ticker))


class AtomicRecommendationStore:
    """Parquet store with replacement writes and duplicate-key protection."""

    key_columns = ("as_of_date", "strategy_id", "strategy_fingerprint", "ticker")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "daily_recommendations.parquet"

    def persist(self, recommendations: Sequence[DailyRecommendation]) -> int:
        if not recommendations:
            return 0
        incoming = pd.DataFrame(
            [row.as_dict() for row in recommendations],
            columns=list(RECOMMENDATION_COLUMNS),
        )
        if incoming.duplicated(list(self.key_columns)).any():
            raise FinalRuntimeError("RECOMMENDATION_KEY_DUPLICATE")
        if self.path.exists():
            current = pd.read_parquet(self.path)
            conflict = current.merge(
                incoming,
                on=list(self.key_columns),
                how="inner",
                suffixes=("_old", "_new"),
            )
            for column in RECOMMENDATION_COLUMNS:
                if column in self.key_columns:
                    continue
                if (
                    f"{column}_old" in conflict
                    and f"{column}_new" in conflict
                    and (
                        conflict[f"{column}_old"].astype(str)
                        != conflict[f"{column}_new"].astype(str)
                    ).any()
                ):
                    raise FinalRuntimeError("RECOMMENDATION_IDEMPOTENCY_CONFLICT")
            incoming = pd.concat([current, incoming], ignore_index=True)
        incoming = (
            incoming.drop_duplicates(list(self.key_columns), keep="last")
            .sort_values(list(self.key_columns))
            .reset_index(drop=True)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix="daily_recommendations-", suffix=".parquet", dir=self.root
        )
        os.close(fd)
        try:
            incoming.to_parquet(temporary, index=False)
            Path(temporary).replace(self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return len(incoming)


def web_serialize(
    recommendations: Sequence[DailyRecommendation], *, data_status: str
) -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "as_of_date": recommendations[0].as_of_date if recommendations else None,
        "top5": [
            {
                "ticker": row.ticker,
                "score": row.score,
                "target_weight": row.target_weight,
                "action": row.action,
            }
            for row in recommendations
            if row.selected
        ][:5],
        "data_status": data_status,
        "generated_at": recommendations[0].created_at if recommendations else None,
    }


def line_format(
    recommendations: Sequence[DailyRecommendation], *, data_status: str
) -> str:
    if not recommendations:
        return f"{STRATEGY_ID} | {data_status} | NO_ACTIVE_RECOMMENDATIONS"
    top = ", ".join(
        f"{row.ticker}:{row.target_weight:.6f}"
        for row in recommendations
        if row.selected
    )
    return f"{recommendations[0].as_of_date} | {STRATEGY_ID} | {data_status} | {top}"


def run_daily_fundamental(
    *,
    provider: CanonicalFundamentalRuntimeProvider,
    as_of_date: str,
    sessions: Sequence[str],
    evidence: Mapping[str, Any] | None = None,
    prior_targets: Sequence[Mapping[str, Any]] = (),
    last_rebalance_date: str | None = None,
    write_recommendations: bool = False,
    store: AtomicRecommendationStore | None = None,
    explicit_enable: bool = False,
) -> dict[str, Any]:
    evidence = evidence or build_current_promotion_evidence()
    spec = FundamentalStrategySpec()
    gate = ProductionEligibilityGate()
    eligibility = gate.evaluate(evidence, spec, explicit_enable=explicit_enable)
    health: dict[str, Any] = {
        "as_of_date": as_of_date,
        "provider_status": "NOT_RUN",
        "pit_status": evidence.get("gates", {}).get("PIT_INTEGRITY", "FAIL"),
        "data_freshness": evidence.get("gates", {}).get("DATA_FRESHNESS_GATE", "FAIL"),
        "factor_health": evidence.get("gates", {}).get("FACTOR_HEALTH_GATE", "FAIL"),
        "strategy_status": "REGISTERED",
        "rebalance_status": "NOT_RUN",
        "parity_status": evidence.get("gates", {}).get(
            "RESEARCH_RUNTIME_PARITY", "FAIL"
        ),
        "promotion_status": evidence.get("production_promotion_status"),
        "recommendations_written": 0,
        "broker_submission": BROKER_ORDER_SUBMISSION,
        "overall_status": "BLOCKED",
    }
    if not eligibility.allowed:
        health["eligibility_status"] = eligibility.status
        health["eligibility_reason"] = eligibility.reason
        return {
            "status": "BLOCKED",
            "reason": eligibility.reason,
            "recommendations": [],
            "health": health,
        }
    try:
        rows = provider.snapshot(as_of_date)
        due = trading_session_rebalance_due(sessions, as_of_date, last_rebalance_date)
        recommendations = build_daily_recommendations(
            rows, as_of_date=as_of_date, rebalance_due=due, prior_targets=prior_targets
        )
    except FinalRuntimeError as exc:
        health["provider_status"] = "BLOCKED"
        health["eligibility_reason"] = str(exc)
        return {
            "status": "BLOCKED",
            "reason": str(exc),
            "recommendations": [],
            "health": health,
        }
    health.update(
        {
            "provider_status": "PASS",
            "rebalance_status": "PASS" if due else "NO_REBALANCE",
            "eligibility_status": "PASS",
            "eligibility_reason": "ELIGIBLE",
        }
    )
    if write_recommendations:
        if store is None:
            raise FinalRuntimeError("RECOMMENDATION_STORE_REQUIRED")
        health["recommendations_written"] = store.persist(recommendations)
    health["overall_status"] = "PASS"
    return {
        "status": "PASS",
        "reason": "ELIGIBLE",
        "recommendations": recommendations,
        "health": health,
    }


def fresh_oos_audit() -> dict[str, Any]:
    evidence = build_current_promotion_evidence()
    return {
        "strategy_id": STRATEGY_ID,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "freeze_date": "2026-07-28",
        "raw_data_existed": True,
        "performance_observed_for_selection": True,
        "contaminated_periods": ["2018-01-01/2026-07-28", "2026-01-02/2026-08-31"],
        "legal_untouched_window": None,
        "required": evidence["required_fresh_oos"],
        "observed": evidence["observed_fresh_oos"],
        "fresh_oos_status": "INSUFFICIENT_DATA",
        "reason": "NO_LEGAL_UNTOUCHED_WINDOW_WITH_REQUIRED_HISTORY",
        "fresh_oos_available": False,
    }


def production_contract() -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "factors": ["G2_OPERATING_INCOME_YOY", "G3_EPS_YOY"],
        "factor_weighting": "EQUAL",
        "top_n": 5,
        "rebalance": "60D",
        "portfolio_weighting": "SCORE_WEIGHTED",
        "research_knowledge_cutoff": "2026-07-28",
        "production_enable_flag": False,
        "broker_order_submission": BROKER_ORDER_SUBMISSION,
    }


__all__ = [
    "AtomicRecommendationStore",
    "BROKER_ORDER_SUBMISSION",
    "CanonicalFundamentalRuntimeProvider",
    "DailyRecommendation",
    "FACTOR_OUTPUT_COLUMNS",
    "FINAL_NAMESPACE",
    "FinalRuntimeError",
    "RECOMMENDATION_COLUMNS",
    "build_daily_recommendations",
    "build_real_snapshot_parity",
    "build_three_runtime_snapshots",
    "fresh_oos_audit",
    "line_format",
    "predeclared_snapshot_manifest",
    "production_contract",
    "run_daily_fundamental",
    "snapshot_frame",
    "trading_session_rebalance_due",
    "web_serialize",
]

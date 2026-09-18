"""Frozen Fundamental production contract and fail-closed runtime seam.

This module deliberately consumes canonical runtime rows.  It does not derive
G2/G3 values or query latest financial data; research and Shadow own that work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

STRATEGY_ID = "fundamental_g2g3_top5_reb60_score_weighted_v1"
STRATEGY_FINGERPRINT = (
    "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f"
)
ENABLE_FLAG = "ENABLE_FUNDAMENTAL_PRODUCTION"
FROZEN_PROMOTION_STATE = "PRODUCTION_APPROVED"
CURRENT_PROMOTION_STATE = "SHADOW_APPROVED"
REQUIRED_RECOMMENDATION_FIELDS = (
    "strategy_id",
    "stock_id",
    "score",
    "rank",
    "selected",
    "target_weight",
    "reason",
    "asof_date",
)
PIT_COLUMNS = frozenset(
    {"ticker", "factor_id", "score", "available_date", "publication_date", "pit_status"}
)
PARITY_FIELDS = (
    "stock_id",
    "score",
    "rank",
    "selected",
    "target_weight",
    "reason",
    "rebalance_flag",
)


class ProductionIntegrationError(ValueError):
    """Raised when a production boundary cannot be satisfied."""


@dataclass(frozen=True)
class FundamentalStrategySpec:
    strategy_id: str = STRATEGY_ID
    fingerprint: str = STRATEGY_FINGERPRINT
    factors: tuple[str, str] = (
        "G2_OPERATING_INCOME_YOY",
        "G3_EPS_YOY",
    )
    factor_weighting: str = "EQUAL"
    top_n: int = 5
    rebalance: str = "60D"
    portfolio_weighting: str = "SCORE_WEIGHTED"
    research_knowledge_cutoff: str = "2026-07-28"

    def validate(self) -> None:
        if self.strategy_id != STRATEGY_ID or self.fingerprint != STRATEGY_FINGERPRINT:
            raise ProductionIntegrationError("FROZEN_STRATEGY_IDENTITY_MISMATCH")
        if self.factors != ("G2_OPERATING_INCOME_YOY", "G3_EPS_YOY"):
            raise ProductionIntegrationError("FROZEN_FACTOR_SET_MISMATCH")
        if (
            self.factor_weighting != "EQUAL"
            or self.top_n != 5
            or self.rebalance != "60D"
            or self.portfolio_weighting != "SCORE_WEIGHTED"
            or self.research_knowledge_cutoff != "2026-07-28"
        ):
            raise ProductionIntegrationError("FROZEN_STRATEGY_PARAMETERS_MISMATCH")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["factors"] = list(self.factors)
        return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


CURRENT_PROMOTION_EVIDENCE: dict[str, Any] = {
    "artifact_version": "fundamental-promotion-evidence-v1",
    "strategy_id": STRATEGY_ID,
    "strategy_fingerprint": STRATEGY_FINGERPRINT,
    "fresh_oos_status": "INSUFFICIENT_DATA",
    "production_promotion_status": CURRENT_PROMOTION_STATE,
    "production_ready": False,
    "observed_fresh_oos": {
        "trading_days": 37,
        "months": 3,
        "rebalances": 1,
    },
    "required_fresh_oos": {
        "trading_days": 180,
        "months": 9,
        "rebalances": 3,
    },
    "gates": {
        "PIT_INTEGRITY": "PASS",
        "DATA_FRESHNESS_GATE": "PASS",
        "FACTOR_HEALTH_GATE": "PASS",
        "RESEARCH_RUNTIME_PARITY": "PASS",
        "HISTORICAL_SELECTION_DRIFT": 0,
    },
    "source": "authoritative-fundamental-governance-input",
}


def build_current_promotion_evidence() -> dict[str, Any]:
    payload = json.loads(canonical_json(CURRENT_PROMOTION_EVIDENCE))
    payload["artifact_sha256"] = sha256_payload(payload)
    return payload


def _without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("artifact_sha256", None)
    return result


def validate_promotion_evidence(
    evidence: Mapping[str, Any], spec: FundamentalStrategySpec
) -> None:
    spec.validate()
    required = {
        "artifact_version",
        "strategy_id",
        "strategy_fingerprint",
        "fresh_oos_status",
        "production_promotion_status",
        "production_ready",
        "observed_fresh_oos",
        "required_fresh_oos",
        "gates",
        "artifact_sha256",
    }
    if not required.issubset(evidence):
        raise ProductionIntegrationError("PROMOTION_EVIDENCE_INVALID")
    if (
        evidence["strategy_id"] != spec.strategy_id
        or evidence["strategy_fingerprint"] != spec.fingerprint
    ):
        raise ProductionIntegrationError("PROMOTION_EVIDENCE_INVALID")
    if evidence["artifact_sha256"] != sha256_payload(_without_hash(evidence)):
        raise ProductionIntegrationError("PROMOTION_EVIDENCE_INVALID")
    observed = evidence["observed_fresh_oos"]
    required_oos = evidence["required_fresh_oos"]
    if not isinstance(observed, Mapping) or not isinstance(required_oos, Mapping):
        raise ProductionIntegrationError("PROMOTION_EVIDENCE_INVALID")
    if any(
        int(observed.get(key, 0)) < 0
        for key in ("trading_days", "months", "rebalances")
    ):
        raise ProductionIntegrationError("PROMOTION_EVIDENCE_INVALID")


@dataclass(frozen=True)
class ProductionEligibility:
    status: str
    reason: str
    checks: dict[str, Any]

    @property
    def allowed(self) -> bool:
        return self.status == "PASS"


class ProductionEligibilityGate:
    """Single fail-closed gate for Fundamental production output."""

    def evaluate(
        self,
        evidence: Mapping[str, Any],
        spec: FundamentalStrategySpec,
        *,
        explicit_enable: bool = False,
    ) -> ProductionEligibility:
        try:
            validate_promotion_evidence(evidence, spec)
        except ProductionIntegrationError:
            return ProductionEligibility("BLOCKED", "PROMOTION_EVIDENCE_INVALID", {})
        gates = dict(evidence["gates"])
        checks = {
            "strategy_fingerprint": (
                evidence["strategy_fingerprint"] == spec.fingerprint
            ),
            "PIT_INTEGRITY": gates.get("PIT_INTEGRITY") == "PASS",
            "DATA_FRESHNESS_GATE": gates.get("DATA_FRESHNESS_GATE") == "PASS",
            "FACTOR_HEALTH_GATE": gates.get("FACTOR_HEALTH_GATE") == "PASS",
            "RESEARCH_RUNTIME_PARITY": gates.get("RESEARCH_RUNTIME_PARITY") == "PASS",
            "HISTORICAL_SELECTION_DRIFT": gates.get("HISTORICAL_SELECTION_DRIFT") == 0,
            "FRESH_OOS_STATUS": evidence["fresh_oos_status"] == "PASS",
            "PRODUCTION_PROMOTION_STATUS": evidence["production_promotion_status"]
            == FROZEN_PROMOTION_STATE,
            "PRODUCTION_READY": evidence["production_ready"] is True,
            "explicit_enablement": bool(explicit_enable),
        }
        if evidence["fresh_oos_status"] != "PASS":
            reason = (
                "INSUFFICIENT_OOS"
                if evidence["fresh_oos_status"] == "INSUFFICIENT_DATA"
                else "FRESH_OOS_STATUS_FAIL"
            )
            return ProductionEligibility("BLOCKED", reason, checks)
        for name in (
            "strategy_fingerprint",
            "PIT_INTEGRITY",
            "DATA_FRESHNESS_GATE",
            "FACTOR_HEALTH_GATE",
            "RESEARCH_RUNTIME_PARITY",
            "HISTORICAL_SELECTION_DRIFT",
        ):
            if not checks[name]:
                return ProductionEligibility("BLOCKED", f"{name}_FAIL", checks)
        if not checks["PRODUCTION_PROMOTION_STATUS"]:
            return ProductionEligibility(
                "BLOCKED", "PROMOTION_STATE_NOT_APPROVED", checks
            )
        if not checks["PRODUCTION_READY"]:
            return ProductionEligibility("BLOCKED", "PRODUCTION_NOT_READY", checks)
        if not checks["explicit_enablement"]:
            return ProductionEligibility(
                "BLOCKED", "EXPLICIT_ENABLEMENT_REQUIRED", checks
            )
        return ProductionEligibility("PASS", "ELIGIBLE", checks)


def explicit_enablement(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return str(values.get(ENABLE_FLAG, "false")).strip().lower() == "true"


@dataclass(frozen=True)
class StrategyRegistration:
    spec: FundamentalStrategySpec
    adapter: FundamentalRuntimeAdapter
    gate: ProductionEligibilityGate


class StrategyRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, StrategyRegistration] = {}

    def register(self, registration: StrategyRegistration) -> None:
        registration.spec.validate()
        strategy_id = registration.spec.strategy_id
        if strategy_id in self._entries:
            raise ProductionIntegrationError("DUPLICATE_PRODUCTION_STRATEGY")
        self._entries[strategy_id] = registration

    def get(self, strategy_id: str) -> StrategyRegistration:
        try:
            return self._entries[strategy_id]
        except KeyError as exc:
            raise ProductionIntegrationError("UNKNOWN_PRODUCTION_STRATEGY") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


class FundamentalRuntimeAdapter:
    """Input-only adapter for the already validated Fundamental runtime."""

    def __init__(
        self,
        snapshot_provider: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.snapshot_provider = snapshot_provider

    def snapshot(self, *, asof_date: str, **kwargs: Any) -> list[Mapping[str, Any]]:
        if self.snapshot_provider is None:
            raise ProductionIntegrationError(
                "CANONICAL_FUNDAMENTAL_RUNTIME_UNAVAILABLE"
            )
        rows = list(self.snapshot_provider(asof_date=asof_date, **kwargs))
        validate_pit_snapshot(rows, asof_date)
        return rows

    def normalize(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        asof_date: str,
        rebalance_flag: bool,
        prior_targets: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        return normalize_recommendations(
            rows,
            strategy_id=STRATEGY_ID,
            asof_date=asof_date,
            rebalance_flag=rebalance_flag,
            prior_targets=prior_targets,
        )


def validate_pit_snapshot(
    rows: Iterable[Mapping[str, Any]], execution_asof: str | date
) -> None:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return
    missing = PIT_COLUMNS - set(frame.columns)
    if missing:
        raise ProductionIntegrationError("PIT_INTEGRITY_FAIL")
    for column in ("available_date", "publication_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    asof = pd.Timestamp(execution_asof)
    pit_status = frame["pit_status"].astype(str).str.upper()
    if (
        frame["available_date"].isna().any()
        or frame["publication_date"].isna().any()
        or (frame["publication_date"] > frame["available_date"]).any()
        or (frame["available_date"] > asof).any()
        or (
            ~(pit_status.str.contains("PIT") | pit_status.eq("PUBLICATION_DATE_AWARE"))
        ).any()
    ):
        raise ProductionIntegrationError("PIT_INTEGRITY_FAIL")


def _as_bool(value: Any) -> bool:
    return bool(value) if not isinstance(value, str) else value.lower() == "true"


def normalize_recommendations(
    rows: Iterable[Mapping[str, Any]],
    *,
    strategy_id: str,
    asof_date: str,
    rebalance_flag: bool,
    prior_targets: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Map canonical output only; no factor or ranking calculation occurs here."""

    if not rebalance_flag:
        carried = []
        for row in prior_targets:
            item = dict(row)
            item["strategy_id"] = strategy_id
            item["asof_date"] = asof_date
            item["reason"] = "REB60_CARRY_FORWARD"
            carried.append(item)
        return carried
    result: list[dict[str, Any]] = []
    for source in rows:
        stock_id = source.get("stock_id", source.get("ticker"))
        if stock_id is None:
            raise ProductionIntegrationError("RECOMMENDATION_SCHEMA_INVALID")
        result.append(
            {
                "strategy_id": strategy_id,
                "stock_id": str(stock_id),
                "score": float(source["score"]),
                "rank": int(source["rank"]),
                "selected": _as_bool(source.get("selected", True)),
                "target_weight": float(source["target_weight"]),
                "reason": str(source.get("reason", "G2_G3_SCORE_WEIGHTED")),
                "asof_date": asof_date,
                "rebalance_flag": True,
            }
        )
    selected = [row for row in result if row["selected"]]
    if len(selected) > 5:
        raise ProductionIntegrationError("TOP_N_CONTRACT_VIOLATION")
    return sorted(result, key=lambda row: (row["rank"], row["stock_id"]))


def hash_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    return sha256_payload(list(rows))


def compare_snapshots(
    research: Sequence[Mapping[str, Any]],
    shadow: Sequence[Mapping[str, Any]],
    production: Sequence[Mapping[str, Any]],
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    def canonical(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {field: row.get(field) for field in PARITY_FIELDS}
            for row in sorted(rows, key=lambda item: str(item.get("stock_id", "")))
        ]

    sources = {
        "research": canonical(research),
        "shadow": canonical(shadow),
        "production": canonical(production),
    }
    mismatches: list[str] = []
    baseline = sources["research"]
    for name in ("shadow", "production"):
        other = sources[name]
        if len(other) != len(baseline):
            mismatches.append(f"{name}:row_count")
            continue
        for index, (left, right) in enumerate(zip(baseline, other, strict=True)):
            for field in PARITY_FIELDS:
                value, expected = left[field], right[field]
                if isinstance(value, (int, float)) or isinstance(
                    expected, (int, float)
                ):
                    if not math.isclose(
                        float(value or 0), float(expected or 0), abs_tol=atol
                    ):
                        mismatches.append(f"{name}:{index}:{field}")
                elif value != expected:
                    mismatches.append(f"{name}:{index}:{field}")
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "research_shadow_production_parity": "PASS" if not mismatches else "FAIL",
        "fields": list(PARITY_FIELDS),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "selection_hash": hash_rows(baseline),
        "weights_hash": hash_rows(
            [
                {
                    "stock_id": row.get("stock_id"),
                    "target_weight": row.get("target_weight"),
                }
                for row in baseline
            ]
        ),
        "evidence": sources,
    }


def build_default_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    spec = FundamentalStrategySpec()
    registry.register(
        StrategyRegistration(
            spec=spec,
            adapter=FundamentalRuntimeAdapter(),
            gate=ProductionEligibilityGate(),
        )
    )
    return registry


def reject_broker_submission(*_: Any, **__: Any) -> None:
    raise ProductionIntegrationError("BROKER_ORDER_SUBMISSION_DISABLED")


__all__ = [
    "CURRENT_PROMOTION_EVIDENCE",
    "ENABLE_FLAG",
    "FROZEN_PROMOTION_STATE",
    "FundamentalRuntimeAdapter",
    "FundamentalStrategySpec",
    "ProductionEligibility",
    "ProductionEligibilityGate",
    "ProductionIntegrationError",
    "StrategyRegistration",
    "StrategyRegistry",
    "build_current_promotion_evidence",
    "build_default_registry",
    "compare_snapshots",
    "explicit_enablement",
    "hash_rows",
    "normalize_recommendations",
    "reject_broker_submission",
    "sha256_payload",
    "validate_pit_snapshot",
    "validate_promotion_evidence",
]

from __future__ import annotations

import pandas as pd

from twse_factor_lab.production.final_runtime import (
    AtomicRecommendationStore,
    CanonicalFundamentalRuntimeProvider,
    build_daily_recommendations,
    build_real_snapshot_parity,
    build_three_runtime_snapshots,
    canonical_factor_rows,
    line_format,
    run_daily_fundamental,
    trading_session_rebalance_due,
    web_serialize,
)
from twse_factor_lab.production.fundamental import (
    build_current_promotion_evidence,
    sha256_payload,
)


def _rows(as_of: str = "2024-01-02") -> list[dict]:
    return [
        {
            "ticker": ticker,
            "as_of_date": as_of,
            "universe_eligible": True,
            "eligible": True,
            "g2_raw": 0.1 * rank,
            "g3_raw": 0.2 * rank,
            "normalized_g2": 0.3 * rank,
            "normalized_g3": 0.4 * rank,
            "composite_score": 0.35 * rank,
            "rank": rank,
            "selected": True,
            "target_weight": 1 / 5,
            "fundamental_period_end": "2023-09-30",
            "fundamental_available_date": "2023-11-15",
            "publication_date": "2023-11-14",
            "source": "canonical-test-fixture",
            "data_source": "canonical-test-fixture",
            "rebalance_flag": True,
            "reason": "CANONICAL",
        }
        for rank, ticker in enumerate(("1001", "1002", "1003", "1004", "1005"), 1)
    ]


def _eligible_evidence() -> dict:
    evidence = build_current_promotion_evidence()
    evidence.pop("artifact_sha256")
    evidence["fresh_oos_status"] = "PASS"
    evidence["production_promotion_status"] = "PRODUCTION_APPROVED"
    evidence["production_ready"] = True
    evidence["artifact_sha256"] = sha256_payload(evidence)
    return evidence


def test_provider_emits_canonical_rows_and_rejects_future_data() -> None:
    baseline = _rows()
    baseline[0]["fundamental_available_date"] = "2024-01-01"
    baseline[1]["fundamental_available_date"] = "2024-01-02"
    provider = CanonicalFundamentalRuntimeProvider(
        baseline,
        market_sessions=["2024-01-02"],
        security_master={str(1000 + i) for i in range(1, 6)},
    )
    result = provider.snapshot("2024-01-02")
    assert len(result) == 5
    assert result[0]["score"] == result[0]["composite_score"]
    future = _rows()
    future[0]["fundamental_available_date"] = "2024-01-03"
    try:
        CanonicalFundamentalRuntimeProvider(future).snapshot("2024-01-02")
    except ValueError as exc:
        assert str(exc) == "PIT_INTEGRITY_FAIL"
    else:
        raise AssertionError("future PIT row was accepted")


def test_canonical_factor_rows_use_available_pit_records_only() -> None:
    records = []
    for ticker, operating, eps in (("1001", 120, 12), ("1002", 110, 11)):
        for metric, prior, current in (
            ("operating_income", 100, operating),
            ("eps", 10, eps),
        ):
            for period_end, available_date, value in (
                ("2023-03-31", "2023-05-15", prior),
                ("2024-03-31", "2024-05-15", current),
            ):
                records.append(
                    {
                        "ticker": ticker,
                        "metric": metric,
                        "period_end": period_end,
                        "publication_date": available_date,
                        "available_date": available_date,
                        "value": value,
                    }
                )
    records.append(dict(records[-1], ticker="9999", available_date="2024-05-16"))
    universe = pd.DataFrame(
        {
            "date": ["2024-05-15"] * 2,
            "ticker": ["1001", "1002"],
            "is_eligible": [True, True],
        }
    )
    rows = canonical_factor_rows(pd.DataFrame(records), universe, "2024-05-15")
    assert [row["ticker"] for row in rows] == ["1001", "1002"]
    assert round(rows[0]["g2_raw"], 12) == 0.2
    assert round(rows[0]["g3_raw"], 12) == 0.2
    assert round(sum(row["target_weight"] for row in rows), 12) == 1.0
    assert rows[0]["eligible"] is True
    assert rows[0]["source"] == "canonical_fundamental_pit"


def test_real_parity_requires_non_empty_runtime_rows() -> None:
    blocked = build_real_snapshot_parity([], [], [], dates=["2024-01-02"])
    assert blocked["status"] == "BLOCKED"
    assert blocked["empty_output_parity_accepted"] is False
    rows = _rows()
    passed = build_real_snapshot_parity(rows, rows, rows, dates=["2024-01-02"])
    assert passed["status"] == "PASS"
    assert passed["source_runtime_available"] is True


def test_three_runtime_paths_are_invoked_independently() -> None:
    calls: list[str] = []
    dates = ["2024-01-02", "2024-01-03"]

    def runtime(name: str):
        def run(as_of: str):
            calls.append(f"{name}:{as_of}")
            return _rows(as_of)

        return run

    provider = CanonicalFundamentalRuntimeProvider(_rows())
    snapshots = build_three_runtime_snapshots(
        provider,
        dates,
        research_runtime=runtime("research"),
        shadow_runtime=runtime("shadow"),
        production_runtime=runtime("production"),
    )
    assert all(len(rows) == 10 for rows in snapshots.values())
    assert len(calls) == 6


def test_reb60_uses_trading_sessions_and_no_retarget() -> None:
    sessions = (
        pd.date_range("2024-01-01", periods=121, freq="B").strftime("%Y-%m-%d").tolist()
    )
    assert trading_session_rebalance_due(sessions, sessions[0]) is True
    assert trading_session_rebalance_due(sessions, sessions[59], sessions[0]) is False
    assert trading_session_rebalance_due(sessions, sessions[60], sessions[0]) is True
    assert trading_session_rebalance_due(sessions, sessions[61], sessions[0]) is True
    assert trading_session_rebalance_due(sessions, sessions[119], sessions[0]) is True
    assert trading_session_rebalance_due(sessions, sessions[120], sessions[0]) is True
    try:
        trading_session_rebalance_due(sessions, "2024-01-06", sessions[0])
    except ValueError as exc:
        assert str(exc) == "AS_OF_DATE_NOT_IN_MARKET_CALENDAR"
    else:
        raise AssertionError("non-session date was accepted")
    carried = build_daily_recommendations(
        _rows(), as_of_date=sessions[1], rebalance_due=False, prior_targets=_rows()
    )
    assert carried and {row.action for row in carried} == {"NO_REBALANCE"}


def test_persistence_is_idempotent_and_consumer_contracts(tmp_path) -> None:
    rows = _rows()
    recs = build_daily_recommendations(
        rows, as_of_date="2024-01-02", rebalance_due=True
    )
    store = AtomicRecommendationStore(tmp_path)
    for _ in range(10):
        assert store.persist(recs) == 5
    saved = pd.read_parquet(store.path)
    assert len(saved) == 5
    assert web_serialize(recs, data_status="PASS")["top5"]
    assert "1001" in line_format(recs, data_status="PASS")


def test_eligible_daily_run_can_write_but_broker_is_never_called(tmp_path) -> None:
    sessions = (
        pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
    )
    provider = CanonicalFundamentalRuntimeProvider(_rows(sessions[0]))
    result = run_daily_fundamental(
        provider=provider,
        as_of_date=sessions[0],
        sessions=sessions,
        evidence=_eligible_evidence(),
        explicit_enable=True,
        write_recommendations=True,
        store=AtomicRecommendationStore(tmp_path),
    )
    assert result["status"] == "PASS"
    assert result["health"]["recommendations_written"] == 5
    assert result["health"]["broker_submission"] == "DISABLED"


def test_daily_fail_closed_gates_and_provider_emit_no_recommendations(tmp_path) -> None:
    sessions = (
        pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
    )
    for gate_name in (
        "PIT_INTEGRITY",
        "DATA_FRESHNESS_GATE",
        "FACTOR_HEALTH_GATE",
        "RESEARCH_RUNTIME_PARITY",
    ):
        evidence = _eligible_evidence()
        evidence.pop("artifact_sha256")
        evidence["gates"][gate_name] = "FAIL"
        evidence["artifact_sha256"] = sha256_payload(evidence)
        result = run_daily_fundamental(
            provider=CanonicalFundamentalRuntimeProvider(_rows(sessions[0])),
            as_of_date=sessions[0],
            sessions=sessions,
            evidence=evidence,
            explicit_enable=True,
            write_recommendations=True,
            store=AtomicRecommendationStore(tmp_path / gate_name),
        )
        assert result["status"] == "BLOCKED"
        assert result["recommendations"] == []
    future = _rows(sessions[0])
    future[0]["fundamental_available_date"] = sessions[1]
    result = run_daily_fundamental(
        provider=CanonicalFundamentalRuntimeProvider(future),
        as_of_date=sessions[0],
        sessions=sessions,
        evidence=_eligible_evidence(),
        explicit_enable=True,
        write_recommendations=True,
        store=AtomicRecommendationStore(tmp_path / "provider"),
    )
    assert result["status"] == "BLOCKED"
    assert result["recommendations"] == []

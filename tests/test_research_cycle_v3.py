"""Regression checks for the completed, fail-fast v3 research evidence."""

from __future__ import annotations

import json
from pathlib import Path

from twse_factor_lab.acceptance.research_cycle import verify_research_freeze

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ID = "quality-cost-breadth-v3-expanded-fundamentals"
BASE = ROOT / "data" / "research" / RESEARCH_ID


def _json(name: str) -> dict:
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def test_v3_uses_remediated_pit_and_preserves_fail_fast_policy() -> None:
    preflight = _json("data_preflight_report.json")
    evidence = _json("factor_evidence.json")
    stage = _json("strategy_stage.json")

    assert preflight["formal_fundamental_input"] == "data/processed/fundamental_pit_v2"
    assert preflight["primary_horizon"] == 20
    assert preflight["integrity"]["enumeration_status"] == "FULL_ENUMERATION"
    assert preflight["integrity"]["silently_unprocessed_ticker_count"] == 0
    assert {row["factor"] for row in evidence["rows"]} == {"eps", "roe"}
    assert all(row["verdict"] == "REJECT" for row in evidence["rows"])
    assert evidence["admitted_factor_pool"] == []
    assert stage["executed"] is False
    assert _json("candidate_lock.json")["status"] == "NOT_RUN"
    assert _json("breadth_report.json")["status"] == "NOT_RUN"


def test_v3_freeze_is_verifiable() -> None:
    assert verify_research_freeze(ROOT, RESEARCH_ID)["status"] == "PASS"

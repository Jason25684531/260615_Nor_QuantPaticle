"""Operational tests R1-R28 for the frozen, broker-free S3 shadow runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twse_factor_lab.runtime.shadow import (
    FINGERPRINT,
    ShadowRuntime,
    ShadowRuntimeError,
)

ROOT = Path(__file__).resolve().parents[1]


def market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-11-03", "2026-01-05")
    tickers = [str(1000 + number) for number in range(6)]
    close = pd.DataFrame(
        {
            ticker: 20 + number + np.arange(len(dates)) / 10
            for number, ticker in enumerate(tickers)
        },
        index=dates,
    )
    volume = pd.DataFrame({ticker: 5_000_000 for ticker in tickers}, index=dates)
    universe = pd.DataFrame({"ticker": tickers, "listed_date": "2020-01-01"})
    return close, volume, universe


def test_r1_to_r8_freeze_two_phase_and_no_future_price(tmp_path):
    close, volume, universe = market()
    runtime = ShadowRuntime(ROOT, tmp_path / "shadow")
    first = runtime.run(close, volume, universe, "2026-01-01", forward=False)
    assert first["status"] == "PASS"
    contract = json.loads(
        (tmp_path / "shadow/runtime_shadow_contract.json").read_text()
    )
    assert contract["candidate_fingerprint"] == FINGERPRINT
    assert contract["strategy"]["top_n"] == 5
    assert contract["strategy"]["rebalance"] == "MONTHLY"
    assert contract["strategy"]["buffer"] == "OFF"
    assert len(pd.read_parquet(tmp_path / "shadow/shadow_orders.parquet")) == 5
    assert pd.read_parquet(tmp_path / "shadow/shadow_fills.parquet").empty
    second = runtime.run(close, volume, universe, "2026-01-02", forward=False)
    assert second["filled"] == 5
    assert len(pd.read_parquet(tmp_path / "shadow/shadow_fills.parquet")) == 5


def test_r9_to_r16_halts_is_idempotent_and_recovers_state(tmp_path):
    close, volume, universe = market()
    runtime = ShadowRuntime(ROOT, tmp_path / "shadow")
    bad = close.copy()
    bad.loc[pd.Timestamp("2026-01-01"), bad.columns[0]] = np.nan
    assert (
        runtime.run(bad, volume, universe, "2026-01-01", forward=False)["status"]
        == "SAFE_HALT"
    )
    runtime.run(close, volume, universe, "2026-01-01", forward=False)
    runtime.run(close, volume, universe, "2026-01-02", forward=False)
    before = (
        pd.read_parquet(tmp_path / "shadow/shadow_orders.parquet").shape,
        pd.read_parquet(tmp_path / "shadow/shadow_fills.parquet").shape,
    )
    runtime.run(close, volume, universe, "2026-01-02", forward=False)
    after = (
        pd.read_parquet(tmp_path / "shadow/shadow_orders.parquet").shape,
        pd.read_parquet(tmp_path / "shadow/shadow_fills.parquet").shape,
    )
    assert before == after
    (tmp_path / "shadow/runtime_state.json").write_text("{")
    recovered = ShadowRuntime(ROOT, tmp_path / "shadow")
    assert recovered._state()["cash"] >= 0
    assert recovered.failure_injection_report()["status"] == "PASS"


def test_r17_to_r28_reconciliation_gate_and_governance(tmp_path):
    close, volume, universe = market()
    runtime = ShadowRuntime(ROOT, tmp_path / "shadow")
    runtime.run(close, volume, universe, "2026-01-01", forward=False)
    outcome = runtime.run(close, volume, universe, "2026-01-02", forward=False)
    reconciliation = json.loads(
        (tmp_path / "shadow/research_runtime_reconciliation.json").read_text()
    )
    gate = outcome["gate"]
    assert reconciliation["status"] == "PASS"
    assert gate["verdict"] == "SHADOW_EXTEND"
    assert gate["production_ready"] is False
    with pytest.raises(ShadowRuntimeError, match="PROHIBITED"):
        runtime.broker_write()

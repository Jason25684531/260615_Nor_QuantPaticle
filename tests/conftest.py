from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twse_factor_lab.production.fundamental import FundamentalStrategySpec
from twse_factor_lab.reporting.performance_adapter import PerformanceData


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def performance_root(tmp_path: Path) -> Path:
    spec = FundamentalStrategySpec()
    final = tmp_path / "data/research/fundamental-production-final-v1"
    write_json(
        final / "production_runtime_contract.json",
        {
            "strategy_id": spec.strategy_id,
            "strategy_fingerprint": spec.fingerprint,
            "factors": list(spec.factors),
            "factor_weighting": spec.factor_weighting,
            "top_n": spec.top_n,
            "rebalance": spec.rebalance,
            "portfolio_weighting": spec.portfolio_weighting,
            "research_knowledge_cutoff": spec.research_knowledge_cutoff,
        },
    )
    write_json(
        final / "fresh_oos_eligibility_audit.json",
        {
            "fresh_oos_available": False,
            "fresh_oos_status": "INSUFFICIENT_DATA",
            "reason": "NO_LEGAL_UNTOUCHED_WINDOW_WITH_REQUIRED_HISTORY",
            "strategy_id": spec.strategy_id,
            "strategy_fingerprint": spec.fingerprint,
        },
    )
    write_json(
        final / "production_final_gate.json",
        {
            "production_eligibility": "BLOCKED",
            "production_ready": False,
            "production_block_reason": "INSUFFICIENT_OOS",
        },
    )
    return tmp_path


@pytest.fixture
def sample_performance_data() -> PerformanceData:
    dates = pd.date_range("2024-01-02", periods=130, freq="B")
    returns = pd.Series(
        0.001 + np.sin(np.arange(len(dates))) * 0.004,
        index=dates,
    )
    benchmark = pd.Series(0.0004, index=dates)
    positions = pd.DataFrame(
        {
            "date": dates,
            "ticker": "2330",
            "market_value": 800_000.0,
            "cash": 200_000.0,
            "portfolio_value": 1_000_000.0,
            "weight": 0.8,
        }
    )
    transactions = pd.DataFrame(
        {
            "date": [dates[0], dates[20]],
            "ticker": ["2330", "2330"],
            "amount": [1000.0, -1000.0],
            "price": [100.0, 110.0],
            "value": [100_000.0, -110_000.0],
            "commission": [100.0, 110.0],
            "tax": [0.0, 330.0],
            "slippage": [50.0, 55.0],
            "total_cost": [150.0, 495.0],
        }
    )
    return PerformanceData.from_frames(
        "BACKTEST",
        returns,
        benchmark_returns=benchmark,
        positions=positions,
        transactions=transactions,
        benchmark_id="FIXTURE",
        benchmark_source="fixture",
    )

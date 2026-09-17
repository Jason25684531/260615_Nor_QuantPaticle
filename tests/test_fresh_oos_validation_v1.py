"""Governance tests for the S3 Fresh OOS runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance.fresh_oos_validation import (
    ATOL,
    FINGERPRINT,
    RTOL,
    _contract,
    _parity,
    _scores_and_targets,
    classify_fresh_oos,
    contamination_audit,
    verify_candidate,
)
from twse_factor_lab.backtest.accounting import canonical_replay
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import _weights_matrix

ROOT = Path(__file__).resolve().parents[1]


def _market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-11-03", "2026-08-31")
    columns = [str(1000 + index) for index in range(6)]
    close = pd.DataFrame(
        {
            ticker: 20
            + index
            + pd.Series(range(len(dates)), index=dates) / (20 + index)
            for index, ticker in enumerate(columns)
        },
        index=dates,
        dtype=float,
    )
    volume = pd.DataFrame(
        {
            ticker: np.where(
                dates.month % len(columns) == index, 100_000_000, 5_000_000
            )
            for index, ticker in enumerate(columns)
        },
        index=dates,
    )
    universe = pd.DataFrame(
        {"ticker": columns, "listed_date": ["2020-01-01"] * len(columns)}
    )
    return close, volume, universe


def test_o1_to_o6_s3_lock_and_cost_are_unchanged():
    verified = verify_candidate(ROOT)
    assert verified["candidate_fingerprint"] == FINGERPRINT
    assert verified["frozen"]["weights"] == {
        "L2_AMIHUD_20D": 0.5,
        "L4_DOLLAR_VOLUME_20D": 0.5,
    }
    assert verified["frozen"]["top_n"] == 5
    assert verified["frozen"]["rebalance"] == "monthly"
    assert verified["frozen"]["buffer"] is False
    assert verified["cost"] == {
        "buy_fee_rate": 0.001425,
        "sell_fee_rate": 0.001425,
        "transaction_tax_rate": 0.003,
        "slippage_rate": 0.001,
    }


def test_o7_audit_precedes_performance_and_distinguishes_raw_data(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/ohlcv-2026.csv").write_text("raw", encoding="utf-8")
    audit = contamination_audit(tmp_path, tmp_path)
    assert audit["performed_before_performance_access"] is True
    assert audit["fresh_oos_eligible"] is True
    assert audit["raw_data_presence"]["status"] == "DATA_PRESENT"
    (tmp_path / "data/research").mkdir(parents=True)
    (tmp_path / "data/research/selection.md").write_text(
        "2026-03-01 Sharpe used to select candidate", encoding="utf-8"
    )
    assert contamination_audit(tmp_path, tmp_path)["status"] == "CONTAMINATED"


def test_o8_to_o13_contract_fresh_state_timing_and_replacement():
    close, volume, universe = _market()
    contract = _contract(close.index)
    assert contract["start_date"] == "2026-01-01"
    assert contract["end_date"] == "2026-08-31"
    assert contract["frozen_before_returns"] and contract["fresh_state"]
    scores, targets, oos_close = _scores_and_targets(close, volume, universe, contract)
    assert not scores.empty
    assert (targets["execution_date"] > targets["date"]).all()
    assert targets.groupby("execution_date").size().ge(5).all()
    assert (targets.groupby("execution_date")["target_weight"].sum() == 1.0).all()
    weights = _weights_matrix(targets, oos_close.index, oos_close.columns)
    result, _returns, _turnover, _orders = canonical_replay(
        oos_close, weights, CostModel(), 1_000_000
    )
    assert result.iloc[0]["cash"] == 1_000_000
    assert result.iloc[0].filter(like="position:").eq(0).all()
    # A full target matrix explicitly carries zero for tickers removed later.
    assert (targets["target_weight"] == 0.0).any()


def test_o15_o16_and_o19_tolerance_window_and_labels_are_fixed():
    assert ATOL == 4.547473508864641e-12
    assert RTOL == 1.4210854715202004e-14
    assert classify_fresh_oos(0.01, 0.01) == "SUPPORTIVE"
    assert classify_fresh_oos(0.01, -0.01) == "MIXED"
    assert classify_fresh_oos(-0.01, -0.01) == "ADVERSE"


def test_o14_three_engine_parity_uses_the_frozen_tolerance():
    close, volume, universe = _market()
    _scores, targets, oos_close = _scores_and_targets(
        close, volume, universe, _contract(close.index)
    )
    _result, _returns, _turnover, parity = _parity(oos_close, targets)
    assert parity["summary"]["status"] == "PASS"
    assert set(parity["detail"]["field"]) == {"cash", "returns", "equity"}

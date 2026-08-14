from __future__ import annotations

import pandas as pd
import pytest

from twse_factor_lab.analysis.pyfolio_adapter import to_pyfolio_inputs
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import _event_matrix, run_weight_backtest


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame(
        {"A": range(10, 20), "B": range(20, 30), "C": range(30, 40)},
        index=dates,
        dtype=float,
    )
    weights = pd.DataFrame(
        {
            "execution_date": [dates[1], dates[1], dates[1], dates[6], dates[6]],
            "ticker": ["A", "B", "C", "A", "C"],
            "target_weight": [0.5, 0.5, 0.0, 0.0, 1.0],
        }
    )
    return close, weights


def _zero_cost() -> CostModel:
    return CostModel(
        buy_fee_rate=0, sell_fee_rate=0, transaction_tax_rate=0, slippage_rate=0
    )


def test_vectorbt_uses_real_engine_and_execution_events_only(tmp_path):
    close, weights = _fixture()
    events = _event_matrix(weights, close.index, close.columns)
    assert events.loc[close.index[0]].isna().all()
    assert events.loc[close.index[2]].isna().all()
    assert events.loc[close.index[6], "B"] == 0
    results, metrics = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=weights,
        cost_model=_zero_cost(),
        initial_cash=1_000,
        top_n=2,
        allow_fallback=False,
        artifacts_dir=str(tmp_path),
    )
    assert metrics.loc[0, "actual_engine"] == "vectorbt"
    assert results.loc[2, "turnover"] == 0
    assert results.loc[6, "position:B"] == pytest.approx(0)
    for name in ["orders", "trades", "positions", "returns"]:
        assert (tmp_path / f"vectorbt_{name}.parquet").stat().st_size > 0


def test_custom_and_vectorbt_match_on_golden_fixture_without_costs():
    close, weights = _fixture()
    common = dict(
        close_matrix=close,
        portfolio_weights=weights,
        cost_model=_zero_cost(),
        initial_cash=1_000,
        top_n=2,
    )
    custom, _ = run_weight_backtest(**common, use_vectorbt=False)
    vectorbt, metrics = run_weight_backtest(**common, allow_fallback=False)
    assert custom["turnover"].to_list() == [0, 1, 0, 0, 0, 0, 2, 0, 0, 0]
    assert custom["cost_returns"].to_list() == [0] * 10
    assert custom["equity"].to_numpy() == pytest.approx(
        [
            1000,
            1000,
            1069.264069,
            1138.528139,
            1207.792208,
            1277.056277,
            1346.320346,
            1383.718134,
            1421.115921,
            1458.513709,
        ]
    )
    assert metrics.loc[0, "actual_engine"] == "vectorbt"
    assert vectorbt["equity"].to_numpy() == pytest.approx(custom["equity"].to_numpy())


def test_fallback_labels_and_adapter_contract(monkeypatch):
    close, weights = _fixture()
    import twse_factor_lab.backtest.vectorbt_engine as engine

    monkeypatch.setattr(
        engine, "_vectorbt_backtest", lambda *args: (_ for _ in ()).throw(ImportError())
    )
    results, metrics = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=weights,
        cost_model=_zero_cost(),
        initial_cash=1_000,
        top_n=2,
        use_vectorbt=True,
    )
    assert metrics.loc[0, "actual_engine"] == "fallback_custom"
    returns, positions, transactions = to_pyfolio_inputs(results)
    assert returns.index.is_unique and returns.index.is_monotonic_increasing
    assert "cash" in positions and transactions.empty
    with pytest.raises(ImportError):
        run_weight_backtest(
            close_matrix=close,
            portfolio_weights=weights,
            cost_model=_zero_cost(),
            initial_cash=1_000,
            top_n=2,
            allow_fallback=False,
        )

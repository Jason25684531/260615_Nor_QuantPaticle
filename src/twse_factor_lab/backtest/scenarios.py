"""Scenario runner for cost sensitivity, Top N, and rebalance frequency analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.portfolio.rebalance import (
    build_rebalance_calendar,
    validate_factor_eligibility,
)
from twse_factor_lab.portfolio.selection import build_topn_positions
from twse_factor_lab.portfolio.weights import build_equal_weight_portfolio

logger = logging.getLogger(__name__)

_SNAPSHOT_FACTORS = frozenset(
    ["pb_inverse", "pe_inverse", "dividend_yield", "latest_snapshot_mixed"]
)


def _cost_model_from_scenario(scenario: dict[str, Any]) -> CostModel:
    return CostModel(
        buy_fee_rate=float(scenario.get("buy_fee_rate", 0.001425)),
        sell_fee_rate=float(scenario.get("sell_fee_rate", 0.001425)),
        transaction_tax_rate=float(scenario.get("transaction_tax_rate", 0.003)),
        slippage_rate=float(scenario.get("slippage_rate", 0.001)),
    )


def _build_weights_for_topn(
    factors_composite: pd.DataFrame,
    *,
    factor_name: str,
    top_n: int,
    rebalance_dates: pd.DatetimeIndex | None,
    hold_until_drop: bool,
    drop_rank_buffer: int,
    rebalance_frequency: str,
    execution_lag_days: int,
) -> pd.DataFrame:
    positions = build_topn_positions(
        factors_composite,
        top_n=top_n,
        factor_name=factor_name,
        rebalance_dates=rebalance_dates,
        hold_until_drop=hold_until_drop,
        drop_rank_buffer=drop_rank_buffer,
        rebalance_frequency=rebalance_frequency,
    )
    return build_equal_weight_portfolio(
        positions,
        execution_lag_days=execution_lag_days,
    )


def _run_single_backtest(
    close_matrix: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    cost_model: CostModel,
    top_n: int,
    initial_cash: float,
) -> pd.DataFrame:
    _, metrics = run_weight_backtest(
        close_matrix=close_matrix,
        portfolio_weights=portfolio_weights,
        cost_model=cost_model,
        initial_cash=initial_cash,
        top_n=top_n,
        use_vectorbt=False,
        allow_fallback=True,
    )
    return metrics


def run_cost_scenarios(
    close_matrix: pd.DataFrame,
    factors_composite: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run the same strategy under multiple cost scenarios."""
    backtest_cfg = config.get("backtest", {}) or {}
    factor_name = str(backtest_cfg.get("factor_name", "historical_price_volume"))
    top_n = int(backtest_cfg.get("top_n", 20))
    execution_lag_days = int(backtest_cfg.get("execution_lag_days", 1))
    initial_cash = float(backtest_cfg.get("initial_cash", 1_000_000))
    rules = backtest_cfg.get("rules", {}) or {}
    hold_until_drop = bool(rules.get("hold_until_drop", False))
    drop_rank_buffer = int(rules.get("drop_rank_buffer", 30))

    try:
        validate_factor_eligibility(factor_name)
    except ValueError as exc:
        logger.warning("Factor excluded: %s", exc)
        return pd.DataFrame(
            columns=[
                "scenario",
                "factor_name",
                "top_n",
                "rebalance_frequency",
                "buffer_enabled",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "turnover",
                "avg_exposure",
                "cost_drag",
                "engine",
                "start_date",
                "end_date",
            ]
        )

    cost_cfg = backtest_cfg.get("cost_sensitivity", {}) or {}
    scenarios_list = cost_cfg.get("scenarios", [])
    if not scenarios_list:
        fees = backtest_cfg.get("fees", {}) or {}
        scenarios_list = [{"name": "base_cost", **fees}]

    rebalance_frequency = str(backtest_cfg.get("rebalance_frequency", "daily"))
    calendar = build_rebalance_calendar(
        pd.DatetimeIndex(close_matrix.index),
        frequency=rebalance_frequency,
        execution_lag_days=execution_lag_days,
    )
    rebalance_dates = (
        pd.DatetimeIndex(calendar["signal_date"]) if not calendar.empty else None
    )

    portfolio_weights = _build_weights_for_topn(
        factors_composite,
        factor_name=factor_name,
        top_n=top_n,
        rebalance_dates=rebalance_dates,
        hold_until_drop=hold_until_drop,
        drop_rank_buffer=drop_rank_buffer,
        rebalance_frequency=rebalance_frequency,
        execution_lag_days=execution_lag_days,
    )

    results = []
    no_cost_return: float | None = None

    for scenario in scenarios_list:
        name = str(scenario.get("name", "unnamed"))
        cost_model = _cost_model_from_scenario(scenario)
        try:
            metrics = _run_single_backtest(
                close_matrix, portfolio_weights, cost_model, top_n, initial_cash
            )
            row = metrics.iloc[0]
            total_return = float(row["total_return"])
            if name == "no_cost":
                no_cost_return = total_return
            results.append(
                {
                    "scenario": name,
                    "factor_name": factor_name,
                    "top_n": top_n,
                    "rebalance_frequency": rebalance_frequency,
                    "buffer_enabled": hold_until_drop,
                    "total_return": total_return,
                    "annualized_return": float(row["annualized_return"]),
                    "annualized_volatility": float(row["annualized_volatility"]),
                    "sharpe": float(row["sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "turnover": float(row["turnover"]),
                    "avg_exposure": float(row["avg_exposure"]),
                    "cost_drag": np.nan,
                    "engine": str(row["engine"]),
                    "start_date": str(row["start_date"]),
                    "end_date": str(row["end_date"]),
                }
            )
        except Exception as exc:
            logger.warning("Scenario %s failed: %s", name, exc)

    df = pd.DataFrame(results)
    if not df.empty and no_cost_return is not None:
        df["cost_drag"] = no_cost_return - df["total_return"]
    return df


def run_topn_sensitivity(
    close_matrix: pd.DataFrame,
    factors_composite: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run backtests for each top_n in top_n_grid."""
    backtest_cfg = config.get("backtest", {}) or {}
    factor_name = str(backtest_cfg.get("factor_name", "historical_price_volume"))
    top_n_grid = list(backtest_cfg.get("top_n_grid", [10, 20, 30]))
    rebalance_frequency = str(backtest_cfg.get("rebalance_frequency", "daily"))
    execution_lag_days = int(backtest_cfg.get("execution_lag_days", 1))
    initial_cash = float(backtest_cfg.get("initial_cash", 1_000_000))
    rules = backtest_cfg.get("rules", {}) or {}
    hold_until_drop = bool(rules.get("hold_until_drop", False))
    drop_rank_buffer = int(rules.get("drop_rank_buffer", 30))
    fees = backtest_cfg.get("fees", {}) or {}
    cost_model = CostModel(
        buy_fee_rate=float(fees.get("buy_fee_rate", 0.001425)),
        sell_fee_rate=float(fees.get("sell_fee_rate", 0.001425)),
        transaction_tax_rate=float(fees.get("transaction_tax_rate", 0.003)),
        slippage_rate=float(fees.get("slippage_rate", 0.001)),
    )

    try:
        validate_factor_eligibility(factor_name)
    except ValueError as exc:
        logger.warning("Factor excluded: %s", exc)
        return pd.DataFrame(
            columns=[
                "top_n",
                "factor_name",
                "rebalance_frequency",
                "buffer_enabled",
                "total_return",
                "annualized_return",
                "sharpe",
                "max_drawdown",
                "turnover",
                "avg_exposure",
                "engine",
                "notes",
            ]
        )

    calendar = build_rebalance_calendar(
        pd.DatetimeIndex(close_matrix.index),
        frequency=rebalance_frequency,
        execution_lag_days=execution_lag_days,
    )
    rebalance_dates = (
        pd.DatetimeIndex(calendar["signal_date"]) if not calendar.empty else None
    )

    results = []
    for top_n in top_n_grid:
        try:
            weights = _build_weights_for_topn(
                factors_composite,
                factor_name=factor_name,
                top_n=top_n,
                rebalance_dates=rebalance_dates,
                hold_until_drop=hold_until_drop,
                drop_rank_buffer=drop_rank_buffer,
                rebalance_frequency=rebalance_frequency,
                execution_lag_days=execution_lag_days,
            )
            metrics = _run_single_backtest(
                close_matrix, weights, cost_model, top_n, initial_cash
            )
            row = metrics.iloc[0]
            available_universe = factors_composite[
                (factors_composite["composite_type"] == factor_name)
                & (~factors_composite["is_snapshot_component_used"].astype(bool))
            ]["ticker"].nunique()
            notes = ""
            if available_universe < top_n:
                notes = f"universe_size={available_universe} < top_n={top_n}"
            results.append(
                {
                    "top_n": top_n,
                    "factor_name": factor_name,
                    "rebalance_frequency": rebalance_frequency,
                    "buffer_enabled": hold_until_drop,
                    "total_return": float(row["total_return"]),
                    "annualized_return": float(row["annualized_return"]),
                    "sharpe": float(row["sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "turnover": float(row["turnover"]),
                    "avg_exposure": float(row["avg_exposure"]),
                    "engine": str(row["engine"]),
                    "notes": notes,
                }
            )
        except Exception as exc:
            logger.warning("top_n=%s failed: %s", top_n, exc)

    return pd.DataFrame(results)


def run_rebalance_sensitivity(
    close_matrix: pd.DataFrame,
    factors_composite: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run backtests for each frequency in rebalance_frequency_grid."""
    backtest_cfg = config.get("backtest", {}) or {}
    factor_name = str(backtest_cfg.get("factor_name", "historical_price_volume"))
    top_n = int(backtest_cfg.get("top_n", 20))
    freq_grid = list(
        backtest_cfg.get("rebalance_frequency_grid", ["daily", "weekly", "monthly"])
    )
    execution_lag_days = int(backtest_cfg.get("execution_lag_days", 1))
    initial_cash = float(backtest_cfg.get("initial_cash", 1_000_000))
    rules = backtest_cfg.get("rules", {}) or {}
    hold_until_drop = bool(rules.get("hold_until_drop", False))
    drop_rank_buffer = int(rules.get("drop_rank_buffer", 30))
    fees = backtest_cfg.get("fees", {}) or {}
    cost_model = CostModel(
        buy_fee_rate=float(fees.get("buy_fee_rate", 0.001425)),
        sell_fee_rate=float(fees.get("sell_fee_rate", 0.001425)),
        transaction_tax_rate=float(fees.get("transaction_tax_rate", 0.003)),
        slippage_rate=float(fees.get("slippage_rate", 0.001)),
    )

    try:
        validate_factor_eligibility(factor_name)
    except ValueError as exc:
        logger.warning("Factor excluded: %s", exc)
        return pd.DataFrame(
            columns=[
                "rebalance_frequency",
                "factor_name",
                "top_n",
                "buffer_enabled",
                "rebalance_count",
                "total_return",
                "annualized_return",
                "sharpe",
                "max_drawdown",
                "turnover",
                "avg_exposure",
                "engine",
                "notes",
            ]
        )

    results = []
    for freq in freq_grid:
        try:
            calendar = build_rebalance_calendar(
                pd.DatetimeIndex(close_matrix.index),
                frequency=freq,
                execution_lag_days=execution_lag_days,
            )
            rebalance_count = len(calendar)
            rebalance_dates = (
                pd.DatetimeIndex(calendar["signal_date"])
                if not calendar.empty
                else None
            )
            weights = _build_weights_for_topn(
                factors_composite,
                factor_name=factor_name,
                top_n=top_n,
                rebalance_dates=rebalance_dates,
                hold_until_drop=hold_until_drop,
                drop_rank_buffer=drop_rank_buffer,
                rebalance_frequency=freq,
                execution_lag_days=execution_lag_days,
            )
            metrics = _run_single_backtest(
                close_matrix, weights, cost_model, top_n, initial_cash
            )
            row = metrics.iloc[0]
            notes = f"rebalance_count={rebalance_count}"
            results.append(
                {
                    "rebalance_frequency": freq,
                    "factor_name": factor_name,
                    "top_n": top_n,
                    "buffer_enabled": hold_until_drop,
                    "rebalance_count": rebalance_count,
                    "total_return": float(row["total_return"]),
                    "annualized_return": float(row["annualized_return"]),
                    "sharpe": float(row["sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "turnover": float(row["turnover"]),
                    "avg_exposure": float(row["avg_exposure"]),
                    "engine": str(row["engine"]),
                    "notes": notes,
                }
            )
        except Exception as exc:
            logger.warning("frequency=%s failed: %s", freq, exc)

    return pd.DataFrame(results)

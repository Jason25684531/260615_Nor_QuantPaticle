"""Regression tests for pre-declared primary_horizon (no post-hoc best-IC selection)."""

import pandas as pd
import pytest

from tests.test_factor_gate import make_dataset, prepare_root
from twse_factor_lab.analysis.factor_gate import (
    FactorGateConfig,
    FactorGateError,
    _select_horizon,
    evaluate_factor,
)


def _metric(horizon: int, ic: float) -> dict[str, object]:
    return {
        "horizon": horizon,
        "mean_ic": ic,
        "top_bottom_spread": ic,
        "sample_status": "SUFFICIENT",
    }


# Task 2.4 fixture: 1D=0.01 / 5D=0.02 / 10D=0.03 / 20D=0.10 mean IC.
_METRICS = [_metric(1, 0.01), _metric(5, 0.02), _metric(10, 0.03), _metric(20, 0.10)]


def test_select_horizon_uses_declared_primary_not_best_ic():
    selected = _select_horizon(_METRICS, primary_horizon=5)
    assert selected["horizon"] == 5
    assert selected["mean_ic"] == 0.02  # not the best (20D's 0.10)


def test_changing_primary_horizon_changes_the_basis():
    selected = _select_horizon(_METRICS, primary_horizon=20)
    assert selected["horizon"] == 20
    assert selected["mean_ic"] == 0.10


def test_invalid_primary_horizon_fails_fast():
    with pytest.raises(FactorGateError):
        _select_horizon(_METRICS, primary_horizon=999)
    with pytest.raises(FactorGateError):
        FactorGateConfig(horizons=(1, 5, 10, 20), primary_horizon=999)


def _build_reversal_then_momentum_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor rank is constant; horizon=1 IC is weak/mixed, horizon=2 IC is strong.

    Day-steps alternate between anti-correlated (X) and strongly correlated (Y)
    returns, so 1-day IC averages toward ~0 while 2-day cumulative IC is ~+1.
    """
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    tickers = ["A", "B", "C", "D", "E"]
    factor_matrix = pd.DataFrame(
        [[1, 2, 3, 4, 5]] * 4, index=dates[:4], columns=tickers
    )

    x_step = [0.10, 0.08, 0.06, 0.04, 0.02]  # anti-correlated with rank
    y_step = [0.01, 0.05, 0.10, 0.20, 0.50]  # strongly correlated with rank
    steps = [x_step, y_step, x_step, y_step, x_step]

    close_rows = [[100.0] * 5]
    for step in steps:
        prev = close_rows[-1]
        close_rows.append([p * (1.0 + r) for p, r in zip(prev, step, strict=True)])
    close_matrix = pd.DataFrame(close_rows, index=dates, columns=tickers)
    return factor_matrix, close_matrix


def _evaluate(tmp_path, experiment_id: str, primary_horizon: int):
    factor_matrix, close_matrix = _build_reversal_then_momentum_dataset()
    config = FactorGateConfig(
        horizons=(1, 2),
        primary_horizon=primary_horizon,
        quantiles=5,
        min_assets=5,
        min_ic_observations=4,
        min_coverage=0.5,
        min_icir=0.0,
    )
    return evaluate_factor(
        factor_id="roe",
        research_id="factor-gate-test",
        experiment_id=experiment_id,
        root=prepare_root(tmp_path),
        factor_matrix=factor_matrix,
        close_matrix=close_matrix,
        dataset_manifest=make_dataset(),
        config=config,
    )


def test_declared_primary_horizon_drives_verdict_end_to_end(tmp_path):
    weak = _evaluate(tmp_path / "weak", "exp-h1", primary_horizon=1)
    assert weak.verdict_horizon == 1
    assert weak.primary_horizon == 1
    assert weak.selected_horizon == weak.verdict_horizon
    # Mixed-sign 1-day IC must not be silently swapped for the far-stronger 2D IC.
    assert weak.verdict == "REJECT"

    strong = _evaluate(tmp_path / "strong", "exp-h2", primary_horizon=2)
    assert strong.verdict_horizon == 2
    assert strong.primary_horizon == 2
    assert strong.mean_ic > weak.mean_ic
    assert strong.verdict != "REJECT"

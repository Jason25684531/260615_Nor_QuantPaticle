import pandas as pd
import pytest

from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.analysis.information_coefficient import (
    compute_information_coefficients,
    summarize_information_coefficients,
)
from twse_factor_lab.analysis.monotonicity import evaluate_monotonicity
from twse_factor_lab.analysis.preparation import select_historical_factor_matrices
from twse_factor_lab.analysis.quantile_returns import (
    assign_factor_quantiles,
    compute_quantile_returns,
)
from twse_factor_lab.analysis.turnover import compute_turnover_summary


def test_select_historical_factor_matrices_excludes_snapshot_inputs():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    price_volume_factors = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["1101", "1102", "1101", "1102"],
            "momentum_60d": [0.9, 0.1, 0.8, 0.2],
            "low_volatility_20d": [0.1, 0.9, 0.2, 0.8],
            "volume_ratio_5d_60d": [1.5, 0.5, 1.4, 0.6],
        }
    )
    composite_factors = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["1101", "1102", "1101", "1102"],
            "composite_score": [0.8, 0.2, 0.7, 0.3],
            "composite_type": [
                "historical_price_volume",
                "historical_price_volume",
                "latest_snapshot_mixed",
                "latest_snapshot_mixed",
            ],
            "is_snapshot_component_used": [False, False, True, True],
        }
    )

    matrices, excluded = select_historical_factor_matrices(
        price_volume_factors=price_volume_factors,
        composite_factors=composite_factors,
        historical_factors=[
            "momentum_60d",
            "low_volatility_20d",
            "volume_ratio_5d_60d",
        ],
        optional_composites=["historical_price_volume"],
        excluded_snapshot=[
            "pb_inverse",
            "pe_inverse",
            "dividend_yield",
            "latest_snapshot_mixed",
        ],
    )

    assert set(matrices) == {
        "momentum_60d",
        "low_volatility_20d",
        "volume_ratio_5d_60d",
        "historical_price_volume",
    }
    assert "latest_snapshot_mixed" not in matrices
    assert excluded["pb_inverse"] == "snapshot_only_not_historical_ready"
    assert excluded["latest_snapshot_mixed"] == "snapshot_only_not_historical_ready"


def test_build_forward_returns_uses_future_prices_without_shifting_factor_dates():
    close_matrix = pd.DataFrame(
        {
            "1101": [100.0, 110.0, 121.0],
            "1102": [50.0, 55.0, 60.5],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    forward_returns = build_forward_returns(close_matrix, horizons=[1])

    row = forward_returns[
        (forward_returns["date"] == pd.Timestamp("2024-01-01"))
        & (forward_returns["ticker"] == "1101")
        & (forward_returns["horizon"] == 1)
    ].iloc[0]

    assert row["forward_return"] == pytest.approx(0.1)
    assert pd.Timestamp("2024-01-03") not in forward_returns["date"].tolist()


def test_information_coefficient_summary_reports_nan_ir_when_std_is_zero():
    factor_matrix = pd.DataFrame(
        {
            "1101": [1.0, 1.0],
            "1102": [2.0, 2.0],
            "1103": [3.0, 3.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    forward_returns = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3,
            "ticker": ["1101", "1102", "1103"] * 2,
            "horizon": [1] * 6,
            "forward_return": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        }
    )

    ic_results = compute_information_coefficients(
        factor_matrices={"momentum_60d": factor_matrix},
        forward_returns=forward_returns,
    )
    summary = summarize_information_coefficients(ic_results)

    assert ic_results["ic"].tolist() == pytest.approx([1.0, 1.0])
    assert summary.loc[0, "ic_mean"] == pytest.approx(1.0)
    assert summary.loc[0, "ic_std"] == pytest.approx(0.0)
    assert pd.isna(summary.loc[0, "ir"])


def test_quantile_returns_respect_lower_is_better_direction():
    factor_matrix = pd.DataFrame(
        {
            "1101": [1.0],
            "1102": [2.0],
            "1103": [3.0],
            "1104": [4.0],
            "1105": [5.0],
        },
        index=pd.to_datetime(["2024-01-01"]),
    )
    assignments = assign_factor_quantiles(
        factor_matrices={"low_volatility_20d": factor_matrix},
        directions={"low_volatility_20d": "lower_is_better"},
        quantiles=5,
    )
    forward_returns = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 5,
            "ticker": ["1101", "1102", "1103", "1104", "1105"],
            "horizon": [1] * 5,
            "forward_return": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )

    quantile_returns = compute_quantile_returns(assignments, forward_returns)

    best_bucket = quantile_returns[
        (quantile_returns["factor"] == "low_volatility_20d")
        & (quantile_returns["quantile"] == 5)
    ].iloc[0]
    worst_bucket = quantile_returns[
        (quantile_returns["factor"] == "low_volatility_20d")
        & (quantile_returns["quantile"] == 1)
    ].iloc[0]

    assert best_bucket["mean_return"] == pytest.approx(0.5)
    assert worst_bucket["mean_return"] == pytest.approx(0.1)


def test_turnover_and_monotonicity_are_reported_without_pipeline_failure():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    factor_matrix = pd.DataFrame(
        {
            "1101": [1.0, 5.0],
            "1102": [2.0, 1.0],
            "1103": [3.0, 2.0],
            "1104": [4.0, 3.0],
            "1105": [5.0, 4.0],
        },
        index=dates,
    )
    assignments = assign_factor_quantiles(
        factor_matrices={"momentum_60d": factor_matrix},
        directions={"momentum_60d": "higher_is_better"},
        quantiles=5,
    )
    forward_returns = pd.DataFrame(
        {
            "date": [dates[0]] * 5 + [dates[1]] * 5,
            "ticker": ["1101", "1102", "1103", "1104", "1105"] * 2,
            "horizon": [1] * 10,
            "forward_return": [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4],
        }
    )

    quantile_returns = compute_quantile_returns(assignments, forward_returns)
    turnover = compute_turnover_summary(assignments)
    monotonicity = evaluate_monotonicity(quantile_returns)

    assert turnover.loc[0, "average_best_bucket_turnover"] == pytest.approx(1.0)
    assert monotonicity.loc[0, "monotonicity_pass"]
    assert monotonicity.loc[0, "adjacent_agreement_ratio"] == pytest.approx(1.0)

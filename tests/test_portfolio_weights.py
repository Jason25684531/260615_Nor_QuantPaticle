import pandas as pd

from twse_factor_lab.portfolio.weights import build_equal_weight_portfolio


def test_equal_weights_sum_to_one_per_execution_date_and_apply_t_plus_one():
    positions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "ticker": ["0001", "0002"],
            "factor_score": [0.9, 0.8],
            "rank": [1, 2],
            "selected": [True, True],
            "top_n": [2, 2],
        }
    )

    weights = build_equal_weight_portfolio(positions, execution_lag_days=1)

    assert weights["target_weight"].sum() == 1.0
    assert weights["execution_lag_days"].unique().tolist() == [1]
    assert (weights["execution_date"] > weights["date"]).all()
    assert weights["execution_date"].unique().tolist() == [pd.Timestamp("2024-01-03")]

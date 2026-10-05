from __future__ import annotations

import json

import pandas as pd


def test_costs_are_nonzero_and_net_return_is_below_gross_return() -> None:
    root = "data/research/fundamental-production-final-v1/historical-evidence/"
    transactions = pd.read_parquet(root + "canonical_backtest_transactions.parquet")
    validation = json.loads(
        open(root + "historical_strategy_validation.json", encoding="utf-8").read()
    )
    assert transactions["total_cost"].sum() > 0
    assert validation["strategy_evidence"]["gross_return"] > validation[
        "strategy_evidence"
    ]["net_return"]

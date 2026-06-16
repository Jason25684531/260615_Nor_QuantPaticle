from twse_factor_lab.backtest.costs import CostModel


def test_cost_model_separates_buy_and_sell_costs():
    model = CostModel(
        buy_fee_rate=0.001425,
        sell_fee_rate=0.001425,
        transaction_tax_rate=0.003,
        slippage_rate=0.001,
    )

    assert model.buy_cost_rate == 0.002425
    assert model.sell_cost_rate == 0.005425
    assert model.summary()["buy_cost_rate"] == 0.002425
    assert model.summary()["sell_cost_rate"] == 0.005425

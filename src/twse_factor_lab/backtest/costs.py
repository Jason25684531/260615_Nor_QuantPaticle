"""Configurable side-aware transaction cost model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    buy_fee_rate: float = 0.001425
    sell_fee_rate: float = 0.001425
    transaction_tax_rate: float = 0.003
    slippage_rate: float = 0.001

    @property
    def buy_cost_rate(self) -> float:
        return self.buy_fee_rate + self.slippage_rate

    @property
    def sell_cost_rate(self) -> float:
        return self.sell_fee_rate + self.transaction_tax_rate + self.slippage_rate

    def summary(self) -> dict[str, float]:
        return {
            "buy_fee_rate": self.buy_fee_rate,
            "sell_fee_rate": self.sell_fee_rate,
            "transaction_tax_rate": self.transaction_tax_rate,
            "slippage_rate": self.slippage_rate,
            "buy_cost_rate": self.buy_cost_rate,
            "sell_cost_rate": self.sell_cost_rate,
        }

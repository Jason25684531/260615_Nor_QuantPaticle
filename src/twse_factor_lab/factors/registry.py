"""Standardized Factor Library: factor definition/metadata registry.

Metadata layer only — links to existing factor implementations and never
recomputes frozen RC1 factor outputs. Alpha evaluation (IC/quantile) belongs
to the Day 3 Factor Gate; composition/weighting belongs to the Day 4 Strategy
Lab.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

FACTOR_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

FACTOR_FAMILIES = frozenset(
    {
        "VALUE",
        "QUALITY",
        "GROWTH",
        "CASH_FLOW",
        "FINANCIAL_HEALTH",
        "EFFICIENCY",
        "MOMENTUM",
        "VOLUME_BEHAVIOR",
        "RISK_VOLATILITY",
        "TECHNICAL",
    }
)
# Reuses the direction vocabulary of factors/ranking.py and config/strategy.yaml.
FACTOR_DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})
DATA_DOMAINS = frozenset({"ohlcv", "fundamental", "valuation_snapshot", "composite"})
FACTOR_STATUSES = frozenset({"active", "deprecated"})


class FactorLibraryError(ValueError):
    pass


def _require_nonempty(value: str, name: str) -> None:
    if not value:
        raise FactorLibraryError(f"{name} must be non-empty")


@dataclass(frozen=True)
class FactorDefinition:
    factor_id: str
    name: str
    family: str
    description: str
    direction: str
    required_inputs: list[str]
    data_domain: str
    pit_required: bool
    implementation: str  # "package.module:callable" of the existing builder
    version: str
    status: str
    lookback_days: int | None = None
    extras: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.factor_id or not FACTOR_ID_PATTERN.match(self.factor_id):
            raise FactorLibraryError(
                "factor_id must be a non-empty filesystem-safe slug"
            )
        _require_nonempty(self.name, "name")
        _require_nonempty(self.description, "description")
        _require_nonempty(self.version, "version")
        if self.family not in FACTOR_FAMILIES:
            raise FactorLibraryError(f"unknown factor family: {self.family!r}")
        if self.direction not in FACTOR_DIRECTIONS:
            raise FactorLibraryError(f"unknown factor direction: {self.direction!r}")
        if self.data_domain not in DATA_DOMAINS:
            raise FactorLibraryError(f"unknown data domain: {self.data_domain!r}")
        if self.status not in FACTOR_STATUSES:
            raise FactorLibraryError(f"unknown factor status: {self.status!r}")
        if not self.required_inputs:
            raise FactorLibraryError("required_inputs must be non-empty")
        if ":" not in self.implementation or not all(
            self.implementation.split(":", 1)
        ):
            raise FactorLibraryError(
                "implementation must reference an existing callable as "
                f"'module:attribute', got {self.implementation!r}"
            )
        if self.lookback_days is not None and self.lookback_days <= 0:
            raise FactorLibraryError("lookback_days must be positive when set")

    def resolve_implementation(self) -> Callable:
        module_name, attribute = self.implementation.split(":", 1)
        module = importlib.import_module(module_name)
        try:
            return getattr(module, attribute)
        except AttributeError as exc:
            raise FactorLibraryError(
                f"implementation not found: {self.implementation!r}"
            ) from exc


class FactorRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, FactorDefinition] = {}

    def register(self, definition: FactorDefinition) -> None:
        definition.validate()
        if definition.factor_id in self._definitions:
            raise FactorLibraryError(
                f"duplicate factor_id: {definition.factor_id!r}"
            )
        self._definitions[definition.factor_id] = definition

    def get(self, factor_id: str) -> FactorDefinition:
        try:
            return self._definitions[factor_id]
        except KeyError as exc:
            raise FactorLibraryError(f"unknown factor_id: {factor_id!r}") from exc

    def list_definitions(self, family: str | None = None) -> list[FactorDefinition]:
        if family is not None and family not in FACTOR_FAMILIES:
            raise FactorLibraryError(f"unknown factor family: {family!r}")
        return [
            self._definitions[factor_id]
            for factor_id in sorted(self._definitions)
            if family is None or self._definitions[factor_id].family == family
        ]

    def to_json(self) -> str:
        payload = [asdict(definition) for definition in self.list_definitions()]
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> FactorRegistry:
        registry = cls()
        for entry in json.loads(text):
            registry.register(FactorDefinition(**entry))
        return registry


_PRICE_VOLUME_IMPL = (
    "twse_factor_lab.factors.price_volume:build_price_volume_factor_frame"
)
_FUNDAMENTAL_IMPL = (
    "twse_factor_lab.factors.fundamental:build_fundamental_factor_frame"
)
_COMPOSER_IMPL = "twse_factor_lab.factors.composer:build_composite_factor_frame"
_OHLCV_INPUTS = ["close_matrix", "high_matrix", "low_matrix", "volume_matrix"]
_FUNDAMENTAL_INPUTS = ["date", "ticker", "metric", "value"]

# (factor_id, name, family, direction, lookback, description)
_PRICE_VOLUME_FACTORS = [
    ("momentum_40d", "Momentum 40D", "MOMENTUM", "higher_is_better", 40,
     "40-day price momentum: close_t / close_(t-40) - 1."),
    ("momentum_60d", "Momentum 60D", "MOMENTUM", "higher_is_better", 60,
     "60-day price momentum: close_t / close_(t-60) - 1."),
    ("risk_adjusted_momentum", "Risk Adjusted Momentum", "MOMENTUM",
     "higher_is_better", 54, "momentum_40d divided by NATR14."),
    ("rsi_28", "RSI 28", "TECHNICAL", "higher_is_better", 28,
     "Relative Strength Index with period 28."),
    ("rsi_z", "RSI-Z", "TECHNICAL", "higher_is_better", 118,
     "Rolling 90-day time-series Z-score of RSI28."),
    ("ma_60", "MA 60", "TECHNICAL", "higher_is_better", 60,
     "60-day simple moving average of close."),
    ("ma_120", "MA 120", "TECHNICAL", "higher_is_better", 120,
     "120-day simple moving average of close."),
    ("obv", "OBV", "VOLUME_BEHAVIOR", "higher_is_better", None,
     "On-balance volume cumulated over full history."),
    ("obv_z90", "OBV-Z90", "VOLUME_BEHAVIOR", "higher_is_better", 90,
     "Rolling 90-day time-series Z-score of OBV."),
    ("volume_ratio_5d_60d", "Volume Ratio 5D/60D", "VOLUME_BEHAVIOR",
     "higher_is_better", 60, "5-day over 60-day mean volume ratio."),
    ("atr_14", "ATR 14", "RISK_VOLATILITY", "lower_is_better", 14,
     "Average true range with period 14."),
    ("natr_14", "NATR 14", "RISK_VOLATILITY", "lower_is_better", 14,
     "Normalized ATR: atr_14 / close."),
    ("low_volatility_20d", "Low Volatility 20D", "RISK_VOLATILITY",
     "lower_is_better", 20, "20-day ATR/close (fallback: 20-day return std)."),
]

_FUNDAMENTAL_FACTORS = [
    ("pe", "PE", "VALUE", "lower_is_better", "Price-to-earnings ratio."),
    ("pb", "PB", "VALUE", "lower_is_better", "Price-to-book ratio."),
    ("dividend_yield", "Dividend Yield", "VALUE", "higher_is_better",
     "Trailing dividend yield."),
    ("roe", "ROE", "QUALITY", "higher_is_better", "Return on equity."),
    ("eps", "EPS", "QUALITY", "higher_is_better", "Earnings per share."),
    ("revenue_yoy", "Revenue YoY", "GROWTH", "higher_is_better",
     "Year-over-year monthly revenue growth."),
]


def build_default_registry() -> FactorRegistry:
    """Register metadata for the factors already implemented in this repo.

    Linkage only: implementation references point at the existing builders;
    nothing here computes factor values or touches frozen RC1 outputs.
    """
    registry = FactorRegistry()
    for factor_id, name, family, direction, lookback, description in (
        _PRICE_VOLUME_FACTORS
    ):
        registry.register(
            FactorDefinition(
                factor_id=factor_id,
                name=name,
                family=family,
                description=description,
                direction=direction,
                required_inputs=list(_OHLCV_INPUTS),
                data_domain="ohlcv",
                pit_required=False,
                implementation=_PRICE_VOLUME_IMPL,
                version="1.0.0",
                status="active",
                lookback_days=lookback,
            )
        )
    for factor_id, name, family, direction, description in _FUNDAMENTAL_FACTORS:
        registry.register(
            FactorDefinition(
                factor_id=factor_id,
                name=name,
                family=family,
                description=description,
                direction=direction,
                required_inputs=list(_FUNDAMENTAL_INPUTS),
                data_domain="fundamental",
                pit_required=True,
                implementation=_FUNDAMENTAL_IMPL,
                version="1.0.0",
                status="active",
            )
        )
    registry.register(
        FactorDefinition(
            factor_id="historical_price_volume",
            name="Historical Price Volume",
            family="TECHNICAL",
            description=(
                "RC1 composite of momentum_60d, low_volatility_20d, and "
                "volume_ratio_5d_60d (frozen canonical pool member)."
            ),
            direction="higher_is_better",
            required_inputs=list(_OHLCV_INPUTS),
            data_domain="composite",
            pit_required=False,
            implementation=_COMPOSER_IMPL,
            version="1.0.0",
            status="active",
            lookback_days=60,
        )
    )
    return registry

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from twse_factor_lab.factors.registry import (
    FactorDefinition,
    FactorLibraryError,
    FactorRegistry,
    build_default_registry,
)

FROZEN_EVIDENCE = [
    "strategy_freeze_manifest.json",
    "reproducibility_manifest.json",
    "d4_acceptance_handoff.json",
    "final_acceptance_handoff.json",
    "final_artifact_inventory.json",
    "research_trial_inventory.parquet",
    "statistical_acceptance.parquet",
    "trade_excursions.parquet",
]


def make_definition(**overrides) -> FactorDefinition:
    base = FactorDefinition(
        factor_id="momentum_test",
        name="Momentum Test",
        family="MOMENTUM",
        description="test factor",
        direction="higher_is_better",
        required_inputs=["close_matrix"],
        data_domain="ohlcv",
        pit_required=False,
        implementation="twse_factor_lab.factors.price_volume:momentum_60d",
        version="1.0.0",
        status="active",
        lookback_days=60,
    )
    return replace(base, **overrides)


def test_register_and_retrieve_by_factor_id():
    registry = FactorRegistry()
    definition = make_definition()
    registry.register(definition)
    assert registry.get("momentum_test") == definition


def test_list_factors_and_filter_by_family():
    registry = FactorRegistry()
    registry.register(make_definition())
    registry.register(
        make_definition(
            factor_id="pe_test", family="VALUE", direction="lower_is_better"
        )
    )
    assert [d.factor_id for d in registry.list_definitions()] == [
        "momentum_test",
        "pe_test",
    ]
    assert [d.factor_id for d in registry.list_definitions(family="VALUE")] == [
        "pe_test"
    ]


def test_filter_by_unknown_family_rejected():
    with pytest.raises(FactorLibraryError, match="unknown factor family"):
        FactorRegistry().list_definitions(family="ASTROLOGY")


def test_duplicate_factor_id_rejected():
    registry = FactorRegistry()
    registry.register(make_definition())
    with pytest.raises(FactorLibraryError, match="duplicate factor_id"):
        registry.register(make_definition(name="Other Display Name"))
    assert len(registry.list_definitions()) == 1


def test_unknown_factor_id_rejected():
    with pytest.raises(FactorLibraryError, match="unknown factor_id"):
        FactorRegistry().get("nope")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"factor_id": ""}, "factor_id"),
        ({"factor_id": "Bad Slug!"}, "factor_id"),
        ({"name": ""}, "name"),
        ({"description": ""}, "description"),
        ({"version": ""}, "version"),
        ({"family": "UNKNOWN"}, "unknown factor family"),
        ({"direction": "sideways"}, "unknown factor direction"),
        ({"data_domain": "tarot"}, "unknown data domain"),
        ({"status": "retired"}, "unknown factor status"),
        ({"required_inputs": []}, "required_inputs"),
        ({"implementation": "no-colon"}, "implementation"),
        ({"implementation": "module:"}, "implementation"),
        ({"lookback_days": 0}, "lookback_days"),
    ],
)
def test_invalid_metadata_fails_fast(overrides, message):
    registry = FactorRegistry()
    with pytest.raises(FactorLibraryError, match=message):
        registry.register(make_definition(**overrides))
    assert registry.list_definitions() == []


def test_pit_required_metadata_preserved():
    registry = FactorRegistry()
    registry.register(
        make_definition(
            factor_id="roe_test",
            family="QUALITY",
            data_domain="fundamental",
            pit_required=True,
        )
    )
    assert registry.get("roe_test").pit_required is True
    reloaded = FactorRegistry.from_json(registry.to_json())
    assert reloaded.get("roe_test").pit_required is True


def test_deterministic_ordering_and_serialization():
    first, second = FactorRegistry(), FactorRegistry()
    a = make_definition(factor_id="alpha_test")
    b = make_definition(factor_id="beta_test")
    first.register(a)
    first.register(b)
    second.register(b)
    second.register(a)
    assert first.list_definitions() == second.list_definitions()
    assert first.to_json() == second.to_json()


def test_serialization_round_trip():
    registry = build_default_registry()
    reloaded = FactorRegistry.from_json(registry.to_json())
    assert reloaded.list_definitions() == registry.list_definitions()


def test_default_registry_directions():
    registry = build_default_registry()
    assert registry.get("pe").direction == "lower_is_better"
    assert registry.get("pb").direction == "lower_is_better"
    assert registry.get("roe").direction == "higher_is_better"
    assert registry.get("momentum_40d").direction == "higher_is_better"
    assert registry.get("momentum_60d").direction == "higher_is_better"


def test_default_registry_pit_and_domains():
    registry = build_default_registry()
    for factor_id in ("pe", "pb", "roe", "eps", "revenue_yoy", "dividend_yield"):
        definition = registry.get(factor_id)
        assert definition.pit_required is True
        assert definition.data_domain == "fundamental"
    assert registry.get("momentum_40d").pit_required is False
    assert registry.get("momentum_40d").data_domain == "ohlcv"


def test_default_registry_implementation_references_resolve():
    for definition in build_default_registry().list_definitions():
        assert callable(definition.resolve_implementation())


def test_unresolvable_implementation_reference():
    definition = make_definition(
        implementation="twse_factor_lab.factors.price_volume:does_not_exist"
    )
    with pytest.raises(FactorLibraryError, match="implementation not found"):
        definition.resolve_implementation()


def test_frozen_rc1_artifacts_untouched():
    root = Path(__file__).resolve().parents[1]
    before = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in FROZEN_EVIDENCE
        if (root / name).exists()
    }
    registry = build_default_registry()
    registry.to_json()
    registry.list_definitions(family="VALUE")
    after = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in before
    }
    assert after == before

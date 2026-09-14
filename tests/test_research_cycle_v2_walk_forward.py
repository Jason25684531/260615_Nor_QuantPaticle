"""Phase 8 tests: walk-forward validation folds and honest labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twse_factor_lab.acceptance.walk_forward import (
    VALIDATION_LABEL,
    run_walk_forward,
)
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.governance import (
    ResearchManifest,
    load_experiment_registry,
    save_research_manifest,
)
from twse_factor_lab.strategy.lab import StrategyDefinition

FOLDS = (("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"))


def _manifest() -> ResearchManifest:
    return ResearchManifest(
        research_id="wf-test",
        hypothesis="walk-forward fixture",
        factor_candidates=["pe"],
        universe="fixture",
        dataset_version="dataset-v1",
        is_start="2021-01-04",
        is_end="2021-12-31",
        oos_start="2022-01-03",
        oos_end="2023-12-29",
        rebalance_search_space=["monthly"],
        top_n_search_space=[1],
        cost_scenarios=["base_cost"],
        selection_relevant=True,
        status="active",
    )


def _definition() -> StrategyDefinition:
    return StrategyDefinition(
        strategy_id="locked",
        research_id="wf-test",
        factor_ids=("pe",),
        factor_weights={"pe": 1.0},
        top_n=1,
        rebalance_frequency="monthly",
        buffer_enabled=False,
        drop_rank_buffer=0,
        cost_scenario="base_cost",
        universe="fixture",
        dataset_version="dataset-v1",
    )


def _data():
    dates = pd.bdate_range("2021-01-04", "2023-12-31")
    rng = np.random.default_rng(7)
    ra = pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates)
    rb = pd.Series(rng.normal(0.0001, 0.01, len(dates)), index=dates)
    close = pd.DataFrame(
        {"A": 100.0 * (1 + ra).cumprod(), "B": 100.0 * (1 + rb).cumprod()}
    )
    pe = pd.DataFrame({"A": [1.0] * len(dates), "B": [2.0] * len(dates)}, index=dates)
    return close, {"pe": pe}


def test_walk_forward_runs_fresh_folds_and_labels_honestly(tmp_path):
    save_research_manifest(_manifest(), tmp_path)
    close, factors = _data()

    outcome = run_walk_forward(
        root=tmp_path,
        research_id="wf-test",
        base_strategy_id="locked",
        definition=_definition(),
        factor_matrices=factors,
        close_matrix=close,
        admission_results={"pe": "ACCEPT"},
        cost_model=CostModel(),
        folds=FOLDS,
    )

    payload = outcome["payload"]
    assert payload["label"] == VALIDATION_LABEL
    assert "FRESH UNTOUCHED OOS" not in payload["reason"].upper()
    assert len(payload["folds"]) == 2
    for fold in payload["folds"]:
        assert fold["fresh_state"] is True
        assert fold["inherited_is_state"] is False
    # Each fold registered its own diagnostic experiment (never selection-relevant).
    records = load_experiment_registry(tmp_path, "wf-test")
    wf = [r for r in records if r.experiment_id.startswith("walk-forward-")]
    assert len(wf) == 2
    assert all(not r.selection_relevant for r in wf)
    assert not outcome["combined_returns"].empty
    assert (tmp_path / payload["artifact_path"]).exists()

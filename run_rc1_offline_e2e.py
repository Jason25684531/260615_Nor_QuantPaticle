"""Read-only deterministic RC1 evidence replay; never overwrites frozen outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent


def main() -> None:
    handoff = json.loads((ROOT / "final_acceptance_handoff.json").read_text())
    statistical = pd.read_parquet(ROOT / "statistical_acceptance.parquet")
    trials = pd.read_parquet(ROOT / "research_trial_inventory.parquet")
    excursions = pd.read_parquet(ROOT / "trade_excursions.parquet")
    required = [
        "data/processed/composite_scores.parquet",
        "data/processed/backtest_results.parquet",
        "data/processed/robustness_scoreboard.parquet",
    ]
    assert all((ROOT / path).exists() for path in required)
    unique_trials = trials.loc[
        trials["included_in_dsr"], "unique_strategy_config_id"
    ].nunique()
    assert handoff["effective_trials"] == unique_trials == 17
    assert len(excursions) == 5973
    assert float(handoff["psr"]) == float(handoff["dsr"])
    assert float(statistical.loc[0, "effective_trials"]) == 17
    assert handoff["strategy_acceptance"] == "REJECTED"
    print(
        json.dumps(
            {
                "status": "PASS_OFFLINE",
                "network": False,
                "stages": [
                    "composite",
                    "performance",
                    "robustness",
                    "final_acceptance",
                ],
            }
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SCHEMA = [
    "trial_id",
    "stage",
    "category",
    "configuration",
    "sample",
    "unique_strategy_config_id",
    "strategy_modified",
    "selection_relevant",
    "validation_only",
    "included_in_dsr",
    "reason",
]


def build_trial_inventory(root: str | Path) -> tuple[pd.DataFrame, int]:
    root = Path(root)
    source = pd.read_parquet(root / "data/processed/robustness_scoreboard.parquet")
    rows = []
    selected_categories = {
        "BASELINE",
        "COST",
        "TOP_N",
        "BUFFER",
        "BREADTH",
        "REBALANCE",
    }
    for i, row in source.iterrows():
        category = str(row.get("category", "UNKNOWN")).upper()
        config = "|".join(
            f"{k}={row.get(k)}"
            for k in (
                "scenario",
                "cost_multiplier",
                "top_n",
                "buffer",
                "breadth_threshold",
                "rebalance_frequency",
            )
        )
        uid = hashlib.sha1(config.encode()).hexdigest()[:12]
        selected = category in selected_categories
        rows.append(
            {
                "trial_id": f"d4-{i:03d}",
                "stage": "D4",
                "category": category,
                "configuration": config,
                "sample": row.get("sample", "UNKNOWN"),
                "unique_strategy_config_id": uid,
                "strategy_modified": category != "BASELINE",
                "selection_relevant": selected,
                "validation_only": not selected,
                "included_in_dsr": selected,
                "reason": "selection-relevant frozen configuration"
                if selected
                else "diagnostic/validation-only",
            }
        )
    frame = pd.DataFrame(rows, columns=SCHEMA)
    effective = int(
        frame.loc[frame.included_in_dsr, "unique_strategy_config_id"].nunique()
    )
    frame.to_parquet(root / "research_trial_inventory.parquet", index=False)
    return frame, effective

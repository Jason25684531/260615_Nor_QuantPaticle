from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = [
    "trade_id",
    "ticker",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "realized_return",
    "MAE",
    "MFE",
    "holding_period",
    "entry_weight",
    "exit_reason",
]


def build_excursions(root: str | Path, start: str, end: str) -> pd.DataFrame:
    root = Path(root)
    trades = pd.read_parquet(root / "data/processed/vectorbt_trades.parquet")
    trades["Entry Timestamp"] = pd.to_datetime(trades["Entry Timestamp"])
    trades["Exit Timestamp"] = pd.to_datetime(trades["Exit Timestamp"])
    trades = trades[
        (trades["Entry Timestamp"] >= start) & (trades["Entry Timestamp"] <= end)
    ].copy()
    close = pd.read_parquet(root / "data/processed/close_matrix.parquet")
    close.index = pd.to_datetime(close.index)
    rows = []
    for i, t in trades.iterrows():
        ticker = str(t["Column"])
        path = (
            close.loc[t["Entry Timestamp"] : t["Exit Timestamp"], ticker]
            if ticker in close
            else pd.Series(dtype=float)
        )
        entry = float(t["Avg Entry Price"])
        exit_price = float(t["Avg Exit Price"])
        rel = (
            path / entry - 1 if len(path) else pd.Series([float(t.get("Return", 0.0))])
        )
        rows.append(
            {
                "trade_id": int(t.get("Exit Trade Id", i)),
                "ticker": ticker,
                "entry_date": t["Entry Timestamp"],
                "exit_date": t["Exit Timestamp"],
                "entry_price": entry,
                "exit_price": exit_price,
                "realized_return": float(t.get("Return", exit_price / entry - 1)),
                "MAE": float(rel.min()),
                "MFE": float(rel.max()),
                "holding_period": int(
                    max(0, (t["Exit Timestamp"] - t["Entry Timestamp"]).days)
                ),
                "entry_weight": np.nan,
                "exit_reason": "end_of_sample"
                if t["Exit Timestamp"] >= pd.Timestamp(end)
                else "other_target_zero",
            }
        )
    frame = pd.DataFrame(rows, columns=SCHEMA)
    frame.to_parquet(root / "trade_excursions.parquet", index=False)
    return frame

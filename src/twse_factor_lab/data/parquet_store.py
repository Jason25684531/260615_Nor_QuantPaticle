"""Parquet storage helpers for processed research data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetStore:
    """Save and load processed DataFrames through a single entry point."""

    def save(
        self,
        frame: pd.DataFrame,
        path: str | Path,
        *,
        include_index: bool = False,
    ) -> Path:
        if frame.empty:
            raise ValueError("Cannot save empty DataFrame to Parquet")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=include_index)
        return output_path

    def load(self, path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(Path(path))

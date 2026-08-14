"""Parquet storage helpers for processed research data."""

from __future__ import annotations

import os
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
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            frame.to_parquet(temporary_path, index=include_index)
            read_back = pd.read_parquet(temporary_path)
            if read_back.empty:
                raise ValueError("Parquet read-back was empty")
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return output_path

    def load(self, path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(Path(path))

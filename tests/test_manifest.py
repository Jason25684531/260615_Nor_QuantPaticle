from datetime import UTC, datetime

import pandas as pd

from twse_factor_lab.data.manifest import build_manifest_entry


def test_build_manifest_entry_records_required_metadata():
    frame = pd.DataFrame({"1101": [1.0, 2.0], "1102": [3.0, 4.0]})
    frame.index = pd.to_datetime(["2024-01-02", "2024-01-03"])

    entry = build_manifest_entry(
        artifact_name="close_matrix",
        path="data/processed/close_matrix.parquet",
        frame=frame,
        source_inputs=["data/processed/ohlcv.parquet"],
        schema_version="1.0.0",
        created_at=datetime(2026, 6, 16, tzinfo=UTC),
        notes="wide close matrix",
    )

    assert entry["artifact_name"] == "close_matrix"
    assert entry["rows"] == 2
    assert entry["columns"] == 2
    assert entry["source_inputs"] == ["data/processed/ohlcv.parquet"]
    assert entry["schema_version"] == "1.0.0"
    assert entry["notes"] == "wide close matrix"

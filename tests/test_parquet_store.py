import pandas as pd
import pytest

from twse_factor_lab.data.parquet_store import ParquetStore


def test_parquet_store_creates_parent_directory_and_round_trips_dataframe(tmp_path):
    store = ParquetStore()
    path = tmp_path / "nested" / "data.parquet"
    frame = pd.DataFrame({"ticker": ["2330"], "close": [1.5]})

    store.save(frame, path)
    loaded = store.load(path)

    assert path.exists()
    pd.testing.assert_frame_equal(loaded, frame)


def test_parquet_store_rejects_empty_dataframe(tmp_path):
    store = ParquetStore()

    with pytest.raises(ValueError, match="empty DataFrame"):
        store.save(pd.DataFrame(), tmp_path / "empty.parquet")

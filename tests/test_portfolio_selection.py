import pandas as pd

from twse_factor_lab.portfolio.selection import build_topn_positions


def test_topn_selection_uses_only_historical_price_volume_rows():
    date = pd.Timestamp("2024-01-02")
    factors = pd.DataFrame(
        {
            "date": [date, date, date, date],
            "ticker": ["0001", "0002", "0003", "9999"],
            "composite_score": [0.2, 0.9, 0.5, 1.0],
            "composite_type": [
                "historical_price_volume",
                "historical_price_volume",
                "historical_price_volume",
                "latest_snapshot_mixed",
            ],
            "is_snapshot_component_used": [False, False, False, True],
        }
    )

    positions = build_topn_positions(factors, top_n=2)

    selected = positions[positions["selected"]]
    assert selected["ticker"].tolist() == ["0002", "0003"]
    assert "9999" not in positions["ticker"].tolist()
    assert positions["top_n"].unique().tolist() == [2]


def test_topn_selection_selects_available_count_when_less_than_n():
    factors = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "ticker": ["0001"],
            "composite_score": [0.2],
            "composite_type": ["historical_price_volume"],
            "is_snapshot_component_used": [False],
        }
    )

    positions = build_topn_positions(factors, top_n=20)

    assert positions["selected"].tolist() == [True]
    assert positions["rank"].tolist() == [1]

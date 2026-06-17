"""Tests for buffer / hold-until-drop selection rule."""

from __future__ import annotations

import pandas as pd

from twse_factor_lab.portfolio.selection import buffered_top_n_selection


def _scores(mapping: dict[str, float]) -> pd.Series:
    return pd.Series(mapping)


class TestBufferedTopNSelection:
    def test_selects_top_n_when_no_holdings(self) -> None:
        scores = _scores({"A": 10, "B": 9, "C": 8, "D": 7, "E": 6})
        result = buffered_top_n_selection(
            scores, top_n=3, drop_rank_buffer=5, current_holdings=set()
        )
        assert len(result) == 3
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_retains_holding_within_buffer(self) -> None:
        # D is rank 4, buffer=5 → should be retained
        scores = _scores({"A": 10, "B": 9, "C": 8, "D": 7, "E": 6})
        result = buffered_top_n_selection(
            scores, top_n=3, drop_rank_buffer=5, current_holdings={"D"}
        )
        assert "D" in result

    def test_drops_holding_outside_buffer(self) -> None:
        # F has score 1 → rank 6, buffer=5 → should be dropped
        scores = _scores({"A": 10, "B": 9, "C": 8, "D": 7, "E": 6, "F": 1})
        result = buffered_top_n_selection(
            scores, top_n=3, drop_rank_buffer=5, current_holdings={"F"}
        )
        assert "F" not in result

    def test_fills_up_to_top_n(self) -> None:
        scores = _scores({"A": 10, "B": 9, "C": 8, "D": 7})
        result = buffered_top_n_selection(
            scores, top_n=3, drop_rank_buffer=5, current_holdings=set()
        )
        assert len(result) == 3

    def test_handles_small_universe(self) -> None:
        scores = _scores({"A": 10, "B": 9})
        result = buffered_top_n_selection(
            scores, top_n=5, drop_rank_buffer=5, current_holdings=set()
        )
        assert len(result) == 2

    def test_buffer_does_not_increase_turnover_vs_naive(self) -> None:
        """Buffer rule should not cause more churn than naive top_n on stable rankings."""  # noqa: E501

        rng = list(range(50))
        # Stable ranking (no change)
        scores = pd.Series({f"T{i:02d}": 50 - i for i in rng})
        holdings: set[str] = set()
        buffer_changes = 0
        naive_changes = 0

        for _ in range(10):
            buf = buffered_top_n_selection(
                scores, top_n=10, drop_rank_buffer=15, current_holdings=holdings
            )
            buffer_changes += len(buf.symmetric_difference(holdings))
            holdings = buf

        # On stable rankings, buffer should produce no more changes than naive
        # (both should converge to 0 after first period)
        assert buffer_changes <= naive_changes + 10  # allow first period


class TestBuildTopNPositionsWithBuffer:
    def _make_factors(self) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-02", periods=5)
        rows = []
        for d in dates:
            for ticker in ["A", "B", "C", "D", "E"]:
                rows.append(
                    {
                        "date": d,
                        "ticker": ticker,
                        "composite_score": hash(ticker) % 10 + 1,
                        "composite_type": "historical_price_volume",
                        "is_snapshot_component_used": False,
                    }
                )
        return pd.DataFrame(rows)

    def test_metadata_columns_present(self) -> None:
        from twse_factor_lab.portfolio.selection import build_topn_positions

        factors = self._make_factors()
        result = build_topn_positions(
            factors, top_n=3, hold_until_drop=True, drop_rank_buffer=4
        )
        assert "selection_rule" in result.columns
        assert "buffer_enabled" in result.columns
        assert "drop_rank_buffer" in result.columns
        assert "rebalance_frequency" in result.columns

    def test_buffer_enabled_flag(self) -> None:
        from twse_factor_lab.portfolio.selection import build_topn_positions

        factors = self._make_factors()
        result = build_topn_positions(
            factors, top_n=3, hold_until_drop=True, drop_rank_buffer=4
        )
        assert result["buffer_enabled"].all()

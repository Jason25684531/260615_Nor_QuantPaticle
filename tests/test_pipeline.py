import pandas as pd

from run_data_pipeline import build_quality_report, load_config, resolve_path


def test_load_config_and_resolve_path_reads_strategy_yaml(tmp_path):
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "paths:\n  universe: data/processed/universe.parquet\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert resolve_path(config_path, config["paths"]["universe"]) == (
        tmp_path / "data" / "processed" / "universe.parquet"
    )


def test_build_quality_report_includes_counts_missing_ratios_and_failures():
    universe = pd.DataFrame({"ticker": ["2330", "2317"], "industry": ["A", None]})
    valuation = pd.DataFrame({"ticker": ["2330"], "pe": [20.0]})
    ohlcv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["2330", "2330"],
            "close": [1.5, None],
        }
    )

    report = build_quality_report(
        universe=universe,
        valuation=valuation,
        ohlcv=ohlcv,
        failed_tickers=["9999"],
    )

    assert "Universe rows: 2" in report
    assert "Valuation rows: 1" in report
    assert "OHLCV rows: 2" in report
    assert "Ticker count: 2" in report
    assert "2024-01-02" in report
    assert "9999" in report
    assert "universe" in report

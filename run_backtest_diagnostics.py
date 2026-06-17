"""Backtest diagnostics pipeline: realism checks, sensitivity analysis, turnover."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from run_data_pipeline import resolve_path
from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.diagnostics import (
    build_backtest_realism_report,
    compute_turnover_diagnostics,
    run_engine_comparison,
)
from twse_factor_lab.backtest.scenarios import (
    run_cost_scenarios,
    run_rebalance_sensitivity,
    run_topn_sensitivity,
)
from twse_factor_lab.data.manifest import append_manifest_entries, build_manifest_entry
from twse_factor_lab.data.parquet_store import ParquetStore
from twse_factor_lab.portfolio.rebalance import (
    build_rebalance_calendar,
    save_rebalance_calendar,
    validate_factor_eligibility,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _path_map(config_path: str | Path, paths: dict[str, str]) -> dict[str, Path]:
    defaults: dict[str, str] = {
        "close_matrix": "data/processed/close_matrix.parquet",
        "factors_composite": "data/processed/factors_composite.parquet",
        "portfolio_weights": "data/processed/portfolio_weights.parquet",
        "backtest_metrics": "data/processed/backtest_metrics.parquet",
        "rebalance_calendar": "data/processed/rebalance_calendar.parquet",
        "backtest_scenarios": "data/processed/backtest_scenarios.parquet",
        "topn_sensitivity": "data/processed/topn_sensitivity.parquet",
        "rebalance_sensitivity": "data/processed/rebalance_sensitivity.parquet",
        "backtest_turnover_diagnostics": (
            "data/processed/backtest_turnover_diagnostics.parquet"
        ),
        "backtest_engine_comparison": (
            "data/processed/backtest_engine_comparison.parquet"
        ),
        "backtest_realism_report": "reports/backtest_realism_report.md",
        "manifest": "data/processed/_manifest.json",
    }
    merged = dict(defaults | paths)
    return {name: resolve_path(config_path, value) for name, value in merged.items()}


def run_diagnostics(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    paths = _path_map(config_path, config.get("paths", {}))
    backtest_cfg = config.get("backtest", {}) or {}
    store = ParquetStore()

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("Loading close_matrix and factors_composite ...")
    close_matrix = store.load(paths["close_matrix"])
    factors_composite = store.load(paths["factors_composite"])

    # ── Step 2: Validate factor eligibility ───────────────────────────────
    factor_name = str(backtest_cfg.get("factor_name", "historical_price_volume"))
    logger.info("Validating factor eligibility: %s", factor_name)
    validate_factor_eligibility(factor_name)

    # ── Step 3: Build baseline rebalance calendar ─────────────────────────
    frequency = str(backtest_cfg.get("rebalance_frequency", "daily"))
    execution_lag_days = int(backtest_cfg.get("execution_lag_days", 1))
    logger.info("Building rebalance calendar (frequency=%s) ...", frequency)
    calendar = build_rebalance_calendar(
        pd.DatetimeIndex(close_matrix.index),
        frequency=frequency,
        execution_lag_days=execution_lag_days,
    )
    save_rebalance_calendar(calendar, paths["rebalance_calendar"])
    logger.info(
        "rebalance_calendar → %s (%d rows)", paths["rebalance_calendar"], len(calendar)
    )

    # ── Step 4: Load baseline portfolio_weights for diagnostics ───────────
    try:
        portfolio_weights = store.load(paths["portfolio_weights"])
    except Exception:
        logger.warning("portfolio_weights not found, skipping weight-based diagnostics")
        portfolio_weights = pd.DataFrame()

    # ── Step 5: Cost sensitivity ──────────────────────────────────────────
    logger.info("Running cost sensitivity scenarios ...")
    scenarios_df = run_cost_scenarios(close_matrix, factors_composite, config)
    scenarios_df.to_parquet(paths["backtest_scenarios"], index=False)
    logger.info(
        "backtest_scenarios → %s (%d rows)",
        paths["backtest_scenarios"],
        len(scenarios_df),
    )

    # ── Step 6: Top N sensitivity ─────────────────────────────────────────
    logger.info("Running Top N sensitivity ...")
    topn_df = run_topn_sensitivity(close_matrix, factors_composite, config)
    topn_df.to_parquet(paths["topn_sensitivity"], index=False)
    logger.info(
        "topn_sensitivity → %s (%d rows)", paths["topn_sensitivity"], len(topn_df)
    )

    # ── Step 7: Rebalance frequency sensitivity ───────────────────────────
    logger.info("Running rebalance frequency sensitivity ...")
    rebalance_df = run_rebalance_sensitivity(close_matrix, factors_composite, config)
    rebalance_df.to_parquet(paths["rebalance_sensitivity"], index=False)
    logger.info(
        "rebalance_sensitivity → %s (%d rows)",
        paths["rebalance_sensitivity"],
        len(rebalance_df),
    )

    # ── Step 8: Turnover diagnostics ──────────────────────────────────────
    logger.info("Computing turnover diagnostics ...")
    fees = backtest_cfg.get("fees", {}) or {}
    cost_model = CostModel(
        buy_fee_rate=float(fees.get("buy_fee_rate", 0.001425)),
        sell_fee_rate=float(fees.get("sell_fee_rate", 0.001425)),
        transaction_tax_rate=float(fees.get("transaction_tax_rate", 0.003)),
        slippage_rate=float(fees.get("slippage_rate", 0.001)),
    )
    diagnostics_df = compute_turnover_diagnostics(
        portfolio_weights, cost_model=cost_model
    )
    diagnostics_df.to_parquet(paths["backtest_turnover_diagnostics"], index=False)
    logger.info(
        "backtest_turnover_diagnostics → %s", paths["backtest_turnover_diagnostics"]
    )

    # ── Step 9: Engine cross-check ────────────────────────────────────────
    logger.info("Running engine cross-check ...")
    engine_df = (
        run_engine_comparison(close_matrix, portfolio_weights, config)
        if not portfolio_weights.empty
        else pd.DataFrame(
            [
                {
                    "engine": "fallback_weight_engine",
                    "total_return": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "turnover": float("nan"),
                    "notes": "portfolio_weights unavailable",
                }
            ]
        )
    )
    engine_df.to_parquet(paths["backtest_engine_comparison"], index=False)
    logger.info("backtest_engine_comparison → %s", paths["backtest_engine_comparison"])

    # ── Step 10: Write realism report ─────────────────────────────────────
    logger.info("Building backtest_realism_report.md ...")
    report = build_backtest_realism_report(
        config_path=config_path,
        config=config,
        close_matrix=close_matrix,
        scenarios_df=scenarios_df,
        topn_df=topn_df,
        rebalance_df=rebalance_df,
        diagnostics_df=diagnostics_df,
        engine_df=engine_df,
        output_paths=paths,
    )
    paths["backtest_realism_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["backtest_realism_report"].write_text(report, encoding="utf-8")
    logger.info("backtest_realism_report → %s", paths["backtest_realism_report"])

    # ── Step 11: Update manifest ──────────────────────────────────────────
    logger.info("Updating _manifest.json ...")
    created_at = datetime.now(UTC)
    schema_version = str(
        config.get("week2", {}).get("manifest_schema_version", "1.0.0")
    )
    artifact_frames: dict[str, pd.DataFrame] = {
        "rebalance_calendar": calendar,
        "backtest_scenarios": scenarios_df,
        "topn_sensitivity": topn_df,
        "rebalance_sensitivity": rebalance_df,
        "backtest_turnover_diagnostics": diagnostics_df,
        "backtest_engine_comparison": engine_df,
    }
    entries = [
        build_manifest_entry(
            artifact_name=name,
            path=str(paths[name]),
            frame=frame,
            source_inputs=[str(paths["close_matrix"]), str(paths["factors_composite"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="backtest_diagnostics artifact",
        )
        for name, frame in artifact_frames.items()
    ]
    append_manifest_entries(entries, paths["manifest"])
    logger.info("manifest updated")

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtest diagnostics pipeline.")
    parser.add_argument("--config", required=True, help="Path to strategy YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_diagnostics(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

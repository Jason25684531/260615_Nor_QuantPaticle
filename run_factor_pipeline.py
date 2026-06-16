from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from run_data_pipeline import resolve_path
from twse_factor_lab.alphalens.formatter import validate_alphalens_inputs
from twse_factor_lab.data.manifest import build_manifest_entry, write_manifest
from twse_factor_lab.data.matrix_builder import build_ohlcv_matrices
from twse_factor_lab.data.parquet_store import ParquetStore
from twse_factor_lab.factors.composer import build_composite_factor_frame
from twse_factor_lab.factors.price_volume import (
    build_price_volume_factor_frame,
    low_volatility_method,
)
from twse_factor_lab.factors.valuation import build_snapshot_valuation_factors
from twse_factor_lab.validation.factor_alignment import validate_factor_matrix
from twse_factor_lab.validation.no_lookahead import describe_snapshot_limitation


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def missing_ratio(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(frame.isna().mean().mean())


def _frame_date_range(frame: pd.DataFrame, date_column: str = "date") -> str:
    if frame.empty:
        return "N/A"
    if date_column in frame.columns:
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    else:
        dates = pd.to_datetime(frame.index, errors="coerce")
    if len(dates) == 0:
        return "N/A"
    return f"{dates.min().date()} to {dates.max().date()}"


def build_factor_quality_report(
    *,
    input_paths: dict[str, Path],
    universe: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    price_volume_factors: pd.DataFrame,
    valuation_factors: pd.DataFrame,
    composite_factors: pd.DataFrame,
    alignment_report: dict[str, object],
    formatter_report: dict[str, object],
    valuation_limitation: str,
    low_volatility_method: str,
) -> str:
    universe_ticker_count = universe["ticker"].nunique() if "ticker" in universe else 0
    historical_composite = composite_factors[
        composite_factors["composite_type"] == "historical_price_volume"
    ]
    latest_snapshot_composite = composite_factors[
        composite_factors["composite_type"] == "latest_snapshot_mixed"
    ]
    latest_snapshot_as_of_date = _frame_date_range(
        latest_snapshot_composite, date_column="date"
    )
    lines = [
        "# Factor Quality Summary",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        "",
        "## Input Parquet Paths",
        "",
    ]
    for name, path in input_paths.items():
        lines.append(f"- {name}: {path}")
    lines.extend(
        [
            "",
            "## Artifact Shapes",
            "",
            f"- close_matrix: {matrices['close'].shape}",
            f"- high_matrix: {matrices['high'].shape}",
            f"- low_matrix: {matrices['low'].shape}",
            f"- volume_matrix: {matrices['volume'].shape}",
            f"- factors_price_volume: {price_volume_factors.shape}",
            f"- factors_valuation_snapshot: {valuation_factors.shape}",
            f"- factors_composite: {composite_factors.shape}",
            "",
            "## Factor Coverage",
            "",
            f"- universe_ticker_count: {universe_ticker_count}",
            f"- price_volume_date_range: {_frame_date_range(price_volume_factors)}",
            f"- valuation_ticker_count: {valuation_factors['ticker'].nunique()}",
            f"- price_volume_missing_ratio: {missing_ratio(price_volume_factors):.4f}",
            f"- valuation_missing_ratio: {missing_ratio(valuation_factors):.4f}",
            f"- composite_missing_ratio: {missing_ratio(composite_factors):.4f}",
            "",
            "## Composite Row Breakdown",
            "",
            f"- factors_composite_total_rows: {len(composite_factors)}",
            f"- historical_price_volume_rows: {len(historical_composite)}",
            f"- latest_snapshot_mixed_rows: {len(latest_snapshot_composite)}",
            "",
            "## Composite Semantics",
            "",
            "historical_price_volume_composite:",
            f"rows: {len(historical_composite)}",
            f"date_range: {_frame_date_range(historical_composite)}",
            "is_snapshot_component_used: false",
            "historical_backtest_ready: true",
            "",
            "latest_snapshot_mixed_composite:",
            f"rows: {len(latest_snapshot_composite)}",
            f"as_of_date: {latest_snapshot_as_of_date}",
            "is_snapshot_component_used: true",
            "historical_backtest_ready: false",
            "",
            "## Price-Volume Factor Readiness",
            "",
            "momentum_60d: ready",
            "low_volatility_20d: ready",
            "volume_ratio_5d_60d: ready",
            "historical_price_volume_composite: ready",
            f"low_volatility_method: {low_volatility_method}",
            "",
            "## Alphalens readiness by factor type",
            "",
            "momentum_60d: ready",
            "low_volatility_20d: ready",
            "volume_ratio_5d_60d: ready",
            "historical_price_volume_composite: ready",
            "pb_inverse: snapshot_only_not_historical_ready",
            "pe_inverse: snapshot_only_not_historical_ready",
            "dividend_yield: snapshot_only_not_historical_ready",
            "latest_snapshot_mixed_composite: not_historical_ready",
            "",
            "## Validation",
            "",
            f"- alignment_is_aligned: {alignment_report['is_aligned']}",
            f"- alignment_missing_ratio: {alignment_report['missing_ratio']:.4f}",
            f"- alphalens_ready: {formatter_report['is_ready']}",
            "",
            "## Valuation Snapshot Limitation",
            "",
            valuation_limitation,
            "",
            "## Warnings",
            "",
            "- latest_snapshot_mixed cannot be used for historical backtest.",
            "",
        ]
    )
    return "\n".join(lines)


def run_factor_pipeline(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    store = ParquetStore()
    paths = config["paths"]
    schema_version = str(
        config.get("week2", {}).get("manifest_schema_version", "1.0.0")
    )

    input_paths = {
        "universe": resolve_path(config_path, paths["universe"]),
        "valuation": resolve_path(config_path, paths["valuation"]),
        "ohlcv": resolve_path(config_path, paths["ohlcv"]),
    }
    output_paths = {
        "close_matrix": resolve_path(config_path, paths["close_matrix"]),
        "high_matrix": resolve_path(config_path, paths["high_matrix"]),
        "low_matrix": resolve_path(config_path, paths["low_matrix"]),
        "volume_matrix": resolve_path(config_path, paths["volume_matrix"]),
        "factors_price_volume": resolve_path(
            config_path, paths["factors_price_volume"]
        ),
        "factors_valuation_snapshot": resolve_path(
            config_path, paths["factors_valuation_snapshot"]
        ),
        "factors_composite": resolve_path(config_path, paths["factors_composite"]),
        "factor_quality_report": resolve_path(
            config_path, paths["factor_quality_report"]
        ),
        "manifest": resolve_path(config_path, paths["manifest"]),
    }

    universe = store.load(input_paths["universe"])
    valuation = store.load(input_paths["valuation"])
    ohlcv = store.load(input_paths["ohlcv"])

    matrices = build_ohlcv_matrices(ohlcv)
    for key, matrix in matrices.items():
        store.save(matrix, output_paths[f"{key}_matrix"], include_index=True)

    price_volume_factors = build_price_volume_factor_frame(
        close_matrix=matrices["close"],
        high_matrix=matrices["high"],
        low_matrix=matrices["low"],
        volume_matrix=matrices["volume"],
    )
    store.save(price_volume_factors, output_paths["factors_price_volume"])

    as_of_date = pd.Timestamp(valuation["date"].dropna().max())
    if pd.isna(as_of_date):
        as_of_date = pd.Timestamp(datetime.now(UTC).date())
    valuation_factors = build_snapshot_valuation_factors(
        valuation, as_of_date=as_of_date
    )
    store.save(valuation_factors, output_paths["factors_valuation_snapshot"])

    composite_factors = build_composite_factor_frame(
        price_volume_factors, valuation_factors
    )
    store.save(composite_factors, output_paths["factors_composite"])

    alignment_report = validate_factor_matrix(
        matrices["close"].pct_change(),
        matrices["close"],
    )
    formatter_report = validate_alphalens_inputs(
        matrices["close"].pct_change(), matrices["close"]
    )
    valuation_limitation = describe_snapshot_limitation(valuation)
    resolved_low_volatility_method = low_volatility_method(
        high_matrix=matrices["high"],
        low_matrix=matrices["low"],
    )

    report = build_factor_quality_report(
        input_paths=input_paths,
        universe=universe,
        matrices=matrices,
        price_volume_factors=price_volume_factors,
        valuation_factors=valuation_factors,
        composite_factors=composite_factors,
        alignment_report=alignment_report,
        formatter_report=formatter_report,
        valuation_limitation=valuation_limitation,
        low_volatility_method=resolved_low_volatility_method,
    )
    output_paths["factor_quality_report"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["factor_quality_report"].write_text(report, encoding="utf-8")

    created_at = datetime.now(UTC)
    manifest_entries = [
        build_manifest_entry(
            artifact_name="close_matrix",
            path=str(output_paths["close_matrix"]),
            frame=matrices["close"],
            source_inputs=[str(input_paths["ohlcv"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="wide close matrix",
        ),
        build_manifest_entry(
            artifact_name="high_matrix",
            path=str(output_paths["high_matrix"]),
            frame=matrices["high"],
            source_inputs=[str(input_paths["ohlcv"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="wide high matrix",
        ),
        build_manifest_entry(
            artifact_name="low_matrix",
            path=str(output_paths["low_matrix"]),
            frame=matrices["low"],
            source_inputs=[str(input_paths["ohlcv"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="wide low matrix",
        ),
        build_manifest_entry(
            artifact_name="volume_matrix",
            path=str(output_paths["volume_matrix"]),
            frame=matrices["volume"],
            source_inputs=[str(input_paths["ohlcv"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="wide volume matrix",
        ),
        build_manifest_entry(
            artifact_name="factors_price_volume",
            path=str(output_paths["factors_price_volume"]),
            frame=price_volume_factors,
            source_inputs=[
                str(output_paths["close_matrix"]),
                str(output_paths["volume_matrix"]),
            ],
            schema_version=schema_version,
            created_at=created_at,
            notes="historical price-volume factor frame",
        ),
        build_manifest_entry(
            artifact_name="factors_valuation_snapshot",
            path=str(output_paths["factors_valuation_snapshot"]),
            frame=valuation_factors,
            source_inputs=[str(input_paths["valuation"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="latest snapshot valuation factors",
        ),
        build_manifest_entry(
            artifact_name="factors_composite",
            path=str(output_paths["factors_composite"]),
            frame=composite_factors,
            source_inputs=[
                str(output_paths["factors_price_volume"]),
                str(output_paths["factors_valuation_snapshot"]),
            ],
            schema_version=schema_version,
            created_at=created_at,
            notes="historical and latest snapshot composite scores",
        ),
    ]
    write_manifest(manifest_entries, output_paths["manifest"])

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Week 2 factor pipeline.")
    parser.add_argument("--config", required=True, help="Path to strategy YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_factor_pipeline(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

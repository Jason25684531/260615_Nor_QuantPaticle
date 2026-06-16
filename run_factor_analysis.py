from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from run_data_pipeline import resolve_path
from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.analysis.information_coefficient import (
    compute_information_coefficients,
    summarize_information_coefficients,
)
from twse_factor_lab.analysis.monotonicity import evaluate_monotonicity
from twse_factor_lab.analysis.preparation import (
    FACTOR_DIRECTIONS,
    select_historical_factor_matrices,
)
from twse_factor_lab.analysis.quantile_returns import (
    assign_factor_quantiles,
    compute_quantile_returns,
)
from twse_factor_lab.analysis.report import build_factor_analysis_report
from twse_factor_lab.analysis.turnover import compute_turnover_summary
from twse_factor_lab.data.manifest import (
    append_manifest_entries,
    build_manifest_entry,
)
from twse_factor_lab.data.parquet_store import ParquetStore


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_factor_analysis(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    store = ParquetStore()
    paths = config["paths"]
    analysis_config = config.get("analysis", {})
    factor_config = analysis_config.get("factors", {})
    horizons = [int(value) for value in analysis_config.get("horizons", [1, 5, 10, 20])]
    quantiles = int(analysis_config.get("quantiles", 5))
    historical_factors = list(
        factor_config.get(
            "historical",
            ["momentum_60d", "low_volatility_20d", "volume_ratio_5d_60d"],
        )
    )
    optional_composites = list(
        factor_config.get("optional_composites", ["historical_price_volume"])
    )
    excluded_snapshot = list(
        factor_config.get(
            "excluded_snapshot",
            ["pb_inverse", "pe_inverse", "dividend_yield", "latest_snapshot_mixed"],
        )
    )
    schema_version = str(
        config.get("week2", {}).get("manifest_schema_version", "1.0.0")
    )

    input_paths = {
        "close_matrix": resolve_path(config_path, paths["close_matrix"]),
        "factors_price_volume": resolve_path(
            config_path, paths["factors_price_volume"]
        ),
        "factors_composite": resolve_path(config_path, paths["factors_composite"]),
        "manifest": resolve_path(config_path, paths["manifest"]),
    }
    output_paths = {
        "factor_forward_returns": resolve_path(
            config_path, paths["factor_forward_returns"]
        ),
        "factor_ic_summary": resolve_path(config_path, paths["factor_ic_summary"]),
        "factor_quantile_returns": resolve_path(
            config_path, paths["factor_quantile_returns"]
        ),
        "factor_turnover": resolve_path(config_path, paths["factor_turnover"]),
        "factor_monotonicity": resolve_path(config_path, paths["factor_monotonicity"]),
        "factor_analysis_report": resolve_path(
            config_path, paths["factor_analysis_report"]
        ),
        "manifest": input_paths["manifest"],
    }

    close_matrix = store.load(input_paths["close_matrix"])
    price_volume_factors = store.load(input_paths["factors_price_volume"])
    composite_factors = store.load(input_paths["factors_composite"])

    factor_matrices, excluded_map = select_historical_factor_matrices(
        price_volume_factors=price_volume_factors,
        composite_factors=composite_factors,
        historical_factors=historical_factors,
        optional_composites=optional_composites,
        excluded_snapshot=excluded_snapshot,
    )
    directions = {
        factor_name: FACTOR_DIRECTIONS[factor_name] for factor_name in factor_matrices
    }

    forward_returns = build_forward_returns(close_matrix, horizons=horizons)
    ic_results = compute_information_coefficients(
        factor_matrices=factor_matrices,
        forward_returns=forward_returns,
    )
    ic_summary = summarize_information_coefficients(ic_results)
    assignments = assign_factor_quantiles(
        factor_matrices=factor_matrices,
        directions=directions,
        quantiles=quantiles,
    )
    quantile_returns = compute_quantile_returns(assignments, forward_returns)
    turnover_summary = compute_turnover_summary(assignments)
    monotonicity = evaluate_monotonicity(quantile_returns)

    store.save(forward_returns, output_paths["factor_forward_returns"])
    store.save(ic_summary, output_paths["factor_ic_summary"])
    store.save(quantile_returns, output_paths["factor_quantile_returns"])
    store.save(turnover_summary, output_paths["factor_turnover"])
    store.save(monotonicity, output_paths["factor_monotonicity"])

    report = build_factor_analysis_report(
        config_path=config_path,
        input_paths=input_paths,
        output_paths=output_paths,
        factors_analyzed=list(factor_matrices),
        excluded_snapshot=excluded_map,
        horizons=horizons,
        quantiles=quantiles,
        close_matrix=close_matrix,
        ic_summary=ic_summary,
        quantile_returns=quantile_returns,
        turnover_summary=turnover_summary,
        monotonicity=monotonicity,
    )
    output_paths["factor_analysis_report"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["factor_analysis_report"].write_text(report, encoding="utf-8")

    created_at = datetime.now(UTC)
    manifest_entries = [
        build_manifest_entry(
            artifact_name="factor_forward_returns",
            path=str(output_paths["factor_forward_returns"]),
            frame=forward_returns,
            source_inputs=[str(input_paths["close_matrix"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="forward returns for historical factor analysis",
        ),
        build_manifest_entry(
            artifact_name="factor_ic_summary",
            path=str(output_paths["factor_ic_summary"]),
            frame=ic_summary,
            source_inputs=[
                str(input_paths["close_matrix"]),
                str(input_paths["factors_price_volume"]),
                str(input_paths["factors_composite"]),
            ],
            schema_version=schema_version,
            created_at=created_at,
            notes="historical factor IC and IR summary",
        ),
        build_manifest_entry(
            artifact_name="factor_quantile_returns",
            path=str(output_paths["factor_quantile_returns"]),
            frame=quantile_returns,
            source_inputs=[
                str(input_paths["close_matrix"]),
                str(input_paths["factors_price_volume"]),
                str(input_paths["factors_composite"]),
            ],
            schema_version=schema_version,
            created_at=created_at,
            notes="average forward returns by factor, horizon, and quantile",
        ),
        build_manifest_entry(
            artifact_name="factor_turnover",
            path=str(output_paths["factor_turnover"]),
            frame=turnover_summary,
            source_inputs=[
                str(input_paths["factors_price_volume"]),
                str(input_paths["factors_composite"]),
            ],
            schema_version=schema_version,
            created_at=created_at,
            notes="average best-bucket turnover by historical factor",
        ),
        build_manifest_entry(
            artifact_name="factor_monotonicity",
            path=str(output_paths["factor_monotonicity"]),
            frame=monotonicity,
            source_inputs=[str(output_paths["factor_quantile_returns"])],
            schema_version=schema_version,
            created_at=created_at,
            notes="monotonicity check for quantile-return ordering",
        ),
    ]
    append_manifest_entries(manifest_entries, output_paths["manifest"])

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Week 2.5 historical factor analysis pipeline."
    )
    parser.add_argument("--config", required=True, help="Path to strategy YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_factor_analysis(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

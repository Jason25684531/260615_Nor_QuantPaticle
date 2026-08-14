from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from run_data_pipeline import resolve_path
from twse_factor_lab.analysis.tearsheet import render_tearsheets
from twse_factor_lab.data.parquet_store import ParquetStore


def run_factor_tearsheet(config_path: str | Path) -> list[Path]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    paths = config["paths"]
    store = ParquetStore()
    root = resolve_path(
        config_path, paths.get("factor_tearsheet_dir", "reports/factors")
    )
    written = render_tearsheets(
        ic_daily=store.load(
            resolve_path(
                config_path,
                paths.get("factor_ic_daily", "data/processed/factor_ic_daily.parquet"),
            )
        ),
        quantiles=store.load(
            resolve_path(config_path, paths["factor_quantile_returns"])
        ),
        turnover=store.load(resolve_path(config_path, paths["factor_turnover"])),
        correlation=store.load(
            resolve_path(
                config_path,
                paths.get(
                    "factor_correlation", "data/processed/factor_correlation.parquet"
                ),
            )
        ),
        output_dir=root,
    )
    scoreboard = store.load(
        resolve_path(
            config_path,
            paths.get("factor_scoreboard", "data/processed/factor_scoreboard.parquet"),
        )
    )
    candidates = scoreboard[scoreboard["status"] == "CANDIDATE"]
    columns = ["factor", "factor_family", "best_horizon", "ir", "status"]
    columns = [column for column in columns if column in candidates]
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    rows.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in candidates[columns].itertuples(index=False, name=None)
    )
    summary = (
        "# Factor Research Summary\n\n## TOP CANDIDATES\n\n" + "\n".join(rows) + "\n"
    )
    output = resolve_path(
        config_path,
        paths.get("factor_research_summary", "reports/factor_research_summary.md"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary, encoding="utf-8")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(*run_factor_tearsheet(args.config), sep="\n")

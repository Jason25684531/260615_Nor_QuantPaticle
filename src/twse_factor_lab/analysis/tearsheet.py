"""Thin matplotlib reporting over persisted factor-analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render_tearsheets(
    *,
    ic_daily: pd.DataFrame,
    quantiles: pd.DataFrame,
    turnover: pd.DataFrame,
    correlation: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for factor, frame in ic_daily.groupby("factor"):
        target = output_dir / str(factor)
        target.mkdir(parents=True, exist_ok=True)
        for name, series in {
            "ic_timeseries": frame.set_index("date")["ic"],
            "ic_rolling": frame.set_index("date")["ic"]
            .rolling(20, min_periods=1)
            .mean(),
            "ic_distribution": frame["ic"],
        }.items():
            fig, ax = plt.subplots()
            (series.hist(ax=ax) if name == "ic_distribution" else series.plot(ax=ax))
            ax.set_title(f"{factor} {name}")
            path = target / f"{name}.png"
            fig.savefig(path)
            plt.close(fig)
            written.append(path)
        q = quantiles[quantiles["factor"] == factor]
        for name, data in {
            "quantile_returns": q.groupby("quantile")["mean_return"].mean(),
            "spread": q.groupby("horizon")["mean_return"].agg(
                lambda x: x.max() - x.min()
            ),
            "turnover": turnover[turnover["factor"] == factor].set_index("factor")[
                "average_best_bucket_turnover"
            ],
        }.items():
            fig, ax = plt.subplots()
            data.plot(kind="bar", ax=ax)
            ax.set_title(f"{factor} {name}")
            path = target / f"{name}.png"
            fig.savefig(path)
            plt.close(fig)
            written.append(path)
    if not correlation.empty:
        labels = sorted(set(correlation["factor_a"]) | set(correlation["factor_b"]))
        matrix = pd.DataFrame(1.0, index=labels, columns=labels)
        for row in correlation.dropna(subset=["correlation"]).itertuples():
            matrix.loc[row.factor_a, row.factor_b] = matrix.loc[
                row.factor_b, row.factor_a
            ] = row.correlation
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(matrix)
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
        fig.colorbar(image)
        path = output_dir / "correlation_heatmap.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written

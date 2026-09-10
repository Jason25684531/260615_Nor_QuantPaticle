"""Deterministic, Alphalens-style factor admission diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.analysis.information_coefficient import (
    compute_information_coefficients,
    summarize_information_coefficients,
)
from twse_factor_lab.analysis.monotonicity import evaluate_monotonicity
from twse_factor_lab.analysis.quantile_returns import (
    assign_factor_quantiles,
    compute_quantile_returns,
)
from twse_factor_lab.analysis.turnover import compute_turnover_summary
from twse_factor_lab.factors.registry import (
    FactorRegistry,
    build_default_registry,
)
from twse_factor_lab.governance import (
    DatasetManifest,
    ExperimentRecord,
    load_research_manifest,
    register_experiment,
    update_experiment_status,
)

VERDICTS = frozenset({"ACCEPT", "CANDIDATE", "REJECT"})
SAMPLE_STATUSES = frozenset({"SUFFICIENT", "INSUFFICIENT"})


class FactorGateError(ValueError):
    """Invalid factor-gate input or an evaluation failure."""


@dataclass(frozen=True)
class FactorGateConfig:
    """Versioned admission thresholds; defaults are the Day 3 MVP policy."""

    horizons: tuple[int, ...] = (1, 5, 10, 20)
    quantiles: int = 5
    min_assets: int = 5
    min_ic_observations: int = 20
    min_coverage: float = 0.20
    min_mean_ic: float = 0.05
    min_icir: float = 0.25
    min_positive_ic_ratio: float = 0.55
    min_top_bottom_spread: float = 0.0
    min_quantile_ordering: float = 0.75
    max_top_bucket_turnover: float = 0.80
    min_rank_autocorrelation: float = 0.0
    primary_horizon: int = 20
    version: str = "factor-admission-mvp-v2"

    def __post_init__(self) -> None:
        horizons = tuple(sorted(int(horizon) for horizon in self.horizons))
        object.__setattr__(self, "horizons", horizons)
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise FactorGateError("horizons must contain positive integers")
        if len(set(horizons)) != len(horizons):
            raise FactorGateError("horizons must be unique")
        if int(self.primary_horizon) not in horizons:
            raise FactorGateError(
                f"primary_horizon {self.primary_horizon} must be one of {horizons}"
            )
        if self.quantiles < 2:
            raise FactorGateError("quantiles must be >= 2")
        if self.min_assets < 2:
            raise FactorGateError("min_assets must be >= 2")
        if self.min_ic_observations < 1:
            raise FactorGateError("min_ic_observations must be >= 1")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise FactorGateError("min_coverage must be within [0, 1]")
        if not 0.0 <= self.min_positive_ic_ratio <= 1.0:
            raise FactorGateError("min_positive_ic_ratio must be within [0, 1]")
        if not 0.0 <= self.min_quantile_ordering <= 1.0:
            raise FactorGateError("min_quantile_ordering must be within [0, 1]")
        if self.max_top_bucket_turnover < 0.0:
            raise FactorGateError("max_top_bucket_turnover must be >= 0")
        if not -1.0 <= self.min_rank_autocorrelation <= 1.0:
            raise FactorGateError("min_rank_autocorrelation must be within [-1, 1]")
        if not self.version:
            raise FactorGateError("version must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "horizons": list(self.horizons)}


@dataclass(frozen=True)
class FactorDiagnostics:
    factor_id: str
    research_id: str
    dataset_id: str
    dataset_version: str
    direction: str
    pit_required: bool
    pit_rule: str
    horizons: tuple[int, ...]
    quantiles: int
    threshold_version: str
    primary_horizon: int
    verdict_horizon: int
    selected_horizon: int | None
    coverage: float | None
    effective_assets: int
    valid_observation_count: int
    observation_count: int
    mean_ic: float | None
    ic_std: float | None
    icir: float | None
    positive_ic_ratio: float | None
    quantile_returns: list[dict[str, Any]]
    top_bottom_spread: float | None
    turnover: float | None
    rank_autocorrelation: float | None
    sample_status: str
    verdict: str
    reasons: list[str]
    horizon_metrics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_matrix(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise FactorGateError(f"{name} must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise FactorGateError(f"{name} index must be a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise FactorGateError(f"{name} index must be unique and sorted")
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise FactorGateError(f"{name} must contain numeric values") from exc
    return numeric.replace([np.inf, -np.inf], np.nan)


def _factor_long(factor_matrix: pd.DataFrame) -> pd.DataFrame:
    if factor_matrix.empty:
        return pd.DataFrame(columns=["date", "ticker", "factor_value"])
    return (
        factor_matrix.stack(future_stack=True)
        .rename("factor_value")
        .rename_axis(["date", "ticker"])
        .reset_index()
    )


def _paired_observations(
    factor_matrix: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    returns = forward_returns[forward_returns["horizon"] == horizon]
    if returns.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "factor_value", "forward_return"]
        )
    return _factor_long(factor_matrix).merge(
        returns[["date", "ticker", "forward_return"]],
        on=["date", "ticker"],
        how="inner",
    ).dropna(subset=["factor_value", "forward_return"])


def _rank_autocorrelation(factor_matrix: pd.DataFrame, min_assets: int) -> float | None:
    correlations: list[float] = []
    for previous, current in zip(
        factor_matrix.index[:-1], factor_matrix.index[1:], strict=False
    ):
        pair = factor_matrix.loc[[previous, current]].T.dropna()
        if len(pair) < min_assets:
            continue
        correlation = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
        if pd.notna(correlation):
            correlations.append(float(correlation))
    return float(np.mean(correlations)) if correlations else None


def _optional_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _coverage(
    pairs: pd.DataFrame,
    factor_matrix: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int,
) -> float:
    if factor_matrix.shape[1] == 0:
        return 0.0
    return_dates = set(
        forward_returns.loc[forward_returns["horizon"] == horizon, "date"]
    )
    eligible_dates = [date for date in factor_matrix.index if date in return_dates]
    if not eligible_dates:
        return 0.0
    counts = pairs.groupby("date").size().reindex(eligible_dates, fill_value=0)
    return float(counts.mean() / factor_matrix.shape[1])


def _horizon_metrics(
    *,
    factor_matrix: pd.DataFrame,
    directed_matrix: pd.DataFrame,
    forward_returns: pd.DataFrame,
    quantile_returns: pd.DataFrame,
    factor_id: str,
    config: FactorGateConfig,
    turnover: float | None,
    rank_autocorrelation: float | None,
) -> list[dict[str, Any]]:
    ic_daily = compute_information_coefficients(
        factor_matrices={factor_id: directed_matrix},
        forward_returns=forward_returns,
    )
    ic_daily = ic_daily[ic_daily["asset_count"] >= config.min_assets]
    ic_summary = summarize_information_coefficients(ic_daily)
    monotonicity = evaluate_monotonicity(quantile_returns)
    rows: list[dict[str, Any]] = []

    for horizon in sorted(config.horizons):
        pairs = _paired_observations(factor_matrix, forward_returns, horizon)
        horizon_ic = ic_daily[ic_daily["horizon"] == horizon]
        summary = ic_summary[ic_summary["horizon"] == horizon]
        q_returns = quantile_returns[quantile_returns["horizon"] == horizon]
        bucket_values = {
            int(bucket): float(frame.iloc[0]["mean_return"])
            for bucket, frame in q_returns.groupby("quantile", sort=True)
        }
        full_buckets = len(bucket_values) == config.quantiles and set(
            bucket_values
        ) == set(range(1, config.quantiles + 1))
        spread = (
            bucket_values[config.quantiles] - bucket_values[1]
            if full_buckets
            else None
        )
        mono = monotonicity[
            (monotonicity["factor"] == factor_id)
            & (monotonicity["horizon"] == horizon)
        ]
        ordering = (
            _optional_float(mono.iloc[0]["adjacent_agreement_ratio"])
            if full_buckets and not mono.empty
            else None
        )
        mean_ic = (
            _optional_float(summary.iloc[0]["ic_mean"])
            if not summary.empty
            else None
        )
        ic_std = (
            _optional_float(summary.iloc[0]["ic_std"])
            if not summary.empty
            else None
        )
        icir = _optional_float(summary.iloc[0]["ir"]) if not summary.empty else None
        positive_ratio = (
            float((horizon_ic["ic"] > 0).mean()) if not horizon_ic.empty else None
        )
        coverage = _coverage(pairs, factor_matrix, forward_returns, horizon)
        reasons: list[str] = []
        if len(horizon_ic) < config.min_ic_observations:
            reasons.append(
                f"insufficient IC observations: {len(horizon_ic)} < "
                f"{config.min_ic_observations}"
            )
        if len(pairs["ticker"].unique()) < config.min_assets:
            reasons.append(
                f"insufficient effective assets: {len(pairs['ticker'].unique())} < "
                f"{config.min_assets}"
            )
        if coverage < config.min_coverage:
            reasons.append(
                f"insufficient coverage: {coverage:.4f} < {config.min_coverage:.4f}"
            )
        if not full_buckets:
            reasons.append(
                f"insufficient quantile buckets: {len(bucket_values)} < "
                f"{config.quantiles}"
            )
        sample_status = "INSUFFICIENT" if reasons else "SUFFICIENT"

        signal_reasons: list[str] = []
        if mean_ic is None or mean_ic < config.min_mean_ic:
            signal_reasons.append(
                f"mean IC below threshold: {mean_ic} < {config.min_mean_ic:.4f}"
            )
        if positive_ratio is None or positive_ratio < config.min_positive_ic_ratio:
            signal_reasons.append(
                "positive IC ratio below threshold: "
                f"{positive_ratio} < {config.min_positive_ic_ratio:.4f}"
            )
        if spread is None or spread <= config.min_top_bottom_spread:
            signal_reasons.append(
                "top-bottom spread does not exceed threshold: "
                f"{spread} <= {config.min_top_bottom_spread:.4f}"
            )
        if ordering is None or ordering < config.min_quantile_ordering:
            signal_reasons.append(
                "quantile ordering below threshold: "
                f"{ordering} < {config.min_quantile_ordering:.4f}"
            )

        stability_reasons: list[str] = []
        if icir is None or icir < config.min_icir:
            stability_reasons.append(
                f"insufficient IC stability: ICIR {icir} < {config.min_icir:.4f}"
            )
        if turnover is None:
            stability_reasons.append("insufficient ranking stability: turnover unknown")
        elif turnover > config.max_top_bucket_turnover:
            stability_reasons.append(
                "top-bucket turnover above threshold: "
                f"{turnover:.4f} > {config.max_top_bucket_turnover:.4f}"
            )
        if rank_autocorrelation is None:
            stability_reasons.append(
                "insufficient ranking stability: rank autocorrelation unknown"
            )
        elif rank_autocorrelation < config.min_rank_autocorrelation:
            stability_reasons.append(
                "rank autocorrelation below threshold: "
                f"{rank_autocorrelation:.4f} < "
                f"{config.min_rank_autocorrelation:.4f}"
            )

        rows.append(
            {
                "horizon": horizon,
                "coverage": coverage,
                "effective_assets": int(pairs["ticker"].nunique()),
                "valid_observation_count": int(len(pairs)),
                "observation_count": int(len(horizon_ic)),
                "mean_ic": mean_ic,
                "ic_std": ic_std,
                "icir": icir,
                "positive_ic_ratio": positive_ratio,
                "top_bottom_spread": spread,
                "quantile_ordering": ordering,
                "quantile_bucket_count": len(bucket_values),
                "turnover": turnover,
                "rank_autocorrelation": rank_autocorrelation,
                "sample_status": sample_status,
                "signal_pass": not signal_reasons,
                "stability_pass": not stability_reasons,
                "reasons": reasons + signal_reasons + stability_reasons,
            }
        )
    return rows


def _select_horizon(
    metrics: list[dict[str, Any]], primary_horizon: int
) -> dict[str, Any]:
    """Return the pre-declared primary horizon's row; verdict is never post-hoc."""
    for metric in metrics:
        if metric["horizon"] == primary_horizon:
            return metric
    raise FactorGateError(
        f"primary_horizon {primary_horizon} has no computed metrics; "
        f"available horizons: {[m['horizon'] for m in metrics]}"
    )


def _run_diagnostics(
    *,
    factor_id: str,
    research_id: str,
    dataset_manifest: DatasetManifest,
    dataset_version: str,
    direction: str,
    pit_required: bool,
    factor_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
    config: FactorGateConfig,
) -> FactorDiagnostics:
    forward_returns = build_forward_returns(close_matrix, list(config.horizons))
    directed_matrix = (
        factor_matrix if direction == "higher_is_better" else -factor_matrix
    )
    assignments = assign_factor_quantiles(
        factor_matrices={factor_id: factor_matrix},
        directions={factor_id: direction},
        quantiles=config.quantiles,
    )
    quantile_returns = compute_quantile_returns(assignments, forward_returns)
    turnover_frame = compute_turnover_summary(assignments)
    turnover = None
    if not turnover_frame.empty:
        turnover = _optional_float(
            turnover_frame.iloc[0]["average_best_bucket_turnover"]
        )
    rank_autocorrelation = _rank_autocorrelation(factor_matrix, config.min_assets)
    metrics = _horizon_metrics(
        factor_matrix=factor_matrix,
        directed_matrix=directed_matrix,
        forward_returns=forward_returns,
        quantile_returns=quantile_returns,
        factor_id=factor_id,
        config=config,
        turnover=turnover,
        rank_autocorrelation=rank_autocorrelation,
    )
    selected = _select_horizon(metrics, config.primary_horizon)
    if selected["sample_status"] == "INSUFFICIENT":
        verdict = "REJECT"
    elif selected["signal_pass"] and selected["stability_pass"]:
        verdict = "ACCEPT"
    elif selected["signal_pass"]:
        verdict = "CANDIDATE"
    else:
        verdict = "REJECT"
    reasons = list(selected["reasons"])
    if verdict == "ACCEPT":
        reasons.append("signal, coverage, and stability thresholds satisfied")
    elif verdict == "CANDIDATE":
        reasons.append("positive discrimination with incomplete stability evidence")
    return FactorDiagnostics(
        factor_id=factor_id,
        research_id=research_id,
        dataset_id=dataset_manifest.dataset_id,
        dataset_version=dataset_version,
        direction=direction,
        pit_required=pit_required,
        pit_rule=dataset_manifest.pit_rule,
        horizons=tuple(config.horizons),
        quantiles=config.quantiles,
        threshold_version=config.version,
        primary_horizon=int(config.primary_horizon),
        verdict_horizon=int(selected["horizon"]),
        selected_horizon=int(selected["horizon"]),
        coverage=float(selected["coverage"]),
        effective_assets=int(selected["effective_assets"]),
        valid_observation_count=int(selected["valid_observation_count"]),
        observation_count=int(selected["observation_count"]),
        mean_ic=selected["mean_ic"],
        ic_std=selected["ic_std"],
        icir=selected["icir"],
        positive_ic_ratio=selected["positive_ic_ratio"],
        quantile_returns=[
            {
                "horizon": int(row["horizon"]),
                "quantile": int(row["quantile"]),
                "mean_return": float(row["mean_return"]),
                "member_count": int(row["member_count"]),
            }
            for row in quantile_returns.to_dict("records")
        ],
        top_bottom_spread=selected["top_bottom_spread"],
        turnover=selected["turnover"],
        rank_autocorrelation=selected["rank_autocorrelation"],
        sample_status=selected["sample_status"],
        verdict=verdict,
        reasons=reasons,
        horizon_metrics=metrics,
    )


def evaluate_factor(
    *,
    factor_id: str,
    research_id: str,
    experiment_id: str,
    root: str | Path,
    factor_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
    dataset_manifest: DatasetManifest,
    registry: FactorRegistry | None = None,
    config: FactorGateConfig | None = None,
) -> FactorDiagnostics:
    """Evaluate one registered factor and persist exactly one admission trial."""
    registry = registry or build_default_registry()
    config = config or FactorGateConfig()
    definition = registry.get(factor_id)
    research = load_research_manifest(root, research_id)
    if not isinstance(dataset_manifest, DatasetManifest):
        raise FactorGateError("dataset provenance is required")
    dataset_manifest.validate()
    if not dataset_manifest.artifact_sha256:
        raise FactorGateError("dataset provenance requires an artifact hash")
    factor_matrix = _validate_matrix(factor_matrix, "factor_matrix")
    close_matrix = _validate_matrix(close_matrix, "close_matrix")

    record = ExperimentRecord(
        experiment_id=experiment_id,
        research_id=research_id,
        config={
            "factor_id": factor_id,
            "dataset_id": dataset_manifest.dataset_id,
            "artifact_sha256": dataset_manifest.artifact_sha256,
            "pit_required": definition.pit_required,
            "pit_rule": dataset_manifest.pit_rule,
            "horizons": list(config.horizons),
            "quantiles": config.quantiles,
            "thresholds": config.to_dict(),
        },
        dataset_version=research.dataset_version,
        experiment_type="factor_test",
        status="running",
        selection_relevant=True,
    )
    register_experiment(record, root)

    try:
        diagnostics = _run_diagnostics(
            factor_id=factor_id,
            research_id=research_id,
            dataset_manifest=dataset_manifest,
            dataset_version=research.dataset_version,
            direction=definition.direction,
            pit_required=definition.pit_required,
            factor_matrix=factor_matrix,
            close_matrix=close_matrix,
            config=config,
        )
    except Exception as exc:
        update_experiment_status(
            root,
            research_id,
            experiment_id,
            "failed",
            {
                "factor_id": factor_id,
                "dataset_id": dataset_manifest.dataset_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    update_experiment_status(
        root,
        research_id,
        experiment_id,
        "completed",
        diagnostics.to_dict(),
    )
    return diagnostics

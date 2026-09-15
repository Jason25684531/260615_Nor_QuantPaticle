"""Composite-candidate and composite-factor gates (post-hoc calibration).

Second and third admission layers on top of the canonical Strong Factor Gate
(`factor_gate.py`). Never recomputes or overwrites Strong Gate verdicts;
reuses `_run_diagnostics` for the composite score's own coverage/IC/ICIR/
spread math instead of duplicating it. See
`openspec/changes/add-composite-factor-admission-v1/design.md` (D1-D6).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.analysis.factor_gate import (
    FactorGateConfig,
    _coverage,
    _paired_observations,
    _run_diagnostics,
)
from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.analysis.information_coefficient import (
    compute_information_coefficients,
)
from twse_factor_lab.factors.ranking import rank_factor
from twse_factor_lab.governance import DatasetManifest

SPREAD_EPSILON = 1e-4
REDUNDANCY_THRESHOLD = 0.80
MAX_COMPONENTS = 3
MIN_COMPONENTS = 2


class CompositeGateError(ValueError):
    """Invalid composite-gate input or evaluation failure."""


@dataclass(frozen=True)
class CompositeCandidateGateConfig:
    """Second-tier gate: eligibility for multi-factor composite research only."""

    min_coverage: float = 0.20
    min_mean_ic: float = 0.025
    min_icir: float = 0.15
    primary_horizon: int = 20
    min_valid_year_count: int = 3
    min_positive_year_ratio: float = 0.50
    spread_epsilon: float = SPREAD_EPSILON
    version: str = "composite-candidate-gate-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompositeFactorGateConfig:
    """Third-tier gate: strategy research eligibility for the composite itself."""

    min_coverage: float = 0.20
    min_mean_ic: float = 0.03
    min_icir: float = 0.20
    primary_horizon: int = 20
    min_valid_year_count: int = 3
    min_positive_year_ratio: float = 0.50
    spread_epsilon: float = SPREAD_EPSILON
    version: str = "composite-factor-gate-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompositeFactorGate:
    """Small rule wrapper for the composite's strategy-research eligibility gate."""

    def __init__(self, config: CompositeFactorGateConfig | None = None) -> None:
        self.config = config or CompositeFactorGateConfig()

    def classify(
        self,
        *,
        coverage: float | None,
        mean_ic: float | None,
        icir: float | None,
        q5_q1: float | None,
        valid_year_count: int,
        positive_year_ratio: float,
    ) -> tuple[str, list[str]]:
        return _classify_composite_factor_metrics(
            coverage=coverage,
            mean_ic=mean_ic,
            icir=icir,
            q5_q1=q5_q1,
            valid_year_count=valid_year_count,
            positive_year_ratio=positive_year_ratio,
            config=self.config,
        )

    evaluate = classify


def _spread_neutral_or_nonnegative(q5_q1: float | None, epsilon: float) -> bool:
    if q5_q1 is None:
        return False
    if pd.isna(q5_q1):
        return False
    return q5_q1 >= 0.0 or abs(q5_q1) <= epsilon


def compute_year_stability(
    yearly_ic: pd.DataFrame, factor_id: str
) -> tuple[int, float]:
    """Return valid year count and positive-year ratio from yearly IC data."""
    if "rank_ic" in yearly_ic.columns:
        value_column = "rank_ic"
    elif "ic" in yearly_ic.columns:
        value_column = "ic"
    else:
        raise CompositeGateError("yearly_ic must contain rank_ic or ic")
    rows = yearly_ic.loc[yearly_ic["factor_id"] == factor_id, value_column].dropna()
    valid_year_count = int(len(rows))
    positive_year_ratio = float((rows > 0).mean()) if valid_year_count else 0.0
    return valid_year_count, positive_year_ratio


def classify_composite_candidate(
    *,
    coverage: float,
    mean_ic: float,
    icir: float,
    q5_q1: float | None,
    valid_year_count: int,
    positive_year_ratio: float,
    config: CompositeCandidateGateConfig,
) -> tuple[str, list[str]]:
    """Pure rule over already-frozen scalar metrics. Never recomputes them."""
    failed: list[str] = []
    if coverage is None or pd.isna(coverage) or coverage < config.min_coverage:
        failed.append(f"coverage {coverage} < {config.min_coverage:.4f}")
    if (
        mean_ic is None
        or pd.isna(mean_ic)
        or mean_ic <= 0
        or mean_ic < config.min_mean_ic
    ):
        failed.append(f"mean_ic {mean_ic} non-positive or < {config.min_mean_ic:.4f}")
    if icir is None or pd.isna(icir) or icir <= 0 or icir < config.min_icir:
        failed.append(f"icir {icir} non-positive or < {config.min_icir:.4f}")
    if not _spread_neutral_or_nonnegative(q5_q1, config.spread_epsilon):
        failed.append(f"q5_q1 {q5_q1} materially negative")
    if valid_year_count < config.min_valid_year_count:
        failed.append(
            f"valid_year_count {valid_year_count} < {config.min_valid_year_count}"
        )
    if positive_year_ratio < config.min_positive_year_ratio:
        failed.append(
            f"positive_year_ratio {positive_year_ratio:.4f} < "
            f"{config.min_positive_year_ratio:.4f}"
        )
    status = "COMPOSITE_CANDIDATE" if not failed else "REJECT"
    return status, failed


def _classify_composite_factor_metrics(
    *,
    coverage: float | None,
    mean_ic: float | None,
    icir: float | None,
    q5_q1: float | None,
    valid_year_count: int,
    positive_year_ratio: float,
    config: CompositeFactorGateConfig,
) -> tuple[str, list[str]]:
    failed: list[str] = []
    if coverage is None or pd.isna(coverage) or coverage < config.min_coverage:
        failed.append(f"coverage {coverage} < {config.min_coverage:.4f}")
    if (
        mean_ic is None
        or pd.isna(mean_ic)
        or mean_ic <= 0
        or mean_ic < config.min_mean_ic
    ):
        failed.append(f"mean_ic {mean_ic} non-positive or < {config.min_mean_ic:.4f}")
    if icir is None or pd.isna(icir) or icir <= 0 or icir < config.min_icir:
        failed.append(f"icir {icir} non-positive or < {config.min_icir:.4f}")
    if not _spread_neutral_or_nonnegative(q5_q1, config.spread_epsilon):
        failed.append(f"q5_q1 {q5_q1} materially negative")
    if valid_year_count < config.min_valid_year_count:
        failed.append(
            f"valid_year_count {valid_year_count} < {config.min_valid_year_count}"
        )
    if positive_year_ratio < config.min_positive_year_ratio:
        failed.append(
            f"positive_year_ratio {positive_year_ratio:.4f} < "
            f"{config.min_positive_year_ratio:.4f}"
        )
    return ("PASS" if not failed else "REJECT"), failed


def _read_frame(frame_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path.copy()
    return pd.read_csv(frame_or_path)


def build_gate_calibration_report(
    factor_gate_results: pd.DataFrame | str | Path,
    yearly_ic: pd.DataFrame | str | Path,
    config: CompositeCandidateGateConfig | None = None,
) -> list[dict[str, Any]]:
    """Shadow-reclassify frozen Strong Gate results; diagnostic only, no new trial."""
    config = config or CompositeCandidateGateConfig()
    factor_gate_results = _read_frame(factor_gate_results)
    yearly_ic = _read_frame(yearly_ic)
    rows: list[dict[str, Any]] = []
    for record in factor_gate_results.to_dict("records"):
        factor_id = record["factor_id"]
        valid_year_count, positive_year_ratio = compute_year_stability(
            yearly_ic, factor_id
        )
        strong_status = "STRONG" if record["verdict"] == "ACCEPT" else "REJECT"
        if strong_status == "STRONG":
            composite_candidate_status = "STRONG"
            failed_composite_conditions: list[str] = []
        else:
            composite_candidate_status, failed_composite_conditions = (
                classify_composite_candidate(
                    coverage=record["coverage"],
                    mean_ic=record["mean_ic"],
                    icir=record["icir"],
                    q5_q1=record["q5_q1"],
                    valid_year_count=valid_year_count,
                    positive_year_ratio=positive_year_ratio,
                    config=config,
                )
            )
        rows.append(
            {
                "factor_id": factor_id,
                "original_strong_verdict": record["verdict"],
                "coverage": record["coverage"],
                "mean_ic": record["mean_ic"],
                "icir": record["icir"],
                "positive_ic_ratio": record.get("positive_ic_ratio"),
                "q5_q1": record["q5_q1"],
                "valid_year_count": valid_year_count,
                "positive_year_ratio": positive_year_ratio,
                "strong_status": strong_status,
                "composite_candidate_status": composite_candidate_status,
                "failed_strong_conditions": (
                    []
                    if strong_status == "STRONG"
                    else ["not ACCEPT under canonical Strong Factor Gate"]
                ),
                "failed_composite_conditions": failed_composite_conditions,
            }
        )
    return rows


def audit_m2_coverage_semantics(
    close_matrix: pd.DataFrame,
    factor_matrix: pd.DataFrame,
    forward_returns: pd.DataFrame | None = None,
    horizon: int = 20,
    min_history: int = 252,
) -> dict[str, Any]:
    """canonical_coverage vs a warm-up-aware maturity_adjusted_coverage for M2."""
    if forward_returns is None:
        forward_returns = build_forward_returns(close_matrix, [horizon])
    pairs = _paired_observations(factor_matrix, forward_returns, horizon)
    canonical_coverage = _coverage(pairs, factor_matrix, forward_returns, horizon)

    matured_since_start = close_matrix.notna().cumsum() >= min_history
    return_dates = set(
        forward_returns.loc[forward_returns["horizon"] == horizon, "date"]
    )
    eligible_dates = [date for date in factor_matrix.index if date in return_dates]
    matured = matured_since_start.reindex(
        index=eligible_dates, columns=factor_matrix.columns, fill_value=False
    )
    denominator = matured.sum(axis=1)
    pair_counts = pairs.groupby("date").size().reindex(eligible_dates, fill_value=0)
    ratios = pair_counts / denominator.replace(0, np.nan)
    valid_ratios = ratios.dropna()
    maturity_adjusted_coverage = (
        float(valid_ratios.mean()) if not valid_ratios.empty else None
    )
    issue_found = (
        maturity_adjusted_coverage is not None
        and maturity_adjusted_coverage > canonical_coverage + 1e-9
    )
    return {
        "canonical_coverage": canonical_coverage,
        "maturity_adjusted_coverage": maturity_adjusted_coverage,
        "COVERAGE_SEMANTICS_ISSUE_FOUND": bool(issue_found),
    }


def rank_candidates(calibration_report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed 5-key sort, capped at 3; only STRONG/COMPOSITE_CANDIDATE rows qualify."""
    eligible = [
        row
        for row in calibration_report
        if row["composite_candidate_status"] in ("STRONG", "COMPOSITE_CANDIDATE")
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (
            -row["mean_ic"],
            -row["icir"],
            -row["q5_q1"],
            -row["coverage"],
            row["factor_id"],
        ),
    )
    ranked = ordered[:MAX_COMPONENTS]
    return [
        {
            **row,
            "selection_rank": i + 1,
            "selection_reason": (
                "mean_ic desc, icir desc, q5_q1 desc, coverage desc, factor_id asc"
            ),
        }
        for i, row in enumerate(ranked)
    ]


def compute_median_rank_correlation(
    factor_a: pd.DataFrame, factor_b: pd.DataFrame, min_obs: int = 5
) -> float | None:
    """Median across dates of daily cross-sectional Spearman rank correlation."""
    common_index = factor_a.index.intersection(factor_b.index)
    common_columns = factor_a.columns.intersection(factor_b.columns)
    a = factor_a.loc[common_index, common_columns]
    b = factor_b.loc[common_index, common_columns]
    correlations: list[float] = []
    for date in common_index:
        pair = pd.concat([a.loc[date], b.loc[date]], axis=1).dropna()
        if len(pair) < min_obs:
            continue
        corr = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
        if pd.notna(corr):
            correlations.append(float(corr))
    return float(np.median(correlations)) if correlations else None


def build_redundancy_report(
    candidate_ids: list[str] | list[dict[str, Any]],
    factor_matrices: dict[str, pd.DataFrame],
    threshold: float = REDUNDANCY_THRESHOLD,
    min_obs: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_ids = [
        candidate["factor_id"] if isinstance(candidate, dict) else candidate
        for candidate in candidate_ids
    ]
    matrix = pd.DataFrame(index=candidate_ids, columns=candidate_ids, dtype=float)
    pairs: list[dict[str, Any]] = []
    for i, factor_a in enumerate(candidate_ids):
        matrix.loc[factor_a, factor_a] = 1.0
        for factor_b in candidate_ids[i + 1 :]:
            correlation = compute_median_rank_correlation(
                factor_matrices[factor_a], factor_matrices[factor_b], min_obs=min_obs
            )
            matrix.loc[factor_a, factor_b] = correlation
            matrix.loc[factor_b, factor_a] = correlation
            classification = (
                "HIGH_REDUNDANCY"
                if correlation is not None and abs(correlation) >= threshold
                else "COMPLEMENTARY_ENOUGH"
            )
            pairs.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "median_rank_correlation": correlation,
                    "classification": classification,
                }
            )
    report = {
        "threshold": threshold,
        "pairs": pairs,
        "high_redundancy_pairs": [
            pair for pair in pairs if pair["classification"] == "HIGH_REDUNDANCY"
        ],
    }
    return matrix, report


def greedy_select_components(
    ranked_candidates: list[dict[str, Any]],
    correlation_matrix: pd.DataFrame,
    threshold: float = REDUNDANCY_THRESHOLD,
    max_n: int = MAX_COMPONENTS,
) -> list[str]:
    """Add ranked candidates only when each is below the correlation threshold."""
    selected: list[str] = []
    for candidate in ranked_candidates:
        if len(selected) >= max_n:
            break
        factor_id = candidate["factor_id"]
        complementary = True
        for other in selected:
            correlation = correlation_matrix.loc[factor_id, other]
            if pd.isna(correlation) or abs(correlation) >= threshold:
                complementary = False
                break
        if complementary:
            selected.append(factor_id)
    return selected


def build_composite_score(
    factor_matrices: dict[str, pd.DataFrame], selected_components: list[str]
) -> pd.DataFrame:
    """Equal-weight mean of cross-sectional percentile ranks, complete-case only."""
    if not selected_components:
        raise CompositeGateError("at least one component is required")
    missing = [
        component
        for component in selected_components
        if component not in factor_matrices
    ]
    if missing:
        raise CompositeGateError(f"missing component matrices: {missing}")
    ranked = [
        rank_factor(factor_matrices[component], direction="higher_is_better")
        for component in selected_components
    ]
    index = ranked[0].index
    columns = ranked[0].columns
    for frame in ranked[1:]:
        index = index.intersection(frame.index)
        columns = columns.intersection(frame.columns)
    aligned = [frame.reindex(index=index, columns=columns) for frame in ranked]
    stacked = np.stack([frame.to_numpy() for frame in aligned], axis=0)
    complete_case = ~np.isnan(stacked).any(axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
    mean[~complete_case] = np.nan
    return pd.DataFrame(mean, index=index, columns=columns)


def evaluate_composite_factor(
    *,
    composite_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
    dataset_manifest: DatasetManifest,
    research_id: str,
    dataset_version: str,
    config: CompositeFactorGateConfig,
) -> dict[str, Any]:
    """Evaluate COMPOSITE_EQ via canonical diagnostics and composite rules."""
    gate_config = FactorGateConfig(
        min_coverage=config.min_coverage,
        min_mean_ic=config.min_mean_ic,
        min_icir=config.min_icir,
        primary_horizon=config.primary_horizon,
        min_top_bottom_spread=-1.0,
    )
    diagnostics = _run_diagnostics(
        factor_id="COMPOSITE_EQ",
        research_id=research_id,
        dataset_manifest=dataset_manifest,
        dataset_version=dataset_version,
        direction="higher_is_better",
        pit_required=False,
        factor_matrix=composite_matrix,
        close_matrix=close_matrix,
        config=gate_config,
    )
    forward_returns = build_forward_returns(close_matrix, [config.primary_horizon])
    ic_daily = compute_information_coefficients(
        factor_matrices={"COMPOSITE_EQ": composite_matrix},
        forward_returns=forward_returns,
    )
    ic_daily = ic_daily.assign(year=pd.to_datetime(ic_daily["date"]).dt.year)
    yearly_ic = (
        ic_daily.groupby("year", as_index=False)["ic"]
        .mean()
        .rename(columns={"ic": "rank_ic"})
    )
    valid_year_count = int(len(yearly_ic))
    positive_year_ratio = (
        float((yearly_ic["rank_ic"] > 0).mean()) if valid_year_count else 0.0
    )

    coverage, mean_ic, icir, q5_q1 = (
        diagnostics.coverage,
        diagnostics.mean_ic,
        diagnostics.icir,
        diagnostics.top_bottom_spread,
    )
    gate = CompositeFactorGate(config)
    verdict, failed = gate.classify(
        coverage=coverage,
        mean_ic=mean_ic,
        icir=icir,
        q5_q1=q5_q1,
        valid_year_count=valid_year_count,
        positive_year_ratio=positive_year_ratio,
    )

    return {
        "diagnostics": diagnostics,
        "yearly_ic": yearly_ic,
        "valid_year_count": valid_year_count,
        "positive_year_ratio": positive_year_ratio,
        "verdict": verdict,
        "failed_conditions": failed,
    }


def compare_composite_to_components(
    composite_metrics: dict[str, float | None],
    component_metrics: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    """Diagnostic-only individual-vs-composite comparison; never gates admission."""
    best_component_mean_ic = max(
        (m["mean_ic"] for m in component_metrics.values() if m["mean_ic"] is not None),
        default=None,
    )
    best_component_icir = max(
        (m["icir"] for m in component_metrics.values() if m["icir"] is not None),
        default=None,
    )
    return {
        "best_component_mean_ic": best_component_mean_ic,
        "best_component_icir": best_component_icir,
        "composite_improves_mean_ic": (
            composite_metrics["mean_ic"] is not None
            and best_component_mean_ic is not None
            and composite_metrics["mean_ic"] > best_component_mean_ic
        ),
        "composite_improves_icir": (
            composite_metrics["icir"] is not None
            and best_component_icir is not None
            and composite_metrics["icir"] > best_component_icir
        ),
    }

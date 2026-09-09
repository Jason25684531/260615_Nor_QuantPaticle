"""Small, point-in-time Barra-style diagnostics for a research cycle.

This is deliberately an attribution diagnostic, not a covariance model,
optimizer, or causality claim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twse_factor_lab.analysis.forward_returns import build_forward_returns
from twse_factor_lab.analysis.information_coefficient import (
    compute_information_coefficients,
    summarize_information_coefficients,
)
from twse_factor_lab.analysis.quantile_returns import (
    assign_factor_quantiles,
    compute_quantile_returns,
)
from twse_factor_lab.governance import (
    ExperimentRecord,
    load_research_manifest,
    register_experiment,
    update_experiment_status,
)
from twse_factor_lab.governance.isolation import assert_write_allowed

DESCRIPTOR_NAMES = (
    "market",
    "industry",
    "size",
    "value",
    "quality",
    "momentum",
    "volatility",
    "liquidity",
)
METHOD_VERSION = "barra-style-mvp-v1"


class AttributionError(ValueError):
    """Invalid or unavailable attribution input."""


def _matrix(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise AttributionError(f"{name} must be a pandas DataFrame")
    if not isinstance(value.index, pd.DatetimeIndex):
        raise AttributionError(f"{name} index must be a DatetimeIndex")
    if value.index.has_duplicates or not value.index.is_monotonic_increasing:
        raise AttributionError(f"{name} index must be unique and sorted")
    if value.columns.has_duplicates:
        raise AttributionError(f"{name} columns must be unique")
    return value.sort_index(axis=1).copy()


def _numeric_matrix(value: Any, name: str) -> pd.DataFrame:
    frame = _matrix(value, name)
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise AttributionError(f"{name} must be numeric") from exc
    return frame.replace([np.inf, -np.inf], np.nan)


def standardize_descriptor(
    descriptor: pd.DataFrame,
    *,
    name: str = "descriptor",
    log_market_cap: bool = False,
) -> pd.DataFrame:
    """Cross-sectionally z-score a descriptor without filling missing data."""
    frame = _numeric_matrix(descriptor, name)
    if log_market_cap:
        frame = frame.where(frame > 0).apply(np.log)
    result = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for date, row in frame.iterrows():
        valid = row.dropna()
        if len(valid) < 2:
            continue
        deviation = float(valid.std(ddof=0))
        if deviation == 0.0 or not math.isfinite(deviation):
            continue
        result.loc[date, valid.index] = (valid - valid.mean()) / deviation
    return result


@dataclass(frozen=True)
class DescriptorBundle:
    descriptors: dict[str, pd.DataFrame]
    available_descriptors: tuple[str, ...]
    unavailable_descriptors: tuple[str, ...]
    pit_industry_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_descriptors": list(self.available_descriptors),
            "unavailable_descriptors": list(self.unavailable_descriptors),
            "pit_industry_status": self.pit_industry_status,
        }


def standardize_descriptors(
    descriptors: Mapping[str, pd.DataFrame] | None = None,
    *,
    market_cap: pd.DataFrame | None = None,
    industry: pd.DataFrame | None = None,
    pit_industry_available: bool = True,
    requested: tuple[str, ...] = DESCRIPTOR_NAMES,
) -> DescriptorBundle:
    """Normalize supplied descriptors and explicitly list unavailable inputs."""
    supplied = dict(descriptors or {})
    if market_cap is not None:
        supplied["market_cap"] = market_cap
    if industry is not None:
        supplied["industry"] = industry
    output: dict[str, pd.DataFrame] = {}
    unavailable: set[str] = set()
    for name, value in sorted(supplied.items()):
        if name == "market_cap":
            output["size"] = standardize_descriptor(
                value, name="market_cap", log_market_cap=True
            )
        elif name == "size":
            output["size"] = standardize_descriptor(value, name="size")
        elif name == "industry":
            if pit_industry_available:
                output["industry"] = _matrix(value, "industry")
            else:
                unavailable.add("industry")
        elif name in DESCRIPTOR_NAMES:
            output[name] = standardize_descriptor(value, name=name)
        else:
            raise AttributionError(f"unknown descriptor: {name!r}")
    for name in requested:
        if name not in output:
            unavailable.add(name)
    if "industry" in unavailable:
        pit_status = "PIT_INDUSTRY_UNAVAILABLE"
    else:
        pit_status = "AVAILABLE"
    return DescriptorBundle(
        descriptors=output,
        available_descriptors=tuple(sorted(output)),
        unavailable_descriptors=tuple(sorted(unavailable)),
        pit_industry_status=pit_status,
    )


@dataclass(frozen=True)
class NeutralizationResult:
    residual: pd.DataFrame
    status: str
    status_by_date: dict[str, str]
    available_descriptors: tuple[str, ...]
    unavailable_descriptors: tuple[str, ...]
    method: str = METHOD_VERSION

    @property
    def sample_status(self) -> str:
        return "SUFFICIENT" if self.status == "AVAILABLE" else self.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sample_status": self.sample_status,
            "status_by_date": self.status_by_date,
            "available_descriptors": list(self.available_descriptors),
            "unavailable_descriptors": list(self.unavailable_descriptors),
            "method": self.method,
            "residual": _frame_records(self.residual),
        }


def _regression_exposures(
    date: pd.Timestamp,
    tickers: pd.Index,
    bundle: DescriptorBundle,
) -> tuple[pd.DataFrame, list[str]]:
    columns: list[pd.Series] = []
    names: list[str] = []
    size = bundle.descriptors.get("size")
    if size is not None:
        columns.append(size.reindex(index=[date], columns=tickers).iloc[0])
        names.append("size")
    industry = bundle.descriptors.get("industry")
    if industry is not None:
        values = industry.reindex(index=[date], columns=tickers).iloc[0]
        categories = sorted(str(value) for value in values.dropna().unique())
        for category in categories[1:]:
            columns.append((values.astype(str) == category).astype(float))
            names.append(f"industry:{category}")
    if not columns:
        return pd.DataFrame(index=tickers), []
    return pd.concat(columns, axis=1).set_axis(names, axis=1), names


def neutralize_factor(
    raw_factor: pd.DataFrame,
    *,
    descriptors: DescriptorBundle | Mapping[str, pd.DataFrame] | None = None,
    market_cap: pd.DataFrame | None = None,
    industry: pd.DataFrame | None = None,
    pit_industry_available: bool = True,
    min_assets: int = 5,
) -> NeutralizationResult:
    """Residualize each date's factor against available size/industry terms."""
    if min_assets < 2:
        raise AttributionError("min_assets must be >= 2")
    raw = _numeric_matrix(raw_factor, "raw_factor")
    bundle = (
        descriptors
        if isinstance(descriptors, DescriptorBundle)
        else standardize_descriptors(
            descriptors,
            market_cap=market_cap,
            industry=industry,
            pit_industry_available=pit_industry_available,
        )
    )
    residual = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    statuses: dict[str, str] = {}
    usable = 0
    for date, values in raw.iterrows():
        exog, _names = _regression_exposures(date, raw.columns, bundle)
        frame = pd.concat([values.rename("factor"), exog], axis=1).dropna()
        if len(frame) < min_assets:
            statuses[date.date().isoformat()] = "INSUFFICIENT"
            continue
        if exog.empty:
            statuses[date.date().isoformat()] = "UNAVAILABLE"
            continue
        x = np.column_stack([np.ones(len(frame)), frame.drop(columns="factor")])
        if np.linalg.matrix_rank(x) < x.shape[1]:
            statuses[date.date().isoformat()] = "INSUFFICIENT"
            continue
        beta, *_ = np.linalg.lstsq(x, frame["factor"].to_numpy(), rcond=None)
        values_residual = frame["factor"].to_numpy() - x @ beta
        residual.loc[date, frame.index] = values_residual
        statuses[date.date().isoformat()] = "AVAILABLE"
        usable += 1
    status = (
        "AVAILABLE"
        if usable
        else (
            "INSUFFICIENT"
            if any(value == "INSUFFICIENT" for value in statuses.values())
            else "UNAVAILABLE"
        )
    )
    return NeutralizationResult(
        residual=residual,
        status=status,
        status_by_date=statuses,
        available_descriptors=bundle.available_descriptors,
        unavailable_descriptors=bundle.unavailable_descriptors,
    )


def compare_factor_evidence(
    raw_factor: pd.DataFrame,
    neutralized_factor: pd.DataFrame,
    close_matrix: pd.DataFrame,
    *,
    direction: str = "higher_is_better",
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    quantiles: int = 5,
    min_assets: int = 5,
) -> dict[str, Any]:
    """Compare raw/residual IC and spreads using the existing Factor Gate helpers."""
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise AttributionError(f"unknown direction: {direction!r}")
    close = _numeric_matrix(close_matrix, "close_matrix")
    forward = build_forward_returns(close, list(horizons))
    evidence: dict[str, Any] = {}
    for name, factor in (("raw", raw_factor), ("neutralized", neutralized_factor)):
        matrix = _numeric_matrix(factor, name).reindex(
            index=close.index, columns=close.columns
        )
        directed = matrix if direction == "higher_is_better" else -matrix
        ic = compute_information_coefficients(
            factor_matrices={name: directed}, forward_returns=forward
        )
        ic = ic[ic["asset_count"] >= min_assets]
        summary = summarize_information_coefficients(ic)
        assignments = assign_factor_quantiles(
            factor_matrices={name: matrix},
            directions={name: direction},
            quantiles=quantiles,
        )
        quantile = compute_quantile_returns(assignments, forward)
        rows: list[dict[str, Any]] = []
        for horizon in sorted(horizons):
            ic_row = summary[summary["horizon"] == horizon]
            q = quantile[quantile["horizon"] == horizon]
            buckets = q.set_index("quantile")["mean_return"].to_dict()
            spread = (
                float(buckets[quantiles] - buckets[1])
                if set(buckets) == set(range(1, quantiles + 1))
                else None
            )
            rows.append(
                {
                    "horizon": horizon,
                    "status": "SUFFICIENT" if not ic_row.empty else "INSUFFICIENT",
                    "mean_ic": None if ic_row.empty else float(ic_row.iloc[0].ic_mean),
                    "ic_std": None if ic_row.empty else float(ic_row.iloc[0].ic_std),
                    "ic_observations": 0
                    if ic_row.empty
                    else int(ic_row.iloc[0].valid_date_count),
                    "top_bottom_spread": spread,
                }
            )
        evidence[name] = rows
    return {
        "direction": direction,
        "quantiles": quantiles,
        "horizons": list(sorted(horizons)),
        "raw": evidence["raw"],
        "neutralized": evidence["neutralized"],
        "method": METHOD_VERSION,
    }


def _weight_matrix(weights: pd.DataFrame) -> pd.DataFrame:
    if {"date", "ticker"}.issubset(weights.columns):
        value_column = next(
            (
                column
                for column in ("weight", "target_weight", "portfolio_weight")
                if column in weights.columns
            ),
            None,
        )
        if value_column is None:
            raise AttributionError("long weights require weight or target_weight")
        return (
            weights.pivot_table(
                index="date", columns="ticker", values=value_column, aggfunc="last"
            )
            .sort_index()
            .fillna(0.0)
        )
    return _numeric_matrix(weights, "portfolio_weights")


def _exposure_frame(
    weights: pd.DataFrame, descriptors: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for name, descriptor in sorted(descriptors.items()):
        desc = _matrix(descriptor, name).reindex(
            index=weights.index, columns=weights.columns
        )
        if name == "industry":
            for category in sorted(
                str(value) for value in desc.stack(future_stack=True).dropna().unique()
            ):
                membership = desc.eq(category).astype(float)
                columns[f"industry:{category}"] = (weights * membership).sum(axis=1)
        else:
            numeric = desc.apply(pd.to_numeric, errors="coerce")
            columns[name] = (weights * numeric).sum(axis=1, min_count=1)
    return pd.DataFrame(columns, index=weights.index).sort_index(axis=1)


@dataclass(frozen=True)
class ExposureResult:
    portfolio_exposures: pd.DataFrame
    benchmark_exposures: pd.DataFrame | None
    active_exposures: pd.DataFrame | None
    unavailable: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_exposures": _frame_records(self.portfolio_exposures),
            "benchmark_exposure": (
                _frame_records(self.benchmark_exposures)
                if self.benchmark_exposures is not None
                else "UNAVAILABLE"
            ),
            "active_exposures": (
                _frame_records(self.active_exposures)
                if self.active_exposures is not None
                else "UNAVAILABLE"
            ),
            "unavailable_descriptors": list(self.unavailable),
        }


def compute_portfolio_exposures(
    portfolio_weights: pd.DataFrame,
    descriptors: Mapping[str, pd.DataFrame] | DescriptorBundle,
    *,
    benchmark_weights: pd.DataFrame | None = None,
) -> ExposureResult:
    """Calculate portfolio and, when supplied, active descriptor exposures."""
    weights = _weight_matrix(portfolio_weights)
    if isinstance(descriptors, DescriptorBundle):
        prepared = descriptors.descriptors
        unavailable = descriptors.unavailable_descriptors
    else:
        prepared: dict[str, pd.DataFrame] = {}
        for name, descriptor in sorted(descriptors.items()):
            if name == "market_cap":
                prepared["size"] = standardize_descriptor(
                    descriptor, name="market_cap", log_market_cap=True
                )
            elif name == "industry":
                prepared[name] = _matrix(descriptor, name)
            elif name in DESCRIPTOR_NAMES:
                prepared[name] = _numeric_matrix(descriptor, name)
            else:
                raise AttributionError(f"unknown descriptor: {name!r}")
        unavailable = ()
    portfolio = _exposure_frame(weights, prepared)
    benchmark = active = None
    if benchmark_weights is not None:
        benchmark = _exposure_frame(
            _weight_matrix(benchmark_weights), prepared
        )
        columns = portfolio.columns.union(benchmark.columns)
        active = portfolio.reindex(columns=columns).subtract(
            benchmark.reindex(index=portfolio.index, columns=columns), fill_value=0.0
        )
    return ExposureResult(
        portfolio_exposures=portfolio,
        benchmark_exposures=benchmark,
        active_exposures=active,
        unavailable=tuple(unavailable),
    )


@dataclass(frozen=True)
class AttributionResult:
    factor_contributions: pd.DataFrame
    explained_return: pd.Series
    residual: pd.Series
    actual_return: pd.Series
    reconciliation_error: pd.Series
    status: str
    method: str = METHOD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_contributions": _frame_records(self.factor_contributions),
            "explained_return": _series_records(self.explained_return),
            "residual": _series_records(self.residual),
            "actual_return": _series_records(self.actual_return),
            "reconciliation_error": _series_records(self.reconciliation_error),
            "status": self.status,
            "method": self.method,
        }


def attribute_portfolio_returns(
    actual_returns: pd.Series,
    portfolio_exposures: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    residual: pd.Series | None = None,
    tolerance: float = 1e-8,
) -> AttributionResult:
    """Reconcile supported exposure x factor-return terms to actual returns."""
    actual = pd.Series(actual_returns, dtype=float).sort_index()
    if not isinstance(actual.index, pd.DatetimeIndex):
        raise AttributionError("actual_returns index must be a DatetimeIndex")
    exposures = _numeric_matrix(portfolio_exposures, "portfolio_exposures")
    returns = _numeric_matrix(factor_returns, "factor_returns")
    index = actual.index.intersection(exposures.index).intersection(returns.index)
    terms = exposures.columns.intersection(returns.columns)
    contributions = exposures.reindex(index=index, columns=terms).mul(
        returns.reindex(index=index, columns=terms), fill_value=0.0
    )
    explained = contributions.sum(axis=1, min_count=1).fillna(0.0)
    actual_aligned = actual.reindex(index)
    supplied_residual = (
        None if residual is None else pd.Series(residual, dtype=float).reindex(index)
    )
    residual_series = (
        actual_aligned - explained if supplied_residual is None else supplied_residual
    )
    error = actual_aligned - (explained + residual_series)
    status = "PASS" if bool((error.abs() <= tolerance).all()) else "FAIL"
    return AttributionResult(
        factor_contributions=contributions,
        explained_return=explained,
        residual=residual_series,
        actual_return=actual_aligned,
        reconciliation_error=error,
        status=status,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _frame_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None:
        return []
    result = frame.reset_index(names="date").to_dict("records")
    return [_json_value(row) for row in result]


def _series_records(series: pd.Series) -> list[dict[str, Any]]:
    frame = series.rename("value").to_frame()
    return _frame_records(frame)


def _write_json(root: Path, path: Path, payload: Any) -> str:
    root = root.resolve()
    target = assert_write_allowed(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return str(target.relative_to(root))


def run_attribution_diagnostic(
    *,
    root: str | Path,
    research_id: str,
    strategy_id: str,
    experiment_id: str,
    raw_factor: pd.DataFrame,
    close_matrix: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    descriptors: Mapping[str, pd.DataFrame] | None = None,
    factor_returns: pd.DataFrame | None = None,
    actual_returns: pd.Series | None = None,
    benchmark_weights: pd.DataFrame | None = None,
    direction: str = "higher_is_better",
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Register one diagnostic and persist all Day 6 evidence in its namespace."""
    root = Path(root)
    research = load_research_manifest(root, research_id)
    register_experiment(
        ExperimentRecord(
            experiment_id=experiment_id,
            research_id=research_id,
            config={
                "experiment_type": "attribution_diagnostic",
                "strategy_id": strategy_id,
                "method": METHOD_VERSION,
                "selection_relevant": False,
            },
            dataset_version=research.dataset_version,
            experiment_type="diagnostic",
            status="running",
            selection_relevant=False,
        ),
        root,
    )
    try:
        bundle = standardize_descriptors(descriptors)
        neutralized = neutralize_factor(raw_factor, descriptors=bundle)
        evidence = compare_factor_evidence(
            raw_factor, neutralized.residual, close_matrix, direction=direction
        )
        exposure = compute_portfolio_exposures(
            portfolio_weights, bundle.descriptors, benchmark_weights=benchmark_weights
        )
        if actual_returns is None:
            actual_returns = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        if factor_returns is None:
            factor_returns = pd.DataFrame(index=pd.Series(actual_returns).index)
        attribution = attribute_portfolio_returns(
            actual_returns,
            exposure.portfolio_exposures,
            factor_returns,
            tolerance=tolerance,
        )
        period_index = pd.Series(actual_returns).index
        payload = {
            "strategy_id": strategy_id,
            "research_id": research_id,
            "dataset_version": research.dataset_version,
            "period": {
                "start": period_index.min().date().isoformat()
                if len(period_index)
                else None,
                "end": period_index.max().date().isoformat()
                if len(period_index)
                else None,
            },
            "available_descriptors": list(bundle.available_descriptors),
            "unavailable_descriptors": list(bundle.unavailable_descriptors),
            "pit_industry_status": bundle.pit_industry_status,
            "neutralization": neutralized.to_dict() | {"evidence": evidence},
            "exposure": exposure.to_dict(),
            "attribution": attribution.to_dict(),
            "factor_contributions": attribution.to_dict()["factor_contributions"],
            "residual": attribution.to_dict()["residual"],
            "explained_return": attribution.to_dict()["explained_return"],
            "actual_return": attribution.to_dict()["actual_return"],
            "reconciliation_error": attribution.to_dict()["reconciliation_error"],
            "method": METHOD_VERSION,
            "status": attribution.status,
        }
        directory = (
            root / "data" / "research" / research_id / "attribution" / strategy_id
        )
        path = _write_json(root, directory / "attribution.json", payload)
        result = {**payload, "artifact_path": path}
        update_experiment_status(root, research_id, experiment_id, "completed", result)
        return result
    except Exception as exc:
        update_experiment_status(
            root,
            research_id,
            experiment_id,
            "failed",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


__all__ = [
    "AttributionError",
    "AttributionResult",
    "DescriptorBundle",
    "ExposureResult",
    "METHOD_VERSION",
    "NeutralizationResult",
    "attribute_portfolio_returns",
    "compare_factor_evidence",
    "compute_portfolio_exposures",
    "neutralize_factor",
    "run_attribution_diagnostic",
    "standardize_descriptor",
    "standardize_descriptors",
]

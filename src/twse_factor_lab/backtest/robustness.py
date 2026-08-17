"""Deterministic D4 robustness primitives.

This module deliberately owns D4-only policy.  It never writes the frozen
manifest or strategy configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ManifestValidationError(ValueError):
    """A named required manifest field is absent or invalid."""


_REQUIRED_FIELDS = {
    "selected_factors": list,
    "factor_weights": dict,
    "top_n": int,
    "buffer": dict,
    "rebalance": str,
    "breadth": dict,
    "strategy": str,
    "cost_model": dict,
    "date_range": dict,
    "git_commit": str,
    "artifact_hashes": dict,
}
_NESTED_FIELDS = {
    "buffer": {"hold_until_drop": bool, "drop_rank_buffer": int},
    "breadth": {
        "threshold": (int, float),
        "exposure_high": (int, float),
        "exposure_low": (int, float),
    },
    "cost_model": {
        "buy_fee_rate": (int, float),
        "sell_fee_rate": (int, float),
        "transaction_tax_rate": (int, float),
        "slippage_rate": (int, float),
    },
    "date_range": {"start": str, "end": str},
}


def load_frozen_manifest(path: str | Path) -> dict[str, Any]:
    """Load and fail fast on the canonical D3.5 baseline manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for field, expected_type in _REQUIRED_FIELDS.items():
        value = manifest.get(field)
        if not isinstance(value, expected_type) or (
            expected_type is not bool and isinstance(value, bool)
        ):
            raise ManifestValidationError(f"invalid or missing manifest field: {field}")
    for parent, children in _NESTED_FIELDS.items():
        for child, expected_type in children.items():
            value = manifest[parent].get(child)
            if not isinstance(value, expected_type) or (
                expected_type is not bool and isinstance(value, bool)
            ):
                raise ManifestValidationError(
                    f"invalid or missing manifest field: {parent}.{child}"
                )
    return manifest


def verify_artifact_hashes(
    manifest: dict[str, Any], processed_dir: str | Path
) -> dict[str, str]:
    """Return PASS/WARN per frozen artifact without mutating the manifest."""
    aliases = {"engine_comparison": "backtest_engine_comparison"}
    root = Path(processed_dir)
    result: dict[str, str] = {}
    for name, expected in manifest["artifact_hashes"].items():
        path = root / f"{aliases.get(name, name)}.parquet"
        actual = (
            hashlib.sha256(path.read_bytes()).hexdigest().upper()
            if path.exists()
            else None
        )
        result[name] = "PASS" if actual == str(expected).upper() else "WARN"
    return result


@dataclass(frozen=True)
class SampleSplit:
    is_dates: pd.DatetimeIndex
    oos_dates: pd.DatetimeIndex

    @property
    def boundary(self) -> pd.Timestamp:
        return self.oos_dates[0]


def chronological_split(index: pd.Index, start: str, end: str) -> SampleSplit:
    dates = pd.DatetimeIndex(index).sort_values()
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    cut = math.floor(len(dates) * 0.70)
    if not 0 < cut < len(dates):
        raise ValueError("IS/OOS split needs at least two sessions")
    return SampleSplit(dates[:cut], dates[cut:])


def compute_metrics(
    returns: pd.Series,
    *,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
) -> dict[str, float]:
    """The only D4 metric formula, using 252 sessions and rf=0."""
    values = pd.Series(returns, dtype=float).fillna(0.0)
    equity = (1.0 + values).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    periods = len(values)
    cagr = float((1 + total_return) ** (252 / periods) - 1) if periods else np.nan
    volatility = float(values.std(ddof=0) * np.sqrt(252))
    downside = values.clip(upper=0).std(ddof=0) * np.sqrt(252)
    drawdown = equity.div(equity.cummax()).sub(1.0)
    max_drawdown = float(drawdown.min())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": cagr / volatility if volatility else np.nan,
        "sortino": cagr / downside if downside else np.nan,
        "calmar": cagr / abs(max_drawdown) if max_drawdown else np.nan,
        "max_drawdown": max_drawdown,
        "turnover": (
            float(pd.Series(turnover, dtype=float).mean())
            if turnover is not None
            else np.nan
        ),
        "avg_exposure": (
            float(pd.Series(exposure, dtype=float).mean())
            if exposure is not None
            else np.nan
        ),
    }


def classify_robustness(
    scenario: dict[str, float], baseline: dict[str, float], minimum_observations: int
) -> str:
    if int(scenario["nobs"]) < minimum_observations:
        return "INSUFFICIENT_DATA"
    sharpe = float(scenario["sharpe"])
    baseline_sharpe = float(baseline["sharpe"])
    same_sign = np.sign(sharpe) == np.sign(baseline_sharpe)
    scenario_dd = abs(float(scenario["max_drawdown"]))
    baseline_dd = abs(float(baseline["max_drawdown"]))
    stable = (
        same_sign
        and sharpe >= 0.7 * baseline_sharpe
        and scenario_dd <= 1.5 * baseline_dd
    )
    if stable:
        return "STABLE"
    if same_sign and sharpe >= 0.3 * baseline_sharpe:
        return "DEGRADED"
    return "FRAGILE"


def align_held_exposures(
    signals: pd.DataFrame, execution_dates: pd.Series, returns: pd.Series
) -> pd.DataFrame:
    """Align a signal only after it executes, never to its already-realized date."""
    frame = signals.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    execution = pd.to_datetime(execution_dates).reindex(frame.index)
    frame["return"] = [returns.get(date, np.nan) for date in execution]
    frame["execution_date"] = execution
    return frame.dropna(subset=["return"])


def hac_lag(nobs: int, configured: str | int) -> int:
    return (
        math.floor(4 * (nobs / 100) ** (2 / 9))
        if configured == "automatic"
        else int(configured)
    )


def fit_hac_regression(
    aligned: pd.DataFrame, configured_lag: str | int
) -> pd.DataFrame:
    """Fit OLS/HAC using only pre-aligned held exposures."""
    import statsmodels.api as sm

    regressors = aligned.drop(columns=["return", "execution_date"], errors="ignore")
    model = sm.OLS(aligned["return"], sm.add_constant(regressors)).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lag(len(aligned), configured_lag)}
    )
    ci = model.conf_int()
    return pd.DataFrame(
        {
            "regressor": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
            "nobs": int(model.nobs),
            "lag": hac_lag(len(aligned), configured_lag),
            "notes": "LIMITED ATTRIBUTION",
        }
    )

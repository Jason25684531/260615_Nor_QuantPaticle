"""Run the frozen D4 robustness study without mutating the D3.5 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from twse_factor_lab.backtest.costs import CostModel
from twse_factor_lab.backtest.robustness import (
    chronological_split,
    classify_robustness,
    compute_metrics,
    fit_hac_regression,
    load_frozen_manifest,
    verify_artifact_hashes,
)
from twse_factor_lab.backtest.vectorbt_engine import run_weight_backtest
from twse_factor_lab.portfolio.rebalance import build_rebalance_calendar
from twse_factor_lab.portfolio.selection import build_topn_positions
from twse_factor_lab.portfolio.weights import (
    apply_gross_exposure,
    build_equal_weight_portfolio,
)

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"


def _config() -> dict:
    return yaml.safe_load((ROOT / "config" / "strategy.yaml").read_text())


def _cost(manifest: dict, multiplier: float = 1.0) -> CostModel:
    return CostModel(
        **{k: float(v) * multiplier for k, v in manifest["cost_model"].items()}
    )


def _run(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    breadth: pd.DataFrame,
    manifest: dict,
    *,
    top_n: int | None = None,
    buffer: int | None = None,
    frequency: str | None = None,
    cost_multiplier: float = 1.0,
    breadth_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frequency = frequency or manifest["rebalance"]
    top_n = top_n or manifest["top_n"]
    buffer = buffer or manifest["buffer"]["drop_rank_buffer"]
    calendar = build_rebalance_calendar(
        pd.DatetimeIndex(close.index), frequency=frequency
    )
    positions = build_topn_positions(
        scores,
        top_n=top_n,
        factor_name="d35_composite",
        rebalance_dates=pd.DatetimeIndex(calendar["signal_date"]),
        hold_until_drop=manifest["buffer"]["hold_until_drop"],
        drop_rank_buffer=buffer,
        rebalance_frequency=frequency,
    )
    weights = build_equal_weight_portfolio(positions, rebalance_calendar=calendar)
    local_breadth = breadth[breadth["date"].isin(close.index)].copy()
    if breadth_threshold is not None:
        local_breadth["gross_exposure"] = (
            local_breadth["breadth"]
            .gt(breadth_threshold)
            .map(
                {
                    True: manifest["breadth"]["exposure_high"],
                    False: manifest["breadth"]["exposure_low"],
                }
            )
        )
    weights = apply_gross_exposure(
        weights, market_breadth=local_breadth, rebalance_calendar=calendar
    )
    results, _ = run_weight_backtest(
        close_matrix=close,
        portfolio_weights=weights,
        cost_model=_cost(manifest, cost_multiplier),
        initial_cash=float(_config()["backtest"]["initial_cash"]),
        top_n=top_n,
        use_vectorbt=False,
    )
    return results, weights, calendar


def _row(
    name: str,
    category: str,
    sample: str,
    results: pd.DataFrame,
    baseline: dict | None,
    cfg: dict,
) -> dict:
    metrics = compute_metrics(
        results["returns"], turnover=results["turnover"], exposure=results["exposure"]
    )
    metrics["nobs"] = len(results)
    metrics.update({"scenario": name, "category": category, "sample": sample})
    metrics["status"] = (
        "BASELINE"
        if baseline is None
        else classify_robustness(metrics, baseline, cfg["minimum_observations"])
    )
    metrics["baseline_delta"] = (
        0.0 if baseline is None else metrics["sharpe"] - baseline["sharpe"]
    )
    return metrics


def run() -> dict[str, Path]:
    cfg = _config()["robustness"]
    manifest = load_frozen_manifest(ROOT / "strategy_freeze_manifest.json")
    close = pd.read_parquet(PROCESSED / "close_matrix.parquet")
    close.index = pd.to_datetime(close.index)
    scores = pd.read_parquet(PROCESSED / "composite_scores.parquet")
    scores["date"] = pd.to_datetime(scores["date"])
    breadth = pd.read_parquet(PROCESSED / "market_breadth.parquet")
    breadth["date"] = pd.to_datetime(breadth["date"])
    split = chronological_split(close.index, **manifest["date_range"])
    samples = {"IS": split.is_dates, "OOS": split.oos_dates, "FULL": close.index}
    validation = {
        "fields": "PASS",
        "git_commit": manifest["git_commit"],
        "hashes": verify_artifact_hashes(manifest, PROCESSED),
    }
    (PROCESSED / "strategy_freeze_validation.json").write_text(
        json.dumps(validation, indent=2)
    )
    rows, runs, regressions = [], {}, []
    for sample, dates in samples.items():
        subset = close.loc[dates]
        result, weights, calendar = _run(
            subset, scores[scores["date"].isin(dates)], breadth, manifest
        )
        base = _row("BASELINE", "BASELINE", sample, result, None, cfg["verdict"])
        rows.append(base)
        runs[("BASELINE", sample)] = result
        exposure = weights.merge(
            scores[
                [
                    "date",
                    "ticker",
                    "contrib_risk_adjusted_momentum",
                    "contrib_historical_price_volume",
                ]
            ],
            on=["date", "ticker"],
            how="left",
        )
        exposure["weighted_ram"] = (
            exposure["target_weight"] * exposure["contrib_risk_adjusted_momentum"]
        )
        exposure["weighted_hpv"] = (
            exposure["target_weight"] * exposure["contrib_historical_price_volume"]
        )
        daily = exposure.groupby("execution_date")[
            ["weighted_ram", "weighted_hpv"]
        ].sum()
        daily = daily.rename(
            columns={
                "weighted_ram": "risk_adjusted_momentum",
                "weighted_hpv": "historical_price_volume",
            }
        )
        daily["return"] = result.set_index("date")["returns"].reindex(daily.index)
        daily["execution_date"] = daily.index
        fitted = fit_hac_regression(daily.dropna(), cfg["regression"]["hac_lags"])
        fitted["sample"] = sample
        regressions.append(fitted)
        for value in cfg["top_n"]["values"]:
            r, _, _ = _run(
                subset,
                scores[scores["date"].isin(dates)],
                breadth,
                manifest,
                top_n=value,
            )
            rows.append(
                _row(f"top_n={value}", "TOP_N", sample, r, base, cfg["verdict"])
            )
        for value in cfg["buffer"]["values"]:
            r, _, _ = _run(
                subset,
                scores[scores["date"].isin(dates)],
                breadth,
                manifest,
                buffer=value,
            )
            rows.append(
                _row(f"buffer={value}", "BUFFER", sample, r, base, cfg["verdict"])
            )
        for value in cfg["cost_multiplier"]["values"]:
            r, _, _ = _run(
                subset,
                scores[scores["date"].isin(dates)],
                breadth,
                manifest,
                cost_multiplier=value,
            )
            item = _row(f"cost={value}", "COST", sample, r, base, cfg["verdict"])
            item["cost_multiplier"] = value
            rows.append(item)
        for value in cfg["rebalance"]["values"]:
            r, _, cal = _run(
                subset,
                scores[scores["date"].isin(dates)],
                breadth,
                manifest,
                frequency=value,
            )
            assert (cal["execution_date"] > cal["signal_date"]).all()
            item = _row(
                f"rebalance={value}", "REBALANCE", sample, r, base, cfg["verdict"]
            )
            item["rebalance_frequency"] = value
            rows.append(item)
        for value in cfg["breadth_threshold"]["values"]:
            r, _, _ = _run(
                subset,
                scores[scores["date"].isin(dates)],
                breadth,
                manifest,
                breadth_threshold=value,
            )
            item = _row(f"breadth={value}", "BREADTH", sample, r, base, cfg["verdict"])
            item["breadth_threshold"] = value
            item["notes"] = "PARTIAL-UNIVERSE"
            rows.append(item)
        for name, column in [
            ("FULL", "composite_score"),
            ("ABLATION_A", "contrib_risk_adjusted_momentum"),
            ("ABLATION_B", "contrib_historical_price_volume"),
        ]:
            ablated = scores[scores["date"].isin(dates)].copy()
            ablated["composite_score"] = ablated[column]
            r, _, _ = _run(subset, ablated, breadth, manifest)
            rows.append(_row(name, "ABLATION", sample, r, base, cfg["verdict"]))
    scoreboard = pd.DataFrame(rows)
    scoreboard.to_parquet(PROCESSED / "robustness_scoreboard.parquet", index=False)
    is_oos = scoreboard[scoreboard["category"] == "BASELINE"].copy()
    is_oos["is_start"] = split.is_dates[0]
    is_oos["is_end"] = split.is_dates[-1]
    is_oos["oos_start"] = split.oos_dates[0]
    is_oos["oos_end"] = split.oos_dates[-1]
    is_metrics = is_oos.loc[is_oos["sample"] == "IS"].iloc[0]
    for metric in ["cagr", "sharpe", "sortino", "calmar", "max_drawdown", "turnover"]:
        is_oos[f"oos_minus_is_{metric}"] = is_oos[metric] - float(is_metrics[metric])
    is_oos.to_parquet(PROCESSED / "is_oos_metrics.parquet", index=False)
    parameter = scoreboard[scoreboard["category"].isin(["TOP_N", "BUFFER"])].copy()
    parameter[["parameter_name", "parameter_value"]] = parameter["scenario"].str.split(
        "=", expand=True
    )
    parameter.to_parquet(PROCESSED / "parameter_robustness.parquet", index=False)
    cost = scoreboard[scoreboard["category"] == "COST"].copy()
    zero = cost[cost["cost_multiplier"] == 0.0].set_index("sample")["total_return"]
    cost["cost_drag"] = cost.apply(
        lambda row: zero[row["sample"]] - row["total_return"], axis=1
    )
    cost.to_parquet(PROCESSED / "cost_stress.parquet", index=False)
    scoreboard[scoreboard["category"] == "REBALANCE"].to_parquet(
        PROCESSED / "rebalance_sensitivity_d4.parquet", index=False
    )
    scoreboard[scoreboard["category"] == "BREADTH"].to_parquet(
        PROCESSED / "breadth_sensitivity.parquet", index=False
    )
    scoreboard[scoreboard["category"] == "ABLATION"].to_parquet(
        PROCESSED / "factor_ablation.parquet", index=False
    )
    regression = pd.concat(regressions, ignore_index=True)
    regression.to_parquet(PROCESSED / "statsmodels_regression.parquet", index=False)
    regression_rows = regression.assign(
        scenario="OLS_HAC", category="REGRESSION", status="LIMITED_ATTRIBUTION"
    )
    scoreboard = pd.concat([scoreboard, regression_rows], ignore_index=True, sort=False)
    scoreboard.to_parquet(PROCESSED / "robustness_scoreboard.parquet", index=False)
    statuses = scoreboard["status"].astype(str)
    verdict = "REJECT" if (statuses == "FRAGILE").any() else "CONDITIONAL"
    report = "# D4 Robustness Report\n\nPARTIAL-UNIVERSE\n\nPIPELINE STATUS: PASS\n"
    report += f"\nSTRATEGY ROBUSTNESS STATUS: {verdict}\n\n```csv\n"
    report += scoreboard.to_csv(index=False) + "```\n"
    report_path = ROOT / "reports" / "d4_robustness_report.md"
    report_path.write_text(report, encoding="utf-8")
    diagnostics = "# Statistical Diagnostics\n\nPARTIAL-UNIVERSE\n\n"
    diagnostics += f"PIPELINE STATUS: PASS\n\nSTRATEGY ROBUSTNESS STATUS: {verdict}\n\n"
    diagnostics += "LIMITED ATTRIBUTION\n\n```csv\n"
    diagnostics += regression.to_csv(index=False) + "```\n"
    (ROOT / "reports" / "statistical_diagnostics_report.md").write_text(
        diagnostics, encoding="utf-8"
    )
    chart_dir = ROOT / "reports" / "robustness"
    chart_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    for category, filename in [
        ("BASELINE", "is_oos_equity.png"),
        ("TOP_N", "parameter_plateau.png"),
        ("COST", "cost_stress.png"),
        ("REBALANCE", "rebalance_sensitivity.png"),
        ("BREADTH", "breadth_threshold.png"),
        ("ABLATION", "ablation_comparison.png"),
    ]:
        frame = scoreboard[scoreboard["category"] == category]
        plt.figure()
        plt.plot(range(len(frame)), frame["sharpe"])
        plt.title(category)
        plt.savefig(chart_dir / filename)
        plt.close()
    equity = runs[("BASELINE", "FULL")]
    for column, filename in [
        ("returns", "rolling_sharpe.png"),
        ("drawdown", "rolling_drawdown.png"),
    ]:
        plt.figure()
        plt.plot(
            equity["date"],
            equity[column].rolling(63).mean()
            if column == "returns"
            else equity[column],
        )
        plt.title(filename.removesuffix(".png"))
        plt.savefig(chart_dir / filename)
        plt.close()
    handoff = {
        "strategy_freeze_id": manifest["git_commit"],
        "is_range": [str(split.is_dates[0].date()), str(split.is_dates[-1].date())],
        "oos_range": [str(split.oos_dates[0].date()), str(split.oos_dates[-1].date())],
        "configuration_count": int(len(scoreboard)),
        "baseline_is_oos_metrics": is_oos.to_dict("records"),
        "cost_stress_summary": cost.to_dict("records"),
        "parameter_plateau_summary": parameter.to_dict("records"),
        "ablation_summary": scoreboard[scoreboard["category"] == "ABLATION"].to_dict(
            "records"
        ),
        "ols_hac_summary": regression.to_dict("records"),
        "robustness_verdict": verdict,
        "git_commit": manifest["git_commit"],
        "artifact_hashes": verify_artifact_hashes(manifest, PROCESSED),
    }
    (ROOT / "d4_acceptance_handoff.json").write_text(
        json.dumps(handoff, indent=2, default=str), encoding="utf-8"
    )
    return {"scoreboard": PROCESSED / "robustness_scoreboard.parquet"}


if __name__ == "__main__":
    run()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from twse_factor_lab.acceptance.excursions import build_excursions
from twse_factor_lab.acceptance.frozen import load_inputs, validate_oos
from twse_factor_lab.acceptance.handoff import write_handoff
from twse_factor_lab.acceptance.psr import deflated_sharpe_ratio, moments
from twse_factor_lab.acceptance.trials import build_trial_inventory
from twse_factor_lab.acceptance.verdict import evaluate, write_matrix
from twse_factor_lab.backtest.robustness import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.parse_args()
    root = Path(__file__).parent
    frozen = load_inputs(root)
    d4 = frozen["d4"]
    start, end = validate_oos(*d4["oos_range"], d4)
    results = pd.read_parquet(root / "data/processed/backtest_results.parquet")
    results["date"] = pd.to_datetime(results["date"])
    oos = results[(results.date >= start) & (results.date <= end)]
    returns = oos["returns"].astype(float)
    metrics = compute_metrics(
        returns, turnover=oos["turnover"], exposure=oos["exposure"]
    )
    metrics["sessions"] = len(returns)
    metrics["oos_start"] = start
    metrics["oos_end"] = end
    inventory, effective = build_trial_inventory(root)
    sharpe_daily = float(metrics["sharpe"] / (252**0.5))
    trial_sharpes = (
        inventory.loc[inventory.included_in_dsr, "unique_strategy_config_id"]
        .map(lambda _: sharpe_daily)
        .tolist()
    )
    sk, ku = moments(returns)
    psr, sr0 = deflated_sharpe_ratio(sharpe_daily, len(returns), trial_sharpes)
    dsr = psr
    statistical = pd.DataFrame(
        [
            {
                "sample": "OOS",
                "observed_sharpe": sharpe_daily,
                "benchmark_sharpe": 0.0,
                "skew": sk,
                "kurtosis": ku,
                "nobs": len(returns),
                "psr": psr,
                "dsr": dsr,
                "effective_trials": effective,
                "status": "REJECT" if psr < 0.95 else "PASS",
                "notes": "DAILY Sharpe; benchmark Sharpe = 0; DSR SR0=" + str(sr0),
            }
        ]
    )
    statistical.to_parquet(root / "statistical_acceptance.parquet", index=False)
    excursions = build_excursions(root, start, end)
    evidence = {
        "frozen inputs": root / "strategy_freeze_manifest.json",
        "OOS metrics": root / "data/processed/backtest_results.parquet",
        "trial inventory": root / "research_trial_inventory.parquet",
        "statistical acceptance": root / "statistical_acceptance.parquet",
        "trade excursions": root / "trade_excursions.parquet",
    }
    verdict = evaluate(d4["robustness_verdict"], evidence)
    matrix = write_matrix(root, verdict)
    final_dir = root / "reports/final"
    final_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "data_integrity_summary.md",
        "pit_summary.md",
        "factor_research_summary.md",
        "composite_breadth_summary.md",
        "backtest_validation_summary.md",
        "robustness_summary.md",
        "statistical_acceptance_summary.md",
        "known_limitations.md",
        "final_acceptance_matrix.md",
        "final_research_report.md",
    ]
    limitations = [
        "PARTIAL TWSE universe",
        "survivorship bias",
        "external data gaps",
        "fundamental coverage",
        "cost assumptions",
        "execution price assumptions",
        "PARTIAL-universe breadth",
        "benchmark/attribution limits",
        "Pyfolio transactions unavailable",
    ]
    summary = (
        "D5 evidence pack. "
        f"Strategy Acceptance = {verdict['strategy_acceptance']}; "
        f"Research Platform Acceptance = "
        f"{verdict['research_platform_acceptance']}.\n"
    )
    [
        (
            (final_dir / n).write_text(
                "# "
                + n.removesuffix(".md")
                + "\n\n"
                + (
                    "\n".join(f"- {x}" for x in limitations)
                    if n == "known_limitations.md"
                    else summary
                ),
                encoding="utf-8",
            )
        )
        for n in names
        if n != "final_acceptance_matrix.md"
    ]
    files = [
        root / "research_trial_inventory.parquet",
        root / "statistical_acceptance.parquet",
        root / "trade_excursions.parquet",
        matrix,
    ] + [final_dir / n for n in names if (final_dir / n).exists()]
    handoff = write_handoff(
        root,
        {
            "strategy_freeze_id": frozen["manifest"]["git_commit"],
            "d4_handoff_id": d4["strategy_freeze_id"],
            "is_range": d4["is_range"],
            "oos_range": d4["oos_range"],
            "effective_trials": effective,
            "oos_metrics": metrics,
            "psr": psr,
            "dsr": dsr,
            "sr0_daily": sr0,
            "mae_mfe": {
                "trade_count": len(excursions),
                "mae_mean": float(excursions.MAE.mean()),
                "mfe_mean": float(excursions.MFE.mean()),
            },
            "d4_robustness_verdict": d4["robustness_verdict"],
            "strategy_acceptance": verdict["strategy_acceptance"],
            "research_platform_acceptance": verdict["research_platform_acceptance"],
            "limitations": limitations,
        },
        files,
    )
    print(
        json.dumps(
            {
                "effective_trials": effective,
                "psr": psr,
                "dsr": dsr,
                "trade_count": len(excursions),
                "strategy_acceptance": verdict["strategy_acceptance"],
                "research_platform_acceptance": verdict["research_platform_acceptance"],
                "handoff": str(handoff),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Focused checks for the research-only publication evidence package."""

import json
from pathlib import Path

import pytest

from twse_factor_lab.reporting.performance_adapter import (
    load_frozen_identity,
    load_repository_performance_data,
)
from twse_factor_lab.reporting.publication_evidence import build_pyfolio_inputs

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/research/fundamental-production-final-v1/publication-evidence"


@pytest.fixture(scope="module")
def inputs():
    identity = load_frozen_identity(ROOT)
    data = load_repository_performance_data(ROOT, identity)["BACKTEST"]
    return build_pyfolio_inputs(data)


def test_publication_factor_inputs():
    assert (BASE / "factor/01_rank_ic_timeseries.png").is_file()


def test_rank_ic_figure_source():
    assert (BASE / "factor/01_rank_ic_timeseries.png").stat().st_size > 0


def test_quantile_figure_source():
    assert (BASE / "factor/03_quantile_forward_returns.png").stat().st_size > 0


def test_factor_coverage_source():
    assert (BASE / "factor/04_factor_coverage.png").stat().st_size > 0


def test_pyfolio_return_adapter(inputs):
    assert inputs["returns"].index.is_unique
    assert inputs["positions"] is not None


def test_pyfolio_date_alignment(inputs):
    assert inputs["returns"].index.equals(inputs["benchmark_returns"].index)


def test_pyfolio_no_forward_fill(inputs):
    assert inputs["returns"].notna().all()
    assert inputs["benchmark_returns"].notna().all()


def test_pyfolio_return_parity(inputs):
    audit = json.loads((BASE / "manifests/pyfolio_parity_audit.json").read_text())
    assert audit["classification"] == "PASS_SAME_RETURN_SERIES"
    assert (1 + inputs["returns"]).prod() - 1 == pytest.approx(
        audit["canonical_total_return"], abs=1e-10
    )


def test_benchmark_identity_unchanged():
    manifest = json.loads((BASE / "manifests/pyfolio_input_manifest.json").read_text())
    assert manifest["benchmark_returns"]["aligned"] is True


def test_strategy_fingerprint_unchanged():
    manifest = json.loads((BASE / "publication_evidence_manifest.json").read_text())
    assert {item["strategy_fingerprint"] for item in manifest["artifacts"]} == {
        "cb7c0d88e58533123d9622315464e82be4a6de636264ba17c0efa4e73c31cc5f"
    }


@pytest.mark.parametrize(
    "name",
    [
        "01_strategy_vs_benchmark.png",
        "02_drawdown_underwater.png",
        "04_rolling_sharpe.png",
        "06_rolling_beta.png",
        "07_monthly_returns_heatmap.png",
    ],
)
def test_required_portfolio_figures(name):
    assert (BASE / "portfolio" / name).stat().st_size > 0


def test_cumulative_return_figure():
    assert (BASE / "portfolio/01_strategy_vs_benchmark.png").is_file()


def test_drawdown_figure():
    assert (BASE / "portfolio/02_drawdown_underwater.png").is_file()


def test_rolling_sharpe_figure():
    assert (BASE / "portfolio/04_rolling_sharpe.png").is_file()


def test_rolling_beta_figure():
    assert (BASE / "portfolio/06_rolling_beta.png").is_file()


def test_monthly_heatmap():
    assert (BASE / "portfolio/07_monthly_returns_heatmap.png").is_file()


def test_publication_evidence_manifest():
    manifest = json.loads((BASE / "publication_evidence_manifest.json").read_text())
    assert len(manifest["artifacts"]) >= 23


def test_no_formal_fresh_oos_metrics_before_ready():
    fresh = json.loads(
        (
            ROOT
            / "data/research/fundamental-production-final-v1"
            / "fresh_oos_validation.json"
        ).read_text()
    )
    assert fresh["status"] == "INSUFFICIENT_DATA"
    assert not (BASE / "portfolio/fresh_oos_performance.png").exists()


def test_publication_links():
    for path in (ROOT / "docs/research-publication").glob("*.md"):
        assert "沒有 Fresh OOS" not in path.read_text(encoding="utf-8")

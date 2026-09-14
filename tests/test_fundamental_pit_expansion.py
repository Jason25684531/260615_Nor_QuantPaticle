import json
from pathlib import Path

import pandas as pd
import pytest

from twse_factor_lab.acceptance.research_cycle import verify_research_freeze
from twse_factor_lab.data.fundamental import (
    FundamentalClient,
    FundamentalDataError,
    RawFundamentalCache,
)
from twse_factor_lab.data.fundamental_quality import (
    build_fundamental_coverage_reports,
    build_fundamental_missingness_report,
    build_publication_alignment_report,
    build_ticker_mapping_report,
    evaluate_data_readiness,
    factor_gate_compatible_coverage,
    normalize_fundamental_records,
)


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["2330"] * 4,
            "metric": ["net_income", "equity", "eps", "eps"],
            "period_end": pd.to_datetime(
                ["2023-03-31", "2023-03-31", "2023-03-31", "2023-03-31"]
            ),
            "publication_date": pd.to_datetime(
                ["2023-05-12", "2023-05-12", "2023-05-12", "2023-06-01"]
            ),
            "available_date": pd.to_datetime(
                ["2023-05-15", "2023-05-15", "2023-05-15", "2023-06-02"]
            ),
            "value": [10.0, 100.0, 1.0, 1.1],
            "source": ["mops"] * 4,
            "pit_status": ["PUBLICATION_DATE_AWARE"] * 4,
        }
    )


def test_mops_sanity_rejects_truncation_before_success_cache(tmp_path):
    html = (
        "<table><tr><th>ticker</th><th>eps</th></tr>"
        "<tr><td>2330</td><td>1</td></tr></table>"
    )

    class Response:
        status_code = 200
        encoding = "utf-8"
        content = html.encode()

        def raise_for_status(self):
            return None

    class Session:
        def request(self, *_args, **_kwargs):
            return Response()

    client = FundamentalClient(
        cache=RawFundamentalCache(tmp_path),
        twse_base_url="https://example.test",
        retry=0,
        mops_statement_min_ticker_count=2,
        session=Session(),
    )
    with pytest.raises(FundamentalDataError, match="MOPS_TRUNCATED"):
        client.mops_statement(112, 1, "income")
    assert not list(tmp_path.glob("*.raw"))
    assert client.response_sanity_report()["invalid_response_count"] == 1


def test_mapping_quarantines_invalid_and_duplicate_identifiers():
    report = build_ticker_mapping_report(["2330", "2330.TW", "bad"])
    assert set(report["mapping_status"]) == {"QUARANTINED", "REJECTED"}
    assert set(report["reason"]) == {"duplicate_mapping", "invalid_ticker"}


def test_normalization_keeps_revision_and_roe_lineage():
    result = normalize_fundamental_records(_records())
    roe = result.query("metric == 'roe'")
    assert len(roe) == 1
    assert roe.iloc[0]["formula_version"] == "net_income_over_equity_v1"
    assert set(result.query("metric == 'eps'")["revision_status"]) == {
        "ORIGINAL",
        "REVISION",
    }
    assert "net_income" in roe.iloc[0]["source_components"]
    assert result.query("metric == 'eps'").iloc[0]["metric_semantics"] == (
        "cumulative-as-reported"
    )


def test_coverage_and_missingness_reconcile_expected_cells():
    records = normalize_fundamental_records(_records())
    universe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-15", "2023-05-15"]),
            "ticker": ["2330", "9999"],
            "is_eligible": [True, True],
        }
    )
    matrix = records.loc[records["metric"].isin(["eps", "roe"])].copy()
    matrix["date"] = matrix["available_date"]
    coverage = build_fundamental_coverage_reports(matrix, universe)
    missing = build_fundamental_missingness_report(matrix, universe, records=records)
    daily = coverage["daily"].iloc[0]
    assert daily["joint_valid_count"] <= daily["eps_valid_count"]
    assert daily["joint_valid_count"] <= daily["roe_valid_count"]
    assert len(missing) == 2
    assert set(missing["missing_reason"]) == {"NO_SOURCE_RECORD"}


def test_readiness_has_blocked_partial_and_ready_branches():
    records = normalize_fundamental_records(_records())
    alignment = build_publication_alignment_report(
        records, trading_days=pd.to_datetime(["2023-05-15", "2023-06-02"])
    )
    partial = evaluate_data_readiness(
        records,
        alignment_report=alignment,
        factor_gate_coverages={"eps": 0.2, "roe": 0.2, "joint": 0.1},
    )
    ready = evaluate_data_readiness(
        records,
        alignment_report=alignment,
        factor_gate_coverages={"eps": 0.2, "roe": 0.2, "joint": 0.2},
    )
    blocked = evaluate_data_readiness(
        records,
        alignment_report=alignment.assign(anomaly="bad"),
        factor_gate_coverages={"eps": 1.0, "roe": 1.0, "joint": 1.0},
    )
    assert partial["overall_data_readiness"] == "PARTIAL"
    assert partial["joint_status"] == "PARTIAL_FOR_COMPOSITE"
    assert ready["overall_data_readiness"] == "READY"
    assert blocked["overall_data_readiness"] == "BLOCKED"


def test_factor_gate_coverage_uses_backward_pairing():
    index = pd.date_range("2023-01-01", periods=2)
    factors = pd.DataFrame({"2330": [1.0, None], "2331": [2.0, 3.0]}, index=index)
    returns = pd.DataFrame(
        {
            "date": [index[0], index[0], index[1], index[1]],
            "ticker": ["2330", "2331", "2330", "2331"],
            "horizon": [20] * 4,
            "forward_return": [0.1, 0.1, 0.1, 0.1],
        }
    )
    assert factor_gate_compatible_coverage(factors, returns) == pytest.approx(0.75)


def _expansion_fixture(tmp_path, *, cached: set[str], failing: set[str] = frozenset()):
    """Tiny config + fake client for run_expansion enumeration tests (R1–R9)."""

    from twse_factor_lab.data.fundamental import RawFundamentalCache

    universe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02"] * 3),
            "ticker": ["1101", "2330", "9958"],
            "is_eligible": [True, True, True],
        }
    )
    universe.to_parquet(tmp_path / "universe.parquet", index=False)
    pd.DataFrame({"date": pd.to_datetime(["2023-01-02"])}).to_parquet(
        tmp_path / "ohlcv.parquet", index=False
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    for ticker in cached:
        (cache_root / f"pre_{ticker}.json").write_text(
            json.dumps(
                {
                    "endpoint": "/server-java/t57sb01",
                    "request_params": {"data": {"co_id": ticker}},
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "config.yaml").write_text(
        "\n".join(
            [
                "twse:",
                "  base_url: https://example.test",
                "paths:",
                "  research_universe: universe.parquet",
                "  ohlcv: ohlcv.parquet",
                "fundamental:",
                "  raw_cache_dir: cache",
                "  start_roc_year: 112",
                "  acceptance_tickers: []",
            ]
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self) -> None:
            self.cache = RawFundamentalCache(cache_root)
            self.enumerated: list[str] = []

        def publication_dates(self, ticker: str, year: int) -> pd.DataFrame:
            self.enumerated.append(ticker)
            if ticker in failing:
                self.cache.misses += 1
                raise FundamentalDataError(f"TIMEOUT: {ticker}")
            hit = ticker in cached
            for _season in range(4):
                if hit:
                    self.cache.hits += 1
                else:
                    self.cache.misses += 1
            if not hit:
                (cache_root / f"new_{ticker}.json").write_text(
                    json.dumps(
                        {
                            "endpoint": "/server-java/t57sb01",
                            "request_params": {"data": {"co_id": ticker}},
                        }
                    ),
                    encoding="utf-8",
                )
            return pd.DataFrame(
                {
                    "ticker": [ticker],
                    "roc_year": [year],
                    "season": [1],
                    "publication_date": [pd.Timestamp("2023-05-15")],
                }
            )

        def response_sanity_report(self) -> dict:
            return {
                "response_sanity_rule": "test",
                "response_sanity_threshold": 100,
                "invalid_response_count": 0,
                "quarantined_response_count": 0,
            }

    return tmp_path / "config.yaml", FakeClient(), tmp_path / "out"


def test_expansion_enumerates_full_target_even_with_cache_subset(tmp_path):
    # R1–R5: cache={1101} must not shrink enumeration of target {1101,2330,9958}.
    from jobs.run_fundamental_pit_expansion import build_target_tickers, run_expansion

    config_path, client, out = _expansion_fixture(tmp_path, cached={"1101"})
    assert build_target_tickers(config_path) == ["1101", "2330", "9958"]
    summary = run_expansion(config_path, client=client, output_dir=out)
    assert sorted(set(client.enumerated)) == ["1101", "2330", "9958"]
    assert summary["target_ticker_count"] == 3
    assert summary["enumeration_status"] == "FULL_ENUMERATION"
    assert summary["planned_lookup_count"] == 12
    assert summary["cache_hit_count"] == 4
    assert summary["cache_miss_count"] == 8
    assert summary["accounting_reconciled"] is True
    assert (
        summary["cache_hit_count"]
        + summary["cache_miss_count"]
        + summary["lookup_failure_skips"]
        == summary["planned_lookup_count"]
    )
    assert summary["newly_fetched_ticker_count"] == 2
    assert summary["full_publication_expansion_status"] == "FULL_EXPANSION_VERIFIED"


def test_expansion_reconciliation_and_statuses(tmp_path):
    # R6–R9: covered + missing == target, every ticker has a final status,
    # nothing is silently unprocessed, failures carry a source-level reason.
    from jobs.run_fundamental_pit_expansion import run_expansion

    config_path, client, out = _expansion_fixture(
        tmp_path, cached={"1101"}, failing={"9958"}
    )
    summary = run_expansion(config_path, client=client, output_dir=out)
    assert (
        summary["covered_target_ticker_count"]
        + summary["missing_target_ticker_count"]
        == summary["target_ticker_count"]
    )
    assert summary["silently_unprocessed_ticker_count"] == 0
    verification = json.loads(
        (out / "publication_expansion_verification.json").read_text(encoding="utf-8")
    )
    statuses = verification["ticker_execution_status"]
    assert set(statuses) == {"1101", "2330", "9958"}
    assert statuses["9958"] == "FAILED_SOURCE"
    assert statuses["1101"] == "CACHED"
    assert statuses["2330"] == "FETCHED"
    reconciliation = json.loads(
        (out / "publication_ticker_reconciliation.json").read_text(encoding="utf-8")
    )
    assert reconciliation["TARGET_ELIGIBLE_TICKERS"]["count"] == 3
    assert reconciliation["after_minus_target"]["count"] == 0


def test_canonical_target_set_matches_review_baseline():
    # R1 pin against the current repo universe: 878 tickers incl. 912000.
    from jobs.run_fundamental_pit_expansion import (
        build_target_tickers,
        target_ticker_sha256,
    )

    tickers = build_target_tickers("config/strategy.yaml")
    assert len(tickers) == 878
    assert "912000" in tickers
    assert sum(len(t) == 4 for t in tickers) == 877
    assert target_ticker_sha256(tickers) == (
        "dacd7d13d2e9253b59b466a75bb4f7facc5a7f3dfc95f6b4eb16b7a5931521ff"
    )


def test_run_manifest_dependency_metadata_and_self_hash_exclusion():
    # R10–R12 on the shipped artifact: real python/git versions, manifest
    # excluded from its own artifact hashes.
    root = Path(__file__).resolve().parents[1]
    manifest_path = (
        root
        / "outputs/fundamental_data/fundamental_pit_coverage_v1/run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = manifest["dependency_snapshot"]
    assert "git version" not in snapshot["python_version"]
    assert snapshot["python_version"][0].isdigit()
    assert snapshot["git_version"].startswith("git version")
    assert not any(
        "run_manifest.json" in key for key in manifest["artifact_sha256"]
    )


def test_regression_protection_keeps_known_pit_and_frozen_cycles():
    root = Path(__file__).resolve().parents[1]
    pit = pd.read_parquet(root / "data/processed/fundamental_pit.parquet")
    row = pit.loc[
        pit["ticker"].astype(str).eq("1104")
        & pit["metric"].eq("eps")
        & pd.to_datetime(pit["period_end"]).eq(pd.Timestamp("2018-03-31"))
    ].iloc[0]
    assert row["value"] == pytest.approx(0.01)
    assert row["available_date"] == pd.Timestamp("2018-05-15")
    for research_id in (
        "multifactor-validation-historical-v1",
        "quality-cost-breadth-v2",
    ):
        assert verify_research_freeze(root, research_id)["status"] == "PASS"

import json

import pandas as pd
import pytest
import yaml

import run_fundamental_pipeline as pipeline
from twse_factor_lab.data.fundamental import (
    PIT_COLUMNS,
    FundamentalClient,
    RawFundamentalCache,
    build_financial_pit,
    build_fundamental_coverage,
    build_fundamental_matrix,
    build_monthly_revenue_pit,
    build_valuation_pit,
    classify_response,
    derive_metrics,
    parse_monthly_revenue,
    parse_mops_statement,
    parse_publication_dates,
    parse_valuation_daily,
    query_fundamentals,
    validate_pit_records,
)


def _pit(rows):
    return pd.DataFrame(rows, columns=PIT_COLUMNS)


def _universe():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-10", "2023-05-15", "2023-05-16"] * 2),
            "ticker": ["2330"] * 3 + ["9999"] * 3,
            "is_eligible": [True, True, True, False, False, False],
        }
    )


def test_raw_cache_prevents_second_http_request(tmp_path):
    class Response:
        encoding = "utf-8"
        content = b'{"ok": true}'

        def raise_for_status(self):
            pass

    class Session:
        def __init__(self):
            self.calls = 0

        def request(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    session = Session()
    client = FundamentalClient(
        cache=RawFundamentalCache(tmp_path),
        twse_base_url="https://example.test",
        retry=0,
        session=session,
    )
    assert client._request("GET", "https://example.test/source") == '{"ok": true}'
    assert client._request("GET", "https://example.test/source") == '{"ok": true}'
    assert session.calls == 1


def test_response_classification_rejects_http_200_security_page_and_records_metadata(
    tmp_path,
):
    assert (
        classify_response(
            200,
            "FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED.",
            expects_html=True,
        )
        == "SECURITY_BLOCK"
    )

    class Response:
        status_code = 200
        encoding = "utf-8"
        content = b"<html><table><tr><td>ok</td></tr></table></html>"

        def raise_for_status(self):
            pass

    class Session:
        def request(self, *_args, **_kwargs):
            return Response()

    client = FundamentalClient(
        cache=RawFundamentalCache(tmp_path),
        twse_base_url="https://example.test",
        retry=0,
        session=Session(),
    )
    client._request("GET", "https://example.test/source")
    metadata = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert '"classification": "SUCCESS"' in metadata
    assert '"content_hash"' in metadata


def test_valuation_parser_handles_off_market_and_fixture_json():
    assert parse_valuation_daily('{"stat": "NOT FOUND"}', date="2023-05-14").empty
    raw = json.dumps(
        {
            "stat": "OK",
            "fields": ["ticker", "pe", "pb", "dividend yield", "report period"],
            "data": [["2330", "15.2", "3.1", "1.5", "112Q1"]],
        }
    )
    result = parse_valuation_daily(raw, date="2023-05-15")
    assert result.loc[0, ["ticker", "pe", "pb"]].tolist() == ["2330", 15.2, 3.1]


def test_html_fixture_parsers_extract_common_fields():
    statement = parse_mops_statement(
        "<table><tr><th>ticker</th><th>revenue</th><th>net income</th>"
        "<th>eps</th><th>equity</th></tr><tr><td>2330</td><td>100</td>"
        "<td>10</td><td>1.2</td><td>200</td></tr></table>",
        year=112,
        season=1,
    )
    assert statement.query("metric == 'eps'").iloc[0]["value"] == 1.2
    revenue = parse_monthly_revenue(
        "<table><tr><th>ticker</th><th>revenue</th><th>prior revenue</th></tr>"
        "<tr><td>2330</td><td>100</td><td>90</td></tr></table>",
        year=112,
        month=4,
    )
    assert revenue.loc[0, "prior_year_revenue"] == 90
    publication = parse_publication_dates(
        "<table><tr><th>資料年度</th><th>上傳日期</th></tr>"
        "<tr><td>112 年 第一季</td><td>112/05/12 13:51:34</td></tr></table>",
        ticker="2330",
        year=112,
    )
    assert publication.loc[0, "publication_date"] == pd.Timestamp("2023-05-12")


def test_publication_client_treats_official_no_data_page_as_coverage_gap(tmp_path):
    class Response:
        status_code = 200
        encoding = "utf-8"
        content = "<html><body>查無所需資料</body></html>".encode()

        def raise_for_status(self):
            pass

    class Session:
        def request(self, *_args, **_kwargs):
            return Response()

    client = FundamentalClient(
        cache=RawFundamentalCache(tmp_path),
        twse_base_url="https://example.test",
        retry=0,
        session=Session(),
    )
    assert client.publication_dates("1438", 104).empty


def test_financial_and_monthly_pit_use_next_trading_day():
    calendar = pd.DatetimeIndex(
        ["2023-05-09", "2023-05-12", "2023-05-15", "2023-05-16"]
    )
    statements = pd.DataFrame(
        {
            "ticker": ["2330", "2330"],
            "roc_year": [112, 112],
            "season": [1, 1],
            "metric": ["net_income", "equity"],
            "value": [10.0, 100.0],
        }
    )
    publications = pd.DataFrame(
        {
            "ticker": ["2330"],
            "roc_year": [112],
            "season": [1],
            "publication_date": ["2023-05-12"],
        }
    )
    financial = build_financial_pit(statements, publications, trading_days=calendar)
    assert set(financial["available_date"]) == {pd.Timestamp("2023-05-15")}
    monthly = build_monthly_revenue_pit(
        pd.DataFrame(
            {"ticker": ["2330"], "revenue_month": ["2023-04-30"], "revenue": [5.0]}
        ),
        trading_days=calendar,
    )
    assert monthly.loc[0, "available_date"] == pd.Timestamp("2023-05-12")
    assert monthly.loc[0, "pit_status"] == "PERIOD_ONLY"


def test_publication_before_trading_calendar_coverage_is_excluded_not_fabricated():
    """A 2013 publication_date must not be clamped to a 2018-starting calendar."""
    calendar = pd.DatetimeIndex(["2018-01-02", "2018-01-03"])
    statements = pd.DataFrame(
        {
            "ticker": ["2330", "2330"],
            "roc_year": [102, 102],
            "season": [1, 1],
            "metric": ["net_income", "equity"],
            "value": [10.0, 100.0],
        }
    )
    publications = pd.DataFrame(
        {
            "ticker": ["2330"],
            "roc_year": [102],
            "season": [1],
            "publication_date": ["2013-05-12"],
        }
    )
    financial = build_financial_pit(statements, publications, trading_days=calendar)
    assert financial.empty
    monthly = build_monthly_revenue_pit(
        pd.DataFrame(
            {"ticker": ["2330"], "revenue_month": ["2013-04-30"], "revenue": [5.0]}
        ),
        trading_days=calendar,
    )
    assert monthly.empty


def test_missing_financial_publication_can_only_use_explicit_degraded_policy():
    calendar = pd.DatetimeIndex(["2023-05-15", "2023-05-16"])
    statements = pd.DataFrame(
        {
            "ticker": ["2330"],
            "roc_year": [112],
            "season": [1],
            "metric": ["eps"],
            "value": [1.0],
        }
    )
    publications = pd.DataFrame(
        columns=["ticker", "roc_year", "season", "publication_date"]
    )
    assert build_financial_pit(statements, publications, trading_days=calendar).empty
    degraded = build_financial_pit(
        statements,
        publications,
        trading_days=calendar,
        missing_publication_date="legal_deadline",
    )
    assert degraded.loc[0, "pit_status"] == "PERIOD_ONLY"


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("publication_date", "2023-03-30", "ordering"),
        ("available_date", "2023-04-01", "ordering"),
    ],
)
def test_pit_validation_rejects_invalid_time_order(field, value, error):
    row = [
        "2330",
        "eps",
        "2023-03-31",
        "2023-05-12",
        "2023-05-15",
        1.0,
        "mops",
        "PUBLICATION_DATE_AWARE",
    ]
    row[PIT_COLUMNS.index(field)] = value
    with pytest.raises(ValueError, match=error):
        validate_pit_records(_pit([row]))


def test_pit_validation_rejects_duplicates_and_snapshot_matrix():
    row = [
        "2330",
        "eps",
        "2023-03-31",
        "2023-05-12",
        "2023-05-15",
        1.0,
        "mops",
        "PUBLICATION_DATE_AWARE",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_pit_records(_pit([row, row]))
    snapshot = row.copy()
    snapshot[-1] = "SNAPSHOT_ONLY"
    with pytest.raises(ValueError, match="SNAPSHOT"):
        build_fundamental_matrix(_pit([snapshot]), _universe())


def test_asof_matrix_blocks_future_records_and_carries_forward():
    pit = _pit(
        [
            [
                "2330",
                "eps",
                "2022-12-31",
                "2023-03-15",
                "2023-03-16",
                5.0,
                "mops",
                "PUBLICATION_DATE_AWARE",
            ],
            [
                "2330",
                "eps",
                "2023-03-31",
                "2023-05-12",
                "2023-05-15",
                6.0,
                "mops",
                "PUBLICATION_DATE_AWARE",
            ],
        ]
    )
    matrix = build_fundamental_matrix(pit, _universe())
    assert query_fundamentals(matrix, "2023-05-10", "2330").iloc[0]["value"] == 5.0
    assert query_fundamentals(matrix, "2023-05-15", "2330").iloc[0]["value"] == 6.0
    assert query_fundamentals(matrix, "2023-05-16", "9999").empty


def test_derived_metrics_and_coverage_use_eligible_denominator():
    pit = _pit(
        [
            [
                "2330",
                "net_income",
                "2023-03-31",
                "2023-05-12",
                "2023-05-15",
                10.0,
                "mops+doc.twse",
                "PUBLICATION_DATE_AWARE",
            ],
            [
                "2330",
                "equity",
                "2023-03-31",
                "2023-05-12",
                "2023-05-15",
                100.0,
                "mops+doc.twse",
                "PUBLICATION_DATE_AWARE",
            ],
            [
                "2330",
                "revenue",
                "2023-04-30",
                "2023-05-10",
                "2023-05-15",
                30.0,
                "mops_t21sc03",
                "PERIOD_ONLY",
            ],
        ]
    )
    derived = derive_metrics(pit)
    assert derived.query("metric == 'roe'").iloc[0]["value"] == 0.1
    coverage = build_fundamental_coverage(derived, _universe())
    assert coverage["eligible_tickers"].eq(1).all()


def test_coverage_excludes_tickers_outside_the_d2_universe():
    rows = _pit(
        [
            [
                "9999",
                "pe",
                "2023-05-15",
                "2023-05-15",
                "2023-05-15",
                10.0,
                "twse_BWIBBU_d",
                "FULL_PIT",
            ]
        ]
    )
    coverage = build_fundamental_coverage(rows, _universe())
    assert (
        coverage.loc[
            coverage["metric"].eq("valuation_pit_coverage"), "covered_tickers"
        ].item()
        == 0
    )


def test_revenue_yoy_with_zero_prior_year_base_is_excluded_not_raised():
    months = pd.date_range("2022-01-31", periods=13, freq="ME")
    rows = [
        [
            "2330",
            "revenue",
            month,
            month + pd.Timedelta(days=10),
            month + pd.Timedelta(days=12),
            0.0 if index == 0 else 100.0,
            "mops_t21sc03",
            "PERIOD_ONLY",
        ]
        for index, month in enumerate(months)
    ]
    derived = derive_metrics(_pit(rows))
    yoy = derived.query("metric == 'revenue_yoy'")
    assert len(yoy) == 0


def test_valuation_pit_is_full_pit():
    pit = build_valuation_pit(
        pd.DataFrame(
            {
                "date": ["2023-05-15"],
                "ticker": ["2330"],
                "pe": [15.0],
                "pb": [3.0],
                "dividend_yield": [1.5],
            }
        )
    )
    assert set(pit["pit_status"]) == {"FULL_PIT"}
    validate_pit_records(pit)


def test_runner_writes_artifacts_and_manifest_without_live_network(
    tmp_path, monkeypatch
):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    ohlcv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-12", "2023-05-15"]),
            "ticker": ["2330", "2330"],
        }
    )
    universe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-12", "2023-05-15"]),
            "ticker": ["2330", "2330"],
            "is_eligible": [True, True],
        }
    )
    ohlcv.to_parquet(processed / "ohlcv.parquet", index=False)
    universe.to_parquet(processed / "research_universe.parquet", index=False)
    config = {
        "twse": {"base_url": "https://example.test"},
        "fundamental": {"start_roc_year": 112, "raw_cache_dir": "data/raw/fundamental"},
        "paths": {
            "ohlcv": "data/processed/ohlcv.parquet",
            "research_universe": "data/processed/research_universe.parquet",
            "fundamental_pit": "data/processed/fundamental_pit.parquet",
            "fundamental_matrix": "data/processed/fundamental_matrix.parquet",
            "valuation_daily": "data/processed/valuation_daily.parquet",
            "fundamental_coverage": "data/processed/fundamental_coverage.parquet",
            "fundamental_coverage_report": "reports/fundamental_coverage_report.md",
            "manifest": "data/processed/manifest.json",
        },
    }
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fake_collect(*_args, **_kwargs):
        return (
            pd.DataFrame(
                {
                    "ticker": ["2330", "2330"],
                    "roc_year": [112, 112],
                    "season": [1, 1],
                    "metric": ["net_income", "equity"],
                    "value": [10.0, 100.0],
                }
            ),
            pd.DataFrame(
                {
                    "ticker": ["2330"],
                    "roc_year": [112],
                    "season": [1],
                    "publication_date": ["2023-05-12"],
                }
            ),
            pd.DataFrame(
                {"ticker": ["2330"], "revenue_month": ["2023-04-30"], "revenue": [10.0]}
            ),
            pd.DataFrame(
                {
                    "date": ["2023-05-15"],
                    "ticker": ["2330"],
                    "pe": [15.0],
                    "pb": [3.0],
                    "dividend_yield": [1.0],
                }
            ),
            [],
        )

    monkeypatch.setattr(pipeline, "collect_fundamental_data", fake_collect)
    outputs = pipeline.run_pipeline(config_path)
    assert outputs["fundamental_pit"].exists()
    manifest = (processed / "manifest.json").read_text(encoding="utf-8")
    assert "publication_date_source" in manifest
    assert "FULL_PIT" in manifest


def test_runner_does_not_overwrite_artifacts_after_source_failure(
    tmp_path, monkeypatch
):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        {"date": pd.to_datetime(["2023-05-15"]), "ticker": ["2330"]}
    ).to_parquet(processed / "ohlcv.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-15"]),
            "ticker": ["2330"],
            "is_eligible": [True],
        }
    ).to_parquet(processed / "research_universe.parquet", index=False)
    original = b"valid-existing-artifact"
    (processed / "fundamental_pit.parquet").write_bytes(original)
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "twse": {"base_url": "https://example.test"},
                "fundamental": {"start_roc_year": 112, "raw_cache_dir": "data/raw"},
                "paths": {
                    "ohlcv": "data/processed/ohlcv.parquet",
                    "research_universe": "data/processed/research_universe.parquet",
                    "fundamental_pit": "data/processed/fundamental_pit.parquet",
                    "fundamental_matrix": "data/processed/fundamental_matrix.parquet",
                    "valuation_daily": "data/processed/valuation_daily.parquet",
                    "fundamental_coverage": (
                        "data/processed/fundamental_coverage.parquet"
                    ),
                    "fundamental_coverage_report": (
                        "reports/fundamental_coverage_report.md"
                    ),
                    "manifest": "data/processed/manifest.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline,
        "collect_fundamental_data",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            ["source failed"],
        ),
    )
    with pytest.raises(RuntimeError, match="not overwritten"):
        pipeline.run_pipeline(config_path)
    assert (processed / "fundamental_pit.parquet").read_bytes() == original

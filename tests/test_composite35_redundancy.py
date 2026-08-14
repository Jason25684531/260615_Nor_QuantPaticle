import pandas as pd
import pytest

from twse_factor_lab.selection.redundancy import (
    select_composite_factors,
    validate_canonical_provenance,
)

POOL = ["f1", "f2", "f3", "f4"]
FAMILY = {"FAM_A": ["f1", "f2"], "FAM_B": ["f3"], "FAM_C": ["f4"]}


def _scoreboard(rows):
    return pd.DataFrame(rows, columns=["factor", "ir", "status"])


def _correlation(rows):
    return pd.DataFrame(
        rows, columns=["factor_a", "factor_b", "correlation", "sample_count", "status"]
    )


def _pair(factor_a, factor_b, correlation, sample_count, status):
    return {
        "factor_a": factor_a,
        "factor_b": factor_b,
        "correlation": correlation,
        "sample_count": sample_count,
        "status": status,
    }


def test_provenance_fails_for_family_factor_outside_pool():
    board = _scoreboard([{"factor": f, "ir": 0.1, "status": "CANDIDATE"} for f in POOL])
    correlation = _correlation([_pair("f1", "f2", 0.1, 100, "KNOWN")])
    bad_family = dict(FAMILY, FAM_D=["outside_pool_factor"])
    with pytest.raises(ValueError, match="outside the canonical D3 pool"):
        validate_canonical_provenance(
            scoreboard=board,
            correlation=correlation,
            canonical_pool=POOL,
            family_map=bad_family,
        )


def test_provenance_fails_when_scoreboard_missing_pool_factor():
    board = _scoreboard([{"factor": "f1", "ir": 0.1, "status": "CANDIDATE"}])
    correlation = _correlation([_pair("f1", "f2", 0.1, 100, "KNOWN")])
    with pytest.raises(ValueError, match="does not cover canonical pool factors"):
        validate_canonical_provenance(
            scoreboard=board,
            correlation=correlation,
            canonical_pool=POOL,
            family_map=FAMILY,
        )


def test_provenance_fails_when_correlation_artifact_missing():
    board = _scoreboard([{"factor": f, "ir": 0.1, "status": "CANDIDATE"} for f in POOL])
    with pytest.raises(ValueError, match="factor_correlation.parquet is required"):
        validate_canonical_provenance(
            scoreboard=board, correlation=None, canonical_pool=POOL, family_map=FAMILY
        )


def test_family_cap_prefers_candidate_highest_abs_ir_and_excludes_all_reject_family():
    board = _scoreboard(
        [
            {"factor": "f1", "ir": 0.10, "status": "CANDIDATE"},
            {"factor": "f2", "ir": 0.30, "status": "CANDIDATE"},
            {"factor": "f3", "ir": -0.05, "status": "REJECT"},
            {"factor": "f4", "ir": 0.02, "status": "WEAK"},
        ]
    )
    correlation = _correlation([])
    weights = select_composite_factors(
        scoreboard=board,
        correlation=correlation,
        canonical_pool=POOL,
        family_map=FAMILY,
        correlation_cap=0.8,
    )
    fam_a = weights[weights["family"] == "FAM_A"]
    assert fam_a[fam_a["selected"]]["factor"].tolist() == ["f2"]
    fam_b = weights[weights["family"] == "FAM_B"]
    assert not fam_b["selected"].any()
    fam_c = weights[weights["family"] == "FAM_C"]
    assert fam_c[fam_c["selected"]]["factor"].tolist() == ["f4"]

    selected_weights = weights.loc[weights["selected"], "weight"]
    assert selected_weights.sum() == pytest.approx(1.0)
    assert selected_weights.tolist() == pytest.approx([0.5, 0.5])


def test_correlation_cap_removes_lower_abs_ir_member():
    board = _scoreboard(
        [
            {"factor": "f1", "ir": 0.10, "status": "CANDIDATE"},
            {"factor": "f3", "ir": 0.50, "status": "CANDIDATE"},
        ]
    )
    family = {"FAM_A": ["f1"], "FAM_B": ["f3"]}
    correlation = _correlation([_pair("f1", "f3", 0.95, 500, "KNOWN")])
    weights = select_composite_factors(
        scoreboard=board,
        correlation=correlation,
        canonical_pool=["f1", "f3"],
        family_map=family,
        correlation_cap=0.8,
    )
    assert bool(weights.loc[weights["factor"] == "f1", "selected"].iloc[0]) is False
    assert bool(weights.loc[weights["factor"] == "f3", "selected"].iloc[0]) is True


def test_unknown_correlation_does_not_remove_either_factor():
    board = _scoreboard(
        [
            {"factor": "f1", "ir": 0.10, "status": "CANDIDATE"},
            {"factor": "f3", "ir": 0.50, "status": "CANDIDATE"},
        ]
    )
    family = {"FAM_A": ["f1"], "FAM_B": ["f3"]}
    correlation = _correlation([_pair("f1", "f3", float("nan"), 5, "UNKNOWN")])
    weights = select_composite_factors(
        scoreboard=board,
        correlation=correlation,
        canonical_pool=["f1", "f3"],
        family_map=family,
        correlation_cap=0.8,
    )
    assert weights["selected"].all()

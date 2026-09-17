"""Minimal closure-orchestrator checks."""

from pathlib import Path

from twse_factor_lab.project_closure import FINGERPRINT, verify_frozen_hashes


def test_frozen_runtime_manifest_is_intact() -> None:
    root = Path(__file__).resolve().parents[1]
    report = verify_frozen_hashes(root)
    assert report["status"] == "PASS"
    assert report["candidate_fingerprint"] == FINGERPRINT
    assert report["historical_evidence_preserved"] is True

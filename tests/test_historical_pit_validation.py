from __future__ import annotations

import pandas as pd

from twse_factor_lab.analysis.historical_evidence import pit_audit


def _record(**changes: object) -> dict[str, object]:
    row = {
        "ticker": "2330",
        "metric": "eps",
        "period_end": "2023-12-31",
        "publication_date": "2024-02-20",
        "available_date": "2024-02-21",
        "value": 1.0,
        "source": "test",
        "pit_status": "PUBLICATION_DATE_AWARE",
    }
    row.update(changes)
    return row


def test_pit_audit_passes_and_rejects_future_use() -> None:
    valid = pit_audit(pd.DataFrame([_record()]))
    assert valid["status"] == "PASS"
    observations = pd.DataFrame(
        [
            {
                "fiscal_period": "2023-12-31",
                "announcement_date": "2024-02-20",
                "effective_date": "2024-02-21",
                "used_on_trade_date": "2024-02-20",
            }
        ]
    )
    blocked = pit_audit(pd.DataFrame([_record()]), observations)
    assert blocked["status"] == "BLOCKED_PIT"
    assert blocked["future_leakage_detected"] is True


def test_pit_audit_rejects_duplicate_and_bad_ordering() -> None:
    frame = pd.DataFrame(
        [_record(), _record(), _record(publication_date="2023-01-01")]
    )
    result = pit_audit(frame)
    assert result["status"] == "BLOCKED_PIT"
    assert result["duplicate_records"] == 1
    assert result["source_ordering_failures"] > 0


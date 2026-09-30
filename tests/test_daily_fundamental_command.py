from __future__ import annotations

import json

import pandas as pd

from run_daily_fundamental_production import main as adapter_main
from twse_factor_lab.application.commands import fundamental_final as command


def test_daily_adapter_preserves_cli_write_gates(tmp_path, monkeypatch, capsys):
    calls: list[dict] = []

    class Provider:
        @classmethod
        def from_repository(cls, root):
            return cls()

    monkeypatch.setattr(command, "repository_root", lambda _: tmp_path)
    monkeypatch.setattr(command, "CanonicalFundamentalRuntimeProvider", Provider)
    monkeypatch.setattr(command, "build_current_promotion_evidence", lambda: {})
    monkeypatch.setattr(
        command,
        "run_daily_fundamental",
        lambda **kwargs: calls.append(kwargs)
        or {"status": "PASS"},
    )
    monkeypatch.setattr(
        command, "run_final_validation", lambda root: {"status": "PASS"}
    )

    calendar = tmp_path / "data/processed/ohlcv.parquet"
    calendar.parent.mkdir(parents=True)
    pd.DataFrame({"date": ["2025-06-30", "2999-01-01"]}).to_parquet(
        calendar, index=False
    )

    assert adapter_main is command.main
    assert adapter_main(["--historical-validation"]) == 0
    assert adapter_main([]) == 0
    assert command.main(["--as-of-date", "2025-06-30", "--dry-run"]) == 0
    assert command.main(
        ["--as-of-date", "2025-06-30", "--write-recommendations"]
    ) == 0
    assert [call["write_recommendations"] for call in calls] == [False, False, True]
    assert {call["as_of_date"] for call in calls} == {"2025-06-30"}
    assert all(call["observation"] is False for call in calls)
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == {"status": "PASS"}


def test_observation_command_uses_isolated_store(tmp_path, monkeypatch):
    captured = []

    class Provider:
        @classmethod
        def from_repository(cls, root):
            return cls()

    calendar = tmp_path / "data/processed/ohlcv.parquet"
    calendar.parent.mkdir(parents=True)
    pd.DataFrame({"date": ["2025-06-30"]}).to_parquet(calendar, index=False)
    monkeypatch.setattr(command, "repository_root", lambda _: tmp_path)
    monkeypatch.setattr(command, "CanonicalFundamentalRuntimeProvider", Provider)
    monkeypatch.setattr(command, "build_current_promotion_evidence", lambda: {})
    monkeypatch.setattr(
        command,
        "run_daily_fundamental",
        lambda **kwargs: captured.append(kwargs) or {"status": "OBSERVATION"},
    )

    assert command.main(
        ["--observation", "--dry-run", "--write-recommendations"]
    ) == 0
    assert command.main(["--observation", "--write-recommendations"]) == 0
    assert [call["write_recommendations"] for call in captured] == [False, True]
    assert all(call["observation"] is True for call in captured)
    assert all(call["store"].root == (
        tmp_path / "data/observation/fundamental-recommendations"
    ) for call in captured)

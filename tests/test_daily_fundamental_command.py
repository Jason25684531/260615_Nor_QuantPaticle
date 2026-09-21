from __future__ import annotations

import json

from run_daily_fundamental_production import main as adapter_main
from twse_factor_lab.application.commands import fundamental_final as command


def test_daily_adapter_preserves_cli_write_gates(tmp_path, monkeypatch, capsys):
    calls: list[bool] = []

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
        lambda **kwargs: calls.append(kwargs["write_recommendations"])
        or {"status": "PASS"},
    )
    monkeypatch.setattr(
        command, "run_final_validation", lambda root: {"status": "PASS"}
    )

    assert adapter_main is command.main
    assert adapter_main([]) == 0
    assert command.main(["--as-of-date", "2025-06-30", "--dry-run"]) == 0
    assert command.main(
        ["--as-of-date", "2025-06-30", "--write-recommendations"]
    ) == 0
    assert calls == [False, True]
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == {"status": "PASS"}

from __future__ import annotations

from pathlib import Path

STATES = {"PASS", "PARTIAL", "REJECT", "NOT_APPLICABLE"}
RC1_GATES = {"CI", "README", "clean-install", "historical OpenSpec sync", "all-strict"}


def evaluate(d4_verdict: str, artifacts: dict[str, Path]) -> dict:
    if d4_verdict not in {"PASS", "CONDITIONAL", "REJECT"}:
        raise ValueError("invalid D4 verdict")
    gates = []
    for name, path in artifacts.items():
        gates.append(
            {
                "gate": name,
                "evidence": str(path),
                "status": "PASS" if path.exists() and path.stat().st_size else "REJECT",
                "artifact": str(path),
                "limitation": "",
            }
        )
    for name in RC1_GATES:
        gates.append(
            {
                "gate": name,
                "evidence": "RC1 closure",
                "status": "PARTIAL",
                "artifact": "",
                "limitation": "pending RC1",
            }
        )
    platform = (
        "FAIL"
        if any(g["status"] == "REJECT" for g in gates)
        else ("PARTIAL" if any(g["status"] == "PARTIAL" for g in gates) else "PASS")
    )
    return {
        "gates": gates,
        "strategy_acceptance": "REJECTED" if d4_verdict == "REJECT" else "CONDITIONAL",
        "research_platform_acceptance": platform,
    }


def write_matrix(root: str | Path, result: dict) -> Path:
    path = Path(root) / "reports/final/final_acceptance_matrix.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Acceptance Matrix",
        "",
        f"Strategy Acceptance: **{result['strategy_acceptance']}**",
        f"Research Platform Acceptance: **{result['research_platform_acceptance']}**",
        "",
        "| gate | evidence | status | artifact | limitation |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {g['gate']} | {g['evidence']} | {g['status']} | "
        f"{g['artifact']} | {g['limitation']} |"
        for g in result["gates"]
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

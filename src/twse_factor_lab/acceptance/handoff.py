from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def write_handoff(root: str | Path, payload: dict, files: list[Path]) -> Path:
    root = Path(root)
    payload["git_commit"] = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "uncommitted"
    )
    payload["artifact_hashes"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.exists()
    }
    path = root / "final_acceptance_handoff.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path

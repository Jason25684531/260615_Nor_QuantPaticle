"""Small application-layer helpers shared by command entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repository_root(start: str | Path | None = None) -> Path:
    """Find the checkout root from a file/directory path."""

    path = Path(start or __file__).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration without changing caller-relative semantics."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_root_for_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    return path.parent.parent if path.parent.name == "config" else path.parent


def resolve_path(config_path: str | Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root_for_config(config_path) / path


__all__ = ["load_config", "project_root_for_config", "repository_root", "resolve_path"]

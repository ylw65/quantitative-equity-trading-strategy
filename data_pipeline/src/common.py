"""Shared configuration, logging, paths, and atomic I/O helpers."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "acquisition.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML project configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_parent(path: str | Path) -> Path:
    """Create and return a file's parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_json(payload: Any, path: str | Path) -> None:
    """Atomically write JSON so interrupted runs do not leave partial files."""
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def configure_logging(log_file: str | Path | None = None) -> logging.Logger:
    """Configure consistent console and optional file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        target = ensure_parent(log_file)
        handlers.append(logging.FileHandler(target, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("us_equity_data")

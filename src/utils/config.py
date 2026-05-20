"""YAML configuration loading."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "config/default.yaml") -> Dict[str, Any]:
    """Load a YAML config file, returning a plain dict.

    Path is resolved relative to the current working directory.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """Return the project root directory (two levels above this file)."""
    return Path(__file__).resolve().parents[2]

"""Configuration loading and path resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
_CONFIG_FILE = Path(os.environ.get("GBMC_CONFIG", ROOT / "config.yaml"))

with open(_CONFIG_FILE, "r", encoding="utf-8") as fh:
    CFG: dict[str, Any] = yaml.safe_load(fh)

SEED: int = int(CFG.get("seed", 42))


def path(key: str) -> Path:
    """Absolute path for a key inside the ``paths`` section of config.yaml."""
    raw = CFG["paths"][key]
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


def out_dir(*parts: str) -> Path:
    """Create (if needed) and return a directory under the repo root."""
    d = ROOT.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(*parts: str) -> Path:
    return out_dir("results", *parts)

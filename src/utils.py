"""Small shared helpers: seeding, logging, IO, filename humanisation."""
from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = module.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


# --------------------------------------------------------------------------- #
# logging / io
# --------------------------------------------------------------------------- #


def get_logger(name: str = "gbmc") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOG = get_logger()


def save_json(obj: Any, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return dest


def load_json(src: str | Path) -> Any:
    with open(src, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class Timer:
    def __init__(self, label: str = "elapsed"):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        LOG.info("%s: %.1f s", self.label, time.time() - self.t0)


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def humanize(token: str) -> str:
    """`j_s__bach_solo_cantatas` -> `j s bach solo cantatas`."""
    s = token.replace("_", " ").replace("-", " ")
    return _WS.sub(" ", s).strip()


def clean_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = re.sub(r"<[^>]+>", " ", str(s))          # strip html from FMA fields
    return _WS.sub(" ", s).strip()


def join_labels(labels: Iterable[str]) -> str:
    return "|".join(labels)


def split_labels(s: Any) -> list[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return []
    return [t for t in str(s).split("|") if t]

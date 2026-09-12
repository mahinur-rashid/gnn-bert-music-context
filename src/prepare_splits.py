"""Train / val / test splits, written to ``data/splits/<dataset>.json``.

Rules (Section 3, "Splits"):
  * FMA and MagnaTagATune ship **official** splits -- those are used verbatim.
  * Everything else gets a seeded 80/10/10 split that is **grouped by artist**
    so no artist appears in more than one split (no artist leakage).
  * GTZAN has no artist metadata, so it is split stratified by genre.

CLI::

    python -m src.prepare_splits --dataset all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SEED, out_dir, path
from .utils import LOG, load_json, save_json

TEXT_DATASETS = ["musiccaps", "mtat", "fma_small", "fma_medium", "deam"]
ALL_DATASETS = TEXT_DATASETS + ["gtzan"]


# --------------------------------------------------------------------------- #
# generic splitters
# --------------------------------------------------------------------------- #


def grouped_split(ids: list[str], groups: list[str], ratios=(0.8, 0.1, 0.1),
                  seed: int = SEED) -> dict[str, list[str]]:
    """Assign whole groups (artists) to splits until each quota is filled."""
    rng = np.random.RandomState(seed)
    by_group: dict[str, list[str]] = {}
    for i, g in zip(ids, groups):
        by_group.setdefault(str(g), []).append(str(i))

    keys = sorted(by_group)
    rng.shuffle(keys)
    n = len(ids)
    quota = [int(round(r * n)) for r in ratios]
    buckets: list[list[str]] = [[], [], []]
    for k in keys:
        # place the group in whichever split is furthest below its quota
        deficits = [quota[j] - len(buckets[j]) for j in range(3)]
        j = int(np.argmax(deficits))
        buckets[j].extend(by_group[k])
    return {"train": buckets[0], "val": buckets[1], "test": buckets[2]}


def stratified_split(ids: list[str], labels: list[str], ratios=(0.8, 0.1, 0.1),
                     seed: int = SEED) -> dict[str, list[str]]:
    rng = np.random.RandomState(seed)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    df = pd.DataFrame({"id": [str(i) for i in ids], "y": labels})
    for _, sub in df.groupby("y"):
        idx = sub["id"].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        out["train"] += idx[:n_tr].tolist()
        out["val"] += idx[n_tr:n_tr + n_va].tolist()
        out["test"] += idx[n_tr + n_va:].tolist()
    return out


def official_split(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        s: df.loc[df["split"] == s, "id"].astype(str).tolist()
        for s in ("train", "val", "test")
    }


# --------------------------------------------------------------------------- #
# per-dataset entry points
# --------------------------------------------------------------------------- #


def has_official_split(df: pd.DataFrame) -> bool:
    if "split" not in df.columns:
        return False
    col = df["split"].fillna("").astype(str).str.strip()
    return bool(col.isin({"train", "val", "test"}).all())


def split_text_dataset(tag: str, seed: int = SEED,
                       force_artist_safe: bool = False) -> dict[str, list[str]]:
    from .text_data import load

    df, _ = load(tag)
    df["split"] = df["split"].fillna("").astype(str).str.strip()
    if has_official_split(df) and not force_artist_safe:
        sp = official_split(df)
        LOG.info("%s: using OFFICIAL split", tag)
    else:
        sp = grouped_split(df["id"].tolist(), df["group"].astype(str).tolist(), seed=seed)
        LOG.info("%s: using grouped (artist-safe) split", tag)
    return sp


def split_musiccaps(seed: int = SEED) -> dict[str, list[str]]:
    """Split over *every* captioned clip, not only those kept by the tag table,
    so Task 4 can use all downloaded audio."""
    from .text_data import load_musiccaps_captions

    caps = load_musiccaps_captions()
    ids = caps["id"].astype(str).tolist()
    LOG.info("musiccaps: %d captioned clips", len(ids))
    return grouped_split(ids, ids, seed=seed)


def split_gtzan(seed: int = SEED) -> dict[str, list[str]]:
    root = path("gtzan_audio")
    files = sorted(root.rglob("*.wav"))
    ids = [f"{f.parent.name}/{f.stem}" for f in files]
    genres = [f.parent.name for f in files]
    LOG.info("gtzan: %d clips, %d genres", len(ids), len(set(genres)))
    return stratified_split(ids, genres, seed=seed)


def build(dataset: str, seed: int = SEED,
          force_artist_safe: bool = False) -> dict[str, list[str]]:
    if dataset == "gtzan":
        return split_gtzan(seed)
    if dataset == "musiccaps":
        return split_musiccaps(seed)
    return split_text_dataset(dataset, seed, force_artist_safe)


def save(split: dict[str, list[str]], dataset: str) -> Path:
    dest = out_dir("data", "splits") / f"{dataset}.json"
    save_json(split, dest)
    LOG.info("%s split -> train %d / val %d / test %d  (%s)",
             dataset, len(split["train"]), len(split["val"]), len(split["test"]), dest)
    return dest


def load(dataset: str) -> dict[str, list[str]]:
    return load_json(out_dir("data", "splits") / f"{dataset}.json")


def split_index(dataset: str) -> dict[str, str]:
    """id -> split-name lookup."""
    sp = load(dataset)
    return {i: name for name, ids in sp.items() for i in ids}


def check_leakage(dataset: str) -> dict[str, int]:
    """Count artists shared between splits (should be 0 for grouped splits)."""
    from .text_data import load as load_text

    try:
        df, _ = load_text(dataset)
    except FileNotFoundError:
        return {}
    sp = load(dataset)
    idx = {i: name for name, ids in sp.items() for i in ids}
    df = df[df["id"].isin(idx)]
    groups = df.assign(split=[idx[i] for i in df["id"]]).groupby("group")["split"].nunique()
    n_shared = int((groups > 1).sum())
    LOG.info("%s: %d/%d artists appear in more than one split", dataset, n_shared, len(groups))
    return {"artists": int(len(groups)), "artists_in_multiple_splits": n_shared}


def main() -> None:
    ap = argparse.ArgumentParser(description="build train/val/test splits")
    ap.add_argument("--dataset", default="all", choices=ALL_DATASETS + ["all"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--artist_safe", action="store_true",
                    help="ignore official splits and group by artist instead")
    args = ap.parse_args()

    targets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    report = {}
    for ds in targets:
        try:
            sp = build(ds, args.seed, args.artist_safe)
        except FileNotFoundError as e:
            LOG.warning("skipping %s (%s)", ds, e)
            continue
        save(sp, ds)
        if ds != "gtzan":
            report[ds] = check_leakage(ds)
    if report:
        save_json(report, out_dir("data", "splits") / "leakage_report.json")


if __name__ == "__main__":
    main()

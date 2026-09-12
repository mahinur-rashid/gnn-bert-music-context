"""Music-structure graph construction (Section 3, step 3).

Two edge families are built per track and merged into a single graph whose
nodes are **audio segments**:

* *segment graph*  -- temporal adjacency (i -> i+1 .. i+r) plus cosine
  similarity edges between segments with sim > tau (k-NN limited),
* *chord-transition graph* -- edges between segments whose estimated chords
  form an observed transition, weighted by the transition count.

Edge attributes are ``[is_temporal, cosine_similarity, chord_transition_weight]``.

CLI::

    python -m src.graph_builder --dataset gtzan --jobs 8
    python -m src.graph_builder --dataset fma_small --jobs 8
    python -m src.graph_builder --dataset mtat --jobs 8 --no_mel
    python -m src.graph_builder --dataset deam --jobs 8
    python -m src.graph_builder --dataset deam_feats          # no audio decoding
"""
from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from .audio_features import (CHORD_NAMES, FEATURE_DIM, N_CHORDS, audio_params,
                             extract_track)
from .config import CFG, out_dir, path
from .utils import LOG, save_json

AUDIO_DATASETS = ["gtzan", "fma_small", "fma_medium", "mtat", "deam", "musiccaps"]
DATASETS = AUDIO_DATASETS + ["deam_feats"]
SHARD_SIZE = 2000


# --------------------------------------------------------------------------- #
# edge construction
# --------------------------------------------------------------------------- #


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Z = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return np.clip(Z @ Z.T, -1.0, 1.0)


def build_edges(
    X: np.ndarray,
    chords: np.ndarray | None = None,
    knn: int | None = None,
    tau: float | None = None,
    temporal_radius: int | None = None,
    chord_edges: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """-> (edge_index [2, E], edge_attr [E, 3]); the graph is undirected."""
    g = CFG["graph"]
    knn = int(g["knn"]) if knn is None else knn
    tau = float(g["sim_threshold"]) if tau is None else tau
    radius = int(g["temporal_radius"]) if temporal_radius is None else temporal_radius
    use_chords = bool(g["chord_edges"]) if chord_edges is None else chord_edges

    S = len(X)
    attrs: dict[tuple[int, int], list[float]] = {}

    def add(i: int, j: int, slot: int, value: float) -> None:
        if i == j:
            return
        for a, b in ((i, j), (j, i)):
            e = attrs.setdefault((a, b), [0.0, 0.0, 0.0])
            e[slot] = max(e[slot], value)

    # 1. temporal adjacency
    for i in range(S):
        for d in range(1, radius + 1):
            if i + d < S:
                add(i, i + d, 0, 1.0 / d)

    # 2. similarity edges (top-k above tau)
    if S > 1:
        sim = cosine_matrix(X)
        np.fill_diagonal(sim, -np.inf)
        k = min(knn, S - 1)
        for i in range(S):
            for j in np.argpartition(-sim[i], k - 1)[:k]:
                if sim[i, j] > tau:
                    add(i, int(j), 1, float(sim[i, j]))

    # 3. chord-transition induced edges
    if use_chords and chords is not None and S > 1:
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(chords[:-1], chords[1:]):
            if a != b:
                counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
        if counts:
            mx = max(counts.values())
            pos: dict[int, list[int]] = {}
            for i, c in enumerate(chords):
                pos.setdefault(int(c), []).append(i)
            for (ca, cb), n in counts.items():
                for i in pos.get(ca, [])[:4]:
                    for j in pos.get(cb, [])[:4]:
                        add(i, j, 2, n / mx)

    if not attrs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    keys = sorted(attrs)
    edge_index = np.asarray(keys, dtype=np.int64).T
    edge_attr = np.asarray([attrs[k] for k in keys], dtype=np.float32)
    return edge_index, edge_attr


def build_track_graph(x: np.ndarray, chords: np.ndarray, **kw) -> dict:
    edge_index, edge_attr = build_edges(x, chords, **kw)
    return {
        "x": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        "edge_index": torch.from_numpy(edge_index),
        "edge_attr": torch.from_numpy(edge_attr),
        "chords": torch.from_numpy(np.asarray(chords, dtype=np.int64)),
        "num_nodes": int(len(x)),
    }


def to_pyg(record: dict):
    """dict-of-tensors -> ``torch_geometric.data.Data``."""
    from torch_geometric.data import Data

    d = Data(
        x=record["x"],
        edge_index=record["edge_index"],
        edge_attr=record["edge_attr"],
    )
    d.chords = record["chords"]
    d.num_nodes = int(record["num_nodes"])
    d.track_id = str(record["id"])
    return d


# --------------------------------------------------------------------------- #
# dataset file listings
# --------------------------------------------------------------------------- #


def fma_audio_path(track_id: int | str, subset: str) -> Path:
    tid = f"{int(track_id):06d}"
    root = path("fma_small" if subset == "small" else "fma_medium")
    return root / tid[:3] / f"{tid}.mp3"


def list_items(dataset: str) -> list[tuple[str, str]]:
    """-> [(track_id, audio_path)]"""
    if dataset == "gtzan":
        files = sorted(path("gtzan_audio").rglob("*.wav"))
        return [(f"{f.parent.name}/{f.stem}", str(f)) for f in files]

    if dataset in ("fma_small", "fma_medium"):
        from .text_data import load as load_text

        subset = dataset.split("_")[1]
        df, _ = load_text(dataset)
        items = []
        for tid in df["id"]:
            p = fma_audio_path(tid, subset)
            if p.exists():
                items.append((str(tid), str(p)))
        return items

    if dataset == "mtat":
        from .text_data import load as load_text

        df, _ = load_text("mtat")
        root = path("mtat_audio")
        items = []
        for tid, rel in zip(df["id"], df["mp3_path"]):
            p = root / str(rel)
            if p.exists() and p.stat().st_size > 1024:   # 3 clips are empty
                items.append((str(tid), str(p)))
        return items

    if dataset == "deam":
        from .text_data import load as load_text

        df, _ = load_text("deam")
        root = path("deam_audio")
        items = []
        for tid in df["id"]:
            p = root / f"{tid}.mp3"
            if p.exists():
                items.append((str(tid), str(p)))
        return items

    if dataset == "musiccaps":
        return list_musiccaps_items()

    if dataset == "deam_feats":
        from .text_data import load as load_text

        df, _ = load_text("deam")
        root = path("deam_features")
        items = []
        for tid in df["id"]:
            p = root / f"{tid}.csv"
            if p.exists():
                items.append((str(tid), str(p)))
        return items

    raise ValueError(f"unknown dataset {dataset!r}")


MUSICCAPS_FILE_RE = re.compile(r"^\[(?P<ytid>.+)\]-\[(?P<start>\d+)-(?P<end>\d+)\]$")


def list_musiccaps_items() -> list[tuple[str, str]]:
    """MusicCaps clips are named ``[<ytid>]-[<start>-<end>].wav``.

    Only ~5.2k of the 5'521 captioned clips are downloadable from YouTube, so
    the intersection with the caption table is taken here.
    """
    import pandas as pd

    root = path("musiccaps_audio")
    if not root.exists():
        raise FileNotFoundError(
            f"MusicCaps audio not found at {root}. Download the clips first, or "
            "point paths.musiccaps_audio in config.yaml at the wav directory."
        )
    # Intersect with the *raw* caption CSV rather than the processed tag table:
    # Task 4 only needs a caption, not a top-50 aspect label.
    wanted = set(pd.read_csv(path("musiccaps_csv"))["ytid"].astype(str))

    items, skipped = [], 0
    for f in sorted(root.glob("*.wav")):
        m = MUSICCAPS_FILE_RE.match(f.stem)
        if not m:
            skipped += 1
            continue
        ytid = m.group("ytid")
        if wanted is not None and ytid not in wanted:
            skipped += 1
            continue
        items.append((ytid, str(f)))
    LOG.info("musiccaps: %d clips with audio (%d skipped)", len(items), skipped)
    return items


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #

_WORKER_OPTS: dict = {}


def _init_worker(opts: dict) -> None:
    global _WORKER_OPTS
    _WORKER_OPTS = opts
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def process_audio_item(item: tuple[str, str]) -> dict | None:
    track_id, fp = item
    o = _WORKER_OPTS
    try:
        feat = extract_track(fp, segmentation=o.get("segmentation", "fixed"),
                             want_mel=o.get("want_mel", True),
                             max_dur_s=o.get("max_dur_s"),
                             win_s=o.get("win_s"), hop_s=o.get("hop_s"),
                             max_segments=o.get("max_segments"))
        rec = build_track_graph(
            feat["x"], feat["chords"],
            knn=o.get("knn"), tau=o.get("tau"),
            temporal_radius=o.get("temporal_radius"), chord_edges=o.get("chord_edges"),
        )
        rec["id"] = track_id
        rec["bounds_s"] = torch.from_numpy(feat["bounds_s"])
        rec["duration_s"] = float(feat["duration_s"])
        if o.get("want_mel", True):
            rec["mel"] = torch.from_numpy(feat["mel"].astype(np.float16))
        return rec
    except Exception as exc:  # pragma: no cover -- corrupt files
        return {"id": track_id, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=1)}


# --- DEAM: graphs from the shipped openSMILE features (no audio decoding) --- #

DEAM_FEAT_SEGMENTS = 24


def process_deam_feature_item(item: tuple[str, str]) -> dict | None:
    import pandas as pd

    track_id, fp = item
    try:
        df = pd.read_csv(fp, sep=";")
        df = df.drop(columns=[c for c in df.columns if "frameTime" in c], errors="ignore")
        A = df.to_numpy(dtype=np.float32)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(A)
        if n < 4:
            raise ValueError("too few frames")
        edges = np.linspace(0, n, DEAM_FEAT_SEGMENTS + 1).astype(int)
        X = np.stack([A[edges[i]:max(edges[i] + 1, edges[i + 1])].mean(0)
                      for i in range(DEAM_FEAT_SEGMENTS)]).astype(np.float32)
        chords = np.zeros(len(X), dtype=np.int64)     # no chroma in openSMILE
        rec = build_track_graph(X, chords, chord_edges=False)
        rec["id"] = track_id
        rec["duration_s"] = float(n * 0.5)
        rec["bounds_s"] = torch.from_numpy(
            np.stack([edges[:-1], edges[1:]], 1).astype(np.float32) * 0.5
        )
        return rec
    except Exception as exc:  # pragma: no cover
        return {"id": track_id, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def graph_dir(dataset: str) -> Path:
    return out_dir("data", "processed", "graphs", dataset)


def run(dataset: str, jobs: int = 8, limit: int | None = None, want_mel: bool = True,
        segmentation: str = "fixed", overwrite: bool = False, n_samples: int = 20) -> dict:
    items = list_items(dataset)
    if limit:
        items = items[:limit]
    dest = graph_dir(dataset)
    if overwrite:
        for f in dest.glob("shard_*.pt"):
            f.unlink()
    LOG.info("%s: building graphs for %d tracks -> %s", dataset, len(items), dest)

    a = audio_params(dataset)
    opts = {
        "want_mel": want_mel and dataset != "deam_feats",
        "segmentation": segmentation,
        "knn": int(CFG["graph"]["knn"]),
        "tau": float(CFG["graph"]["sim_threshold"]),
        "temporal_radius": int(CFG["graph"]["temporal_radius"]),
        "chord_edges": bool(CFG["graph"]["chord_edges"]),
        "max_dur_s": float(a["max_dur_s"]),
        "win_s": float(a["win_s"]),
        "hop_s": float(a["hop_s"]),
        "max_segments": int(a["max_segments"]),
    }
    worker = process_deam_feature_item if dataset == "deam_feats" else process_audio_item

    records: list[dict] = []
    failures: list[dict] = []
    n_shards = 0

    def flush(force: bool = False) -> None:
        nonlocal records, n_shards
        while len(records) >= SHARD_SIZE or (force and records):
            chunk, records = records[:SHARD_SIZE], records[SHARD_SIZE:]
            torch.save(chunk, dest / f"shard_{n_shards:03d}.pt")
            LOG.info("  wrote shard_%03d.pt (%d graphs)", n_shards, len(chunk))
            n_shards += 1

    done = 0
    if jobs <= 1:
        _init_worker(opts)
        for it in items:
            r = worker(it)
            done += 1
            (failures if r is None or "error" in r else records).append(r or {"id": it[0]})
            if done % 200 == 0:
                LOG.info("  %d/%d", done, len(items))
            flush()
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(opts,)) as pool:
            futures = {pool.submit(worker, it): it for it in items}
            for fut in as_completed(futures):
                r = fut.result()
                done += 1
                (failures if r is None or "error" in r else records).append(
                    r or {"id": futures[fut][0]})
                if done % 500 == 0:
                    LOG.info("  %d/%d (%d failed)", done, len(items), len(failures))
                flush()
    flush(force=True)

    n_ok = sum(1 for f in dest.glob("shard_*.pt") for _ in [0])
    meta = {
        "dataset": dataset,
        "n_requested": len(items),
        "n_failed": len(failures),
        "n_shards": n_shards,
        "feature_dim": FEATURE_DIM if dataset != "deam_feats" else None,
        "n_chords": N_CHORDS,
        "chord_names": CHORD_NAMES,
        "has_mel": opts["want_mel"],
        "graph_cfg": CFG["graph"],
        "audio_cfg": a,
        "segmentation": segmentation,
    }
    save_json(meta, dest / "meta.json")
    if failures:
        save_json(failures[:200], dest / "failures.json")
        LOG.warning("%s: %d tracks failed (see failures.json)", dataset, len(failures))
    export_samples(dataset, n=n_samples)
    LOG.info("%s: done -- %d shards, %d failures", dataset, n_shards, len(failures))
    return meta


# --------------------------------------------------------------------------- #
# loading + sample export
# --------------------------------------------------------------------------- #


def load_graphs(dataset: str, ids: set[str] | None = None,
                want_mel: bool = False) -> dict[str, dict]:
    """Read every shard back into ``{track_id: record}``."""
    dest = graph_dir(dataset)
    shards = sorted(dest.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(
            f"no graphs for {dataset!r}; run: python -m src.graph_builder --dataset {dataset}"
        )
    out: dict[str, dict] = {}
    for sh in shards:
        for rec in torch.load(sh, weights_only=True):
            tid = str(rec["id"])
            if ids is not None and tid not in ids:
                continue
            if not want_mel:
                rec.pop("mel", None)
            out[tid] = rec
    LOG.info("%s: loaded %d graphs from %d shards", dataset, len(out), len(shards))
    return out


def export_samples(dataset: str, n: int = 20) -> Path:
    """Deliverable #2: >= 20 example graphs as .pt and human-readable .json."""
    dest = out_dir("data", "processed", "graph_samples", dataset)
    shards = sorted(graph_dir(dataset).glob("shard_*.pt"))
    if not shards:
        return dest
    recs = torch.load(shards[0], weights_only=True)[:n]
    for rec in recs:
        tid = str(rec["id"]).replace("/", "_")
        torch.save(rec, dest / f"{tid}.pt")
        j = {
            "track_id": str(rec["id"]),
            "num_nodes": int(rec["num_nodes"]),
            "num_edges": int(rec["edge_index"].shape[1]),
            "feature_dim": int(rec["x"].shape[1]),
            "chords": [CHORD_NAMES[int(c)] for c in rec["chords"]],
            "edge_index": rec["edge_index"].tolist(),
            "edge_attr_legend": ["is_temporal", "cosine_similarity", "chord_transition"],
            "edge_attr": [[round(v, 4) for v in e] for e in rec["edge_attr"].tolist()],
            "segment_bounds_s": rec.get("bounds_s", torch.zeros(0)).tolist(),
        }
        (dest / f"{tid}.json").write_text(json.dumps(j, indent=1), encoding="utf-8")
    LOG.info("exported %d sample graphs -> %s", len(recs), dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="build music-structure graphs")
    ap.add_argument("--dataset", default="gtzan", choices=DATASETS + ["all"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_mel", action="store_true", help="skip mel patches (saves disk)")
    ap.add_argument("--segmentation", default="fixed", choices=["fixed", "beat"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    targets = AUDIO_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        run(ds, jobs=args.jobs, limit=args.limit, want_mel=not args.no_mel,
            segmentation=args.segmentation, overwrite=args.overwrite,
            n_samples=args.samples)


if __name__ == "__main__":
    main()

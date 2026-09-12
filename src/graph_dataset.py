"""Assemble PyG datasets: graphs + labels + (optionally) tokenised text.

Used by Task 2 (graph only) and Task 3 (graph + text fusion).

Node features are standardised with statistics computed on the **training
split only** (the chord one-hot block is left untouched).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from . import prepare_splits, text_data
from .audio_features import CONT_DIM
from .config import path
from .graph_builder import load_graphs
from .utils import LOG, split_labels

GRAPH_DATASETS = ["gtzan", "fma_small", "fma_medium", "mtat", "deam", "deam_feats"]


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def gtzan_table() -> pd.DataFrame:
    files = sorted(path("gtzan_audio").rglob("*.wav"))
    return pd.DataFrame({
        "id": [f"{f.parent.name}/{f.stem}" for f in files],
        "labels": [f.parent.name for f in files],
        "text": [f"Track: {f.stem}. Genre excerpt from the GTZAN collection." for f in files],
        "group": [f.parent.name for f in files],
    })


def text_table(dataset: str) -> tuple[pd.DataFrame, list[str]]:
    """Label/text table for a graph dataset (``deam_feats`` reuses ``deam``)."""
    if dataset == "gtzan":
        t = gtzan_table()
        return t, sorted(t["labels"].unique().tolist())
    tag = "deam" if dataset == "deam_feats" else dataset
    return text_data.load(tag)


def split_name(dataset: str) -> str:
    return "deam" if dataset == "deam_feats" else dataset


@dataclass
class LabelSpec:
    classes: list[str]
    multilabel: bool
    Y: np.ndarray                      # [N, K]
    ids: list[str]
    va: np.ndarray | None = None       # [N, 2] valence/arousal, DEAM only
    meta: dict = field(default_factory=dict)


def build_labels(dataset: str, label_mode: str = "auto") -> tuple[pd.DataFrame, LabelSpec]:
    df, vocab = text_table(dataset)
    df = df.copy()
    df["id"] = df["id"].astype(str)

    if dataset == "gtzan":
        classes = sorted(df["labels"].unique().tolist())
        idx = {c: i for i, c in enumerate(classes)}
        Y = np.array([idx[v] for v in df["labels"]], dtype=np.int64)
        multilabel = False
    elif dataset in ("fma_small", "fma_medium") and label_mode in ("auto", "single"):
        # single-label genre_top -- the Task 2 formulation
        df = df[df["genre_top"].astype(str).str.len() > 0]
        classes = sorted(df["genre_top"].unique().tolist())
        idx = {c: i for i, c in enumerate(classes)}
        Y = np.array([idx[v] for v in df["genre_top"]], dtype=np.int64)
        multilabel = False
    else:
        classes = list(vocab)
        Y = text_data.labels_to_matrix(df["labels"], classes)
        multilabel = True

    if label_mode == "multi" and not multilabel:
        classes = list(vocab)
        Y = text_data.labels_to_matrix(df["labels"], classes)
        multilabel = True

    va = None
    if {"valence", "arousal"}.issubset(df.columns):
        va = df[["valence", "arousal"]].to_numpy(dtype=np.float32)

    spec = LabelSpec(classes=classes, multilabel=multilabel, Y=Y,
                     ids=df["id"].tolist(), va=va,
                     meta={"n_classes": len(classes)})
    return df, spec


# --------------------------------------------------------------------------- #
# feature standardisation
# --------------------------------------------------------------------------- #


def fit_feature_stats(records: Sequence[dict], cont_dim: int | None = None):
    X = torch.cat([r["x"] for r in records], dim=0)
    cont_dim = X.shape[1] if cont_dim is None else min(cont_dim, X.shape[1])
    mu = X[:, :cont_dim].mean(0)
    sd = X[:, :cont_dim].std(0).clamp_min(1e-6)
    return mu, sd, cont_dim


def apply_feature_stats(x: torch.Tensor, mu, sd, cont_dim: int) -> torch.Tensor:
    x = x.clone()
    x[:, :cont_dim] = (x[:, :cont_dim] - mu) / sd
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# dataset assembly
# --------------------------------------------------------------------------- #


def build_pyg_splits(
    dataset: str,
    label_mode: str = "auto",
    with_text: bool = False,
    tokenizer=None,
    max_length: int = 192,
    want_mel: bool = False,
    limit: int | None = None,
    standardize: bool = True,
) -> dict:
    """-> {'splits': {'train': [Data], ...}, 'spec': LabelSpec, ...}"""
    from torch_geometric.data import Data

    df, spec = build_labels(dataset, label_mode)
    splits = prepare_splits.load(split_name(dataset))
    wanted = {str(i) for ids in splits.values() for i in ids} & set(spec.ids)
    graphs = load_graphs(dataset, ids=wanted, want_mel=want_mel)

    keep = [i for i in spec.ids if i in graphs]
    if limit:
        keep = keep[:limit]
    pos = {i: k for k, i in enumerate(spec.ids)}
    LOG.info("%s: %d tracks with both graph and label", dataset, len(keep))

    texts = dict(zip(df["id"].astype(str), df["text"].astype(str))) if with_text else {}
    enc_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if with_text:
        if tokenizer is None:
            raise ValueError("with_text=True requires a tokenizer")
        ids_list = keep
        batch = tokenizer([texts.get(i, "") for i in ids_list], truncation=True,
                          padding="max_length", max_length=max_length,
                          return_tensors="pt")
        for n, i in enumerate(ids_list):
            enc_cache[i] = (batch["input_ids"][n], batch["attention_mask"][n])

    split_of = {i: name for name, ids in splits.items() for i in map(str, ids)}
    train_records = [graphs[i] for i in keep if split_of.get(i) == "train"]
    if not train_records:
        raise RuntimeError(f"{dataset}: empty training split")
    mu, sd, cont_dim = fit_feature_stats(train_records,
                                         CONT_DIM if dataset != "deam_feats" else None)

    out: dict[str, list] = {"train": [], "val": [], "test": []}
    for tid in keep:
        name = split_of.get(tid)
        if name not in out:
            continue
        rec = graphs[tid]
        x = apply_feature_stats(rec["x"], mu, sd, cont_dim) if standardize else rec["x"]
        d = Data(x=x, edge_index=rec["edge_index"], edge_attr=rec["edge_attr"])
        d.num_nodes = int(rec["num_nodes"])
        d.chords = rec["chords"]
        d.track_id = tid
        k = pos[tid]
        if spec.multilabel:
            d.y = torch.from_numpy(spec.Y[k]).float().unsqueeze(0)      # [1, K]
        else:
            d.y = torch.tensor([int(spec.Y[k])], dtype=torch.long)      # [1]
        if spec.va is not None:
            d.va = torch.from_numpy(spec.va[k]).float().unsqueeze(0)    # [1, 2]
        if want_mel and "mel" in rec:
            d.mel = rec["mel"].float().unsqueeze(0)                     # [1, n_mels, T]
        if with_text:
            ii, am = enc_cache[tid]
            d.input_ids = ii.unsqueeze(0)
            d.attention_mask = am.unsqueeze(0)
        out[name].append(d)

    LOG.info("%s: train %d / val %d / test %d | %d classes | multilabel=%s",
             dataset, len(out["train"]), len(out["val"]), len(out["test"]),
             len(spec.classes), spec.multilabel)
    return {
        "splits": out,
        "spec": spec,
        "in_dim": int(out["train"][0].x.shape[1]),
        "edge_dim": int(out["train"][0].edge_attr.shape[1]),
        "feature_stats": {"mu": mu, "sd": sd, "cont_dim": cont_dim},
        "texts": texts,
    }


def loader(data_list: list, batch_size: int, shuffle: bool = False, num_workers: int = 0):
    from torch_geometric.loader import DataLoader

    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def labels_matrix(data_list: list, multilabel: bool) -> np.ndarray:
    if multilabel:
        return torch.cat([d.y for d in data_list], 0).numpy()
    return torch.cat([d.y for d in data_list], 0).numpy()


def mean_node_features(data_list: list) -> np.ndarray:
    """Flat per-track descriptor used by the PCA+MLP baseline (B4)."""
    return np.stack([d.x.mean(0).numpy() for d in data_list])

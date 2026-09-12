"""Evaluation metrics (Section 6 of the brief) and plotting utilities.

Also usable as a CLI to aggregate every ``results/metrics/*.json`` file into a
single markdown/JSON summary table::

    python -m src.evaluate --summarize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ROOT, results_dir
from .utils import LOG, save_json

# --------------------------------------------------------------------------- #
# multi-label metrics
# --------------------------------------------------------------------------- #


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, float]:
    """Macro/micro F1 + mean AUC-PR for multi-label tagging."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= thresholds).astype(int)

    has_pos = y_true.sum(0) > 0
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }
    if has_pos.any():
        out["auc_pr"] = float(
            average_precision_score(y_true[:, has_pos], y_prob[:, has_pos], average="macro")
        )
        try:
            out["auc_roc"] = float(
                roc_auc_score(y_true[:, has_pos], y_prob[:, has_pos], average="macro")
            )
        except ValueError:
            out["auc_roc"] = float("nan")
    else:  # pragma: no cover
        out["auc_pr"] = float("nan")
        out["auc_roc"] = float("nan")
    return out


def tune_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, grid: Sequence[float] | None = None
) -> np.ndarray:
    """Per-tag decision threshold maximising F1 on a validation split."""
    from sklearn.metrics import f1_score

    grid = np.asarray(grid if grid is not None else np.arange(0.05, 0.96, 0.05))
    y_true = np.asarray(y_true).astype(int)
    best = np.full(y_true.shape[1], 0.5, dtype=np.float64)
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        scores = [
            f1_score(y_true[:, k], (y_prob[:, k] >= t).astype(int), zero_division=0)
            for t in grid
        ]
        best[k] = float(grid[int(np.argmax(scores))])
    return best


# --------------------------------------------------------------------------- #
# single-label metrics
# --------------------------------------------------------------------------- #


def singlelabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = y_prob.argmax(1)
    onehot = np.zeros_like(y_prob)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    present = onehot.sum(0) > 0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "auc_pr": float(
            average_precision_score(onehot[:, present], y_prob[:, present], average="macro")
        ),
        "n_classes": int(y_prob.shape[1]),
    }


# --------------------------------------------------------------------------- #
# regression metrics (DEAM valence / arousal)
# --------------------------------------------------------------------------- #


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: Sequence[str] = ("valence", "arousal"),
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)
    out: dict[str, float] = {}
    maes = []
    for i, name in enumerate(names[: y_true.shape[1]]):
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        maes.append(mae)
        out[f"mae_{name}"] = mae
        out[f"rmse_{name}"] = float(np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)))
        out[f"r2_{name}"] = float(r2_score(y_true[:, i], y_pred[:, i]))
    out["mae_mean"] = float(np.mean(maes)) if maes else float("nan")
    return out


# --------------------------------------------------------------------------- #
# graph coherence score (optional analysis, Section 6)
# --------------------------------------------------------------------------- #


def graph_coherence(h: np.ndarray, edge_index: np.ndarray, tau: float = 0.5) -> float:
    """Fraction of edges whose endpoint embeddings have cosine similarity > tau."""
    if edge_index.size == 0:
        return float("nan")
    hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
    src, dst = edge_index[0], edge_index[1]
    sims = (hn[src] * hn[dst]).sum(1)
    return float((sims > tau).mean())


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history(history: list[dict[str, Any]], dest: str | Path, title: str = "") -> Path:
    """Macro-F1 / Micro-F1 (and loss) versus epoch -- Task 1 deliverable."""
    plt = _mpl()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    keys = [
        k
        for k in ("val_macro_f1", "val_micro_f1", "val_auc_pr", "val_accuracy", "train_macro_f1")
        if k in history[0]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in keys:
        axes[0].plot(epochs, [h.get(k, np.nan) for h in history], marker="o", label=k)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("score")
    axes[0].set_title(f"{title} metrics vs epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    for k in ("train_loss", "val_loss"):
        if k in history[0]:
            axes[1].plot(epochs, [h.get(k, np.nan) for h in history], marker="o", label=k)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title(f"{title} loss vs epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    LOG.info("saved plot -> %s", dest)
    return dest


def plot_tsne(
    Z: np.ndarray,
    labels: Sequence[str],
    dest: str | Path,
    title: str = "t-SNE of fused representation z",
    max_points: int = 2000,
    perplexity: float = 30.0,
    seed: int = 42,
) -> Path:
    from sklearn.manifold import TSNE

    plt = _mpl()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Z = np.asarray(Z, dtype=np.float32)
    labels = np.asarray(labels)
    if len(Z) > max_points:
        idx = np.random.RandomState(seed).choice(len(Z), max_points, replace=False)
        Z, labels = Z[idx], labels[idx]
    perp = float(min(perplexity, max(5.0, (len(Z) - 1) / 3.0)))
    emb = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(Z)

    uniq = sorted(set(labels.tolist()))[:20]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for i, u in enumerate(uniq):
        m = labels == u
        ax.scatter(emb[m, 0], emb[m, 1], s=9, alpha=0.75, color=cmap(i % 20), label=str(u))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    LOG.info("saved plot -> %s", dest)
    return dest


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    dest: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    from sklearn.metrics import confusion_matrix

    plt = _mpl()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(classes), 1 + 0.55 * len(classes)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cmn[i, j] > 0.005:
                ax.text(
                    j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if cmn[i, j] > 0.5 else "black",
                )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def plot_bar_comparison(
    rows: list[dict[str, Any]], metric: str, dest: str | Path, title: str = ""
) -> Path:
    plt = _mpl()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if metric in r and r[metric] == r[metric]]
    if not rows:
        return dest
    names = [str(r.get("model", "?")) for r in rows]
    vals = [r[metric] for r in rows]
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(names)), 4))
    ax.bar(names, vals, color="#4C78A8")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


# --------------------------------------------------------------------------- #
# result bookkeeping
# --------------------------------------------------------------------------- #


def record_result(payload: dict[str, Any], name: str) -> Path:
    """Persist one experiment's metrics to ``results/metrics/<name>.json``."""
    dest = results_dir("metrics") / f"{name}.json"
    save_json(payload, dest)
    LOG.info("saved metrics -> %s", dest)
    return dest


def summarize(pattern: str = "*.json") -> str:
    files = sorted((ROOT / "results" / "metrics").glob(pattern))
    rows: list[dict[str, Any]] = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        # `*_table.json` side-cars are plain lists of rows, not experiment payloads
        if not isinstance(d, dict):
            continue
        test = d.get("test", {})
        if not isinstance(test, dict):
            continue
        rows.append(
            {
                "experiment": f.stem,
                "task": d.get("task"),
                "dataset": d.get("dataset"),
                "model": d.get("model"),
                **{
                    k: test[k]
                    for k in ("accuracy", "macro_f1", "micro_f1", "auc_pr",
                              "mae_mean", "r2_valence",
                              "caption_to_audio_R@5", "caption_to_audio_R@10")
                    if k in test
                },
            }
        )
    if not rows:
        return "no results yet"

    cols = ["experiment", "task", "dataset", "model", "accuracy", "macro_f1",
            "micro_f1", "auc_pr", "mae_mean", "r2_valence",
            "caption_to_audio_R@5", "caption_to_audio_R@10"]
    cols = [c for c in cols if any(c in r for r in rows)]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    save_json(rows, ROOT / "results" / "metrics.json")
    (ROOT / "results" / "summary.md").write_text(table + "\n", encoding="utf-8")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="aggregate experiment metrics")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--pattern", default="*.json")
    args = ap.parse_args()
    if args.summarize:
        print(summarize(args.pattern))


if __name__ == "__main__":
    main()

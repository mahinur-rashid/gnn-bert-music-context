"""Cross-dataset comparison for Tasks 1-3.

Table 1 of the brief recommends several datasets per task.  This module runs
one task over *every* applicable dataset with identical hyper-parameters and
tabulates train / validation / test results side by side, so the effect of the
dataset (label space, supervision quality, text richness) can be read off
directly.

    python -m src.compare_datasets --task 1
    python -m src.compare_datasets --task 2 --datasets gtzan fma_small
    python -m src.compare_datasets --task 3 --epochs 4
    python -m src.compare_datasets --task all

Results are written to ``results/comparison/``.
"""
from __future__ import annotations

import argparse
import traceback
from typing import Any

import numpy as np

from . import train_task1, train_task2, train_task3
from .config import ROOT, SEED, results_dir
from .evaluate import summarize
from .utils import LOG, Timer, save_json

# Which datasets each task can be run on, following Table 1 of the brief.
#   Task 1 needs text            -> MusicCaps, MagnaTagATune, FMA, DEAM
#   Task 2 needs audio graphs    -> GTZAN, FMA-small/medium, MagnaTagATune, DEAM
#   Task 3 needs graphs AND text -> FMA-small/medium, MagnaTagATune, DEAM
TASK_DATASETS: dict[int, list[str]] = {
    1: ["musiccaps", "mtat", "fma_medium", "fma_small", "deam"],
    2: ["gtzan", "fma_small", "mtat", "deam", "fma_medium"],
    3: ["fma_small", "mtat", "deam", "fma_medium"],
}

# Sensible defaults so a full comparison finishes in reasonable time.
DEFAULT_SELECTION: dict[int, list[str]] = {
    1: ["musiccaps", "mtat", "fma_medium", "deam"],
    2: ["gtzan", "fma_small", "deam"],
    3: ["fma_small", "mtat", "deam"],
}

# Columns reported per task.
REPORT_COLUMNS: dict[int, list[str]] = {
    1: ["macro_f1", "micro_f1", "auc_pr"],
    2: ["accuracy", "macro_f1", "micro_f1", "auc_pr"],
    3: ["macro_f1", "micro_f1", "auc_pr", "mae_mean", "r2_valence", "graph_coherence"],
}


# --------------------------------------------------------------------------- #
# per-task runners
# --------------------------------------------------------------------------- #


def _args(parser_builder, overrides: dict[str, Any]):
    ns = parser_builder().parse_args([])
    for k, v in overrides.items():
        if v is not None:
            setattr(ns, k, v)
    return ns


def run_task1(dataset: str, cli) -> dict:
    args = _args(train_task1.build_argparser,
                 {"dataset": dataset, "epochs": cli.epochs, "batch_size": cli.batch_size,
                  "model_name": cli.model_name, "limit": cli.limit, "seed": cli.seed})
    payload = train_task1.train_one(dataset, args)
    return {
        "dataset": dataset,
        "model": payload["model"],
        "n_train": payload["n_train"], "n_val": payload["n_val"], "n_test": payload["n_test"],
        "n_classes": payload["n_tags"],
        "train": payload.get("train", {}), "val": payload.get("val", {}),
        "test": payload["test"],
        "baseline_random": payload["baselines"]["B1_random_tags"],
        "best_epoch": payload["best_epoch"],
    }


def run_task2(dataset: str, cli) -> dict:
    args = _args(train_task2.build_argparser,
                 {"dataset": dataset, "epochs": cli.epochs, "batch_size": cli.batch_size,
                  "limit": cli.limit, "seed": cli.seed, "compare": True,
                  "pca_mlp": cli.pca_mlp})
    payload = train_task2.run(args)
    runs = {r["model"]: r for r in payload["runs"]}
    best_name = max((m for m in runs if m in train_task2.MODELS),
                    key=lambda m: runs[m]["test"].get("macro_f1", 0.0))
    best = runs[best_name]
    return {
        "dataset": dataset,
        "model": best_name,
        "n_train": payload["n_train"], "n_val": payload["n_val"], "n_test": payload["n_test"],
        "n_classes": len(payload["classes"]),
        "train": best.get("train", {}), "val": best.get("val", {}), "test": best["test"],
        "baseline_random": runs["B1_random"]["test"],
        "cnn_baseline": runs.get("cnn", {}).get("test", {}),
        "per_model": {m: runs[m]["test"] for m in runs},
        "best_epoch": best.get("best_epoch"),
    }


def run_task3(dataset: str, cli) -> dict:
    args = _args(train_task3.build_argparser,
                 {"dataset": dataset, "epochs": cli.epochs, "batch_size": cli.batch_size,
                  "model_name": cli.model_name, "limit": cli.limit, "seed": cli.seed,
                  "ablation": True})
    payload = train_task3.run(args)
    runs = {r["model"]: r for r in payload["runs"]}
    main = runs.get("cross_attn", list(runs.values())[-1])
    return {
        "dataset": dataset,
        "model": "cross_attn",
        "n_train": payload["n_train"], "n_val": payload["n_val"], "n_test": payload["n_test"],
        "n_classes": len(payload["classes"]),
        "train": main.get("train", {}), "val": main.get("val", {}), "test": main["test"],
        "baseline_random": runs.get("B1_random", {}).get("test", {}),
        "ablation": {m: runs[m]["test"] for m in runs},
        "best_epoch": main.get("best_epoch"),
    }


RUNNERS = {1: run_task1, 2: run_task2, 3: run_task3}


# --------------------------------------------------------------------------- #
# tables + plots
# --------------------------------------------------------------------------- #


def _fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.4f}"
    return str(v) if v not in (None, {}) else "-"


def comparison_table(task: int, rows: list[dict]) -> str:
    cols = REPORT_COLUMNS[task]
    present = [c for c in cols if any(c in r["test"] for r in rows)]
    head = (["dataset", "model", "n_train", "n_test", "n_classes"]
            + [f"test_{c}" for c in present]
            + [f"val_{c}" for c in present[:2]]
            + [f"train_{c}" for c in present[:2]]
            + ["random_macro_f1"])
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows:
        cells = [r["dataset"], str(r["model"]), str(r["n_train"]), str(r["n_test"]),
                 str(r["n_classes"])]
        cells += [_fmt(r["test"].get(c)) for c in present]
        cells += [_fmt(r.get("val", {}).get(c)) for c in present[:2]]
        cells += [_fmt(r.get("train", {}).get(c)) for c in present[:2]]
        cells += [_fmt(r.get("baseline_random", {}).get("macro_f1"))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def grouped_bar(task: int, rows: list[dict], dest) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [c for c in REPORT_COLUMNS[task][:3] if any(c in r["test"] for r in rows)]
    if not rows or not metrics:
        return
    names = [r["dataset"] for r in rows]
    x = np.arange(len(names))
    w = 0.8 / max(len(metrics) + 1, 2)

    fig, ax = plt.subplots(figsize=(max(6, 2.1 * len(names)), 4.4))
    for i, m in enumerate(metrics):
        vals = [r["test"].get(m, np.nan) for r in rows]
        bars = ax.bar(x + (i - len(metrics) / 2) * w, vals, w, label=f"test {m}")
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    base = [r.get("baseline_random", {}).get("macro_f1", np.nan) for r in rows]
    ax.bar(x + (len(metrics) / 2) * w, base, w, label="random macro-F1",
           color="#bbbbbb", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("score")
    ax.set_title(f"Task {task}: dataset comparison (test split)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    LOG.info("saved plot -> %s", dest)


def split_gap_plot(task: int, rows: list[dict], dest) -> None:
    """Train vs validation vs test macro-F1 per dataset (over/under-fitting)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("train") and "macro_f1" in r["train"]]
    if not rows:
        return
    names = [r["dataset"] for r in rows]
    x = np.arange(len(names))
    w = 0.26
    series = [("train", "#9ecae1"), ("val", "#4292c6"), ("test", "#08519c")]
    fig, ax = plt.subplots(figsize=(max(6, 2.1 * len(names)), 4.2))
    for i, (split, color) in enumerate(series):
        vals = [r.get(split, {}).get("macro_f1", np.nan) for r in rows]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=split, color=color)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("macro-F1")
    ax.set_title(f"Task {task}: train / val / test macro-F1 per dataset")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    LOG.info("saved plot -> %s", dest)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def compare(task: int, datasets: list[str], cli) -> dict:
    rows, failures = [], {}
    for ds in datasets:
        LOG.info("=" * 72)
        LOG.info("TASK %d | DATASET %s", task, ds)
        LOG.info("=" * 72)
        try:
            with Timer(f"compare/task{task}/{ds}"):
                rows.append(RUNNERS[task](ds, cli))
        except Exception as exc:
            LOG.error("task %d on %s failed: %s", task, ds, exc)
            LOG.debug(traceback.format_exc())
            failures[ds] = f"{type(exc).__name__}: {exc}"

    out = results_dir("comparison")
    table = comparison_table(task, rows) if rows else "no successful runs"
    payload = {"task": task, "datasets": datasets, "rows": rows, "failures": failures,
               "table_markdown": table}
    save_json(payload, out / f"task{task}_dataset_comparison.json")
    (out / f"task{task}_dataset_comparison.md").write_text(
        f"# Task {task} -- dataset comparison\n\n{table}\n", encoding="utf-8")
    if rows:
        grouped_bar(task, rows, out / f"task{task}_dataset_comparison.png")
        split_gap_plot(task, rows, out / f"task{task}_train_val_test.png")
    LOG.info("\nTASK %d DATASET COMPARISON\n%s", task, table)
    if failures:
        LOG.warning("failed datasets: %s", failures)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="cross-dataset comparison for tasks 1-3")
    ap.add_argument("--task", default="all", choices=["1", "2", "3", "all"])
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="override the dataset list for the selected task")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pca_mlp", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    cli = ap.parse_args()

    tasks = [1, 2, 3] if cli.task == "all" else [int(cli.task)]
    all_payloads = {}
    for t in tasks:
        datasets = cli.datasets or DEFAULT_SELECTION[t]
        unknown = [d for d in datasets if d not in TASK_DATASETS[t]]
        if unknown:
            LOG.warning("task %d: ignoring unsupported datasets %s", t, unknown)
            datasets = [d for d in datasets if d in TASK_DATASETS[t]]
        all_payloads[t] = compare(t, datasets, cli)

    save_json(all_payloads, results_dir("comparison") / "all_tasks_comparison.json")
    combined = "\n\n".join(
        f"## Task {t}\n\n{p['table_markdown']}" for t, p in all_payloads.items())
    (ROOT / "results" / "comparison" / "README.md").write_text(
        "# Dataset comparison across tasks\n\n" + combined + "\n", encoding="utf-8")
    print(summarize())


if __name__ == "__main__":
    main()

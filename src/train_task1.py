"""Task 1 (Easy) -- BERT baseline for music tag understanding.

Fine-tunes a BERT/DistilBERT multi-label tag classifier on textual music
context (captions, tags, track metadata) with **no graph structure**.

    python -m src.train_task1 --dataset musiccaps
    python -m src.train_task1 --dataset mtat --epochs 4
    python -m src.train_task1 --dataset fma_medium
    python -m src.train_task1 --dataset all
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import prepare_splits, text_data
from .baselines import majority_multilabel_baseline, random_multilabel_baseline
from .bert_encoder import BertTagClassifier, TextTagDataset, cls_attention_tokens, make_tokenizer
from .config import CFG, SEED, results_dir
from .evaluate import multilabel_metrics, plot_history, record_result, tune_thresholds
from .utils import (LOG, Timer, count_params, get_device, quiet_transformers,
                    save_json, set_seed, split_labels)

quiet_transformers()

DATASETS = ["musiccaps", "mtat", "fma_small", "fma_medium", "deam"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def load_split_frames(dataset: str, limit: int | None = None):
    df, vocab = text_data.load(dataset)
    splits = prepare_splits.load(dataset)
    df = df.set_index("id", drop=False)

    frames = {}
    for name in ("train", "val", "test"):
        ids = [i for i in splits[name] if i in df.index]
        sub = df.loc[ids]
        if limit and name == "train":
            sub = sub.iloc[:limit]
        elif limit:
            sub = sub.iloc[: max(64, limit // 8)]
        frames[name] = sub
    LOG.info("%s: train %d / val %d / test %d, %d tags",
             dataset, *(len(frames[k]) for k in ("train", "val", "test")), len(vocab))
    return frames, vocab


def make_loaders(frames, vocab, tokenizer, batch_size: int, max_length: int,
                 num_workers: int = 0):
    loaders, datasets = {}, {}
    for name, sub in frames.items():
        Y = text_data.labels_to_matrix(sub["labels"], vocab)
        va = None
        if {"valence", "arousal"}.issubset(sub.columns):
            va = sub[["valence", "arousal"]].to_numpy(dtype=np.float32)
        ds = TextTagDataset(sub["text"].tolist(), Y, tokenizer, max_length, va,
                            ids=sub["id"].tolist())
        datasets[name] = ds
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loaders, datasets


# --------------------------------------------------------------------------- #
# train / eval loops
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict(model, loader, device, amp: bool = True):
    model.eval()
    probs, trues, losses = [], [], []
    crit = nn.BCEWithLogitsLoss()
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(ids, mask)
            loss = crit(out["logits"].float(), y)
        probs.append(torch.sigmoid(out["logits"].float()).cpu().numpy())
        trues.append(y.cpu().numpy())
        losses.append(loss.item())
    return np.concatenate(probs), np.concatenate(trues), float(np.mean(losses))


def train_one(dataset: str, args) -> dict:
    set_seed(args.seed)
    device = get_device()
    frames, vocab = load_split_frames(dataset, args.limit)
    tokenizer = make_tokenizer(args.model_name)
    loaders, datasets = make_loaders(frames, vocab, tokenizer, args.batch_size,
                                     args.max_length, args.num_workers)

    model = BertTagClassifier(
        n_tags=len(vocab),
        model_name=args.model_name,
        freeze=args.freeze_bert,
        n_trainable_layers=args.n_trainable_layers,
    ).to(device)
    LOG.info("model: %s, %.1fM trainable params",
             args.model_name, count_params(model) / 1e6)

    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.bert")]
    bert_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("encoder.bert")]
    optim = torch.optim.AdamW(
        [{"params": bert_params, "lr": args.lr},
         {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    steps = max(1, len(loaders["train"])) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.BCEWithLogitsLoss()

    history, best = [], {"val_macro_f1": -1.0}
    best_state: dict | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for step, batch in enumerate(loaders["train"], 1):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(ids, mask)
                loss = crit(out["logits"].float(), y)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            run_loss += loss.detach().item() * len(y)
            n += len(y)
            if step % args.log_every == 0:
                LOG.info("  epoch %d step %d/%d loss %.4f",
                         epoch, step, len(loaders["train"]), run_loss / max(n, 1))

        vp, vt, vloss = predict(model, loaders["val"], device)
        vm = multilabel_metrics(vt, vp)
        rec = {"epoch": epoch, "train_loss": run_loss / max(n, 1), "val_loss": vloss,
               "val_macro_f1": vm["macro_f1"], "val_micro_f1": vm["micro_f1"],
               "val_auc_pr": vm["auc_pr"], "secs": time.time() - t0}
        history.append(rec)
        LOG.info("epoch %d | train loss %.4f | val loss %.4f | macro-F1 %.4f | "
                 "micro-F1 %.4f | AUC-PR %.4f | %.0fs",
                 epoch, rec["train_loss"], vloss, vm["macro_f1"], vm["micro_f1"],
                 vm["auc_pr"], rec["secs"])
        if vm["macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": vm["macro_f1"], "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    LOG.info("restored best epoch %d (val macro-F1 %.4f)", best["epoch"], best["val_macro_f1"])

    # --- threshold tuning on validation, then test -------------------------- #
    vp, vt, _ = predict(model, loaders["val"], device)
    thr = tune_thresholds(vt, vp) if args.tune_thresholds else 0.5
    tp, tt, _ = predict(model, loaders["test"], device)
    test_metrics = multilabel_metrics(tt, tp, thresholds=thr)
    test_at_half = multilabel_metrics(tt, tp, thresholds=0.5)
    val_metrics = multilabel_metrics(vt, vp, thresholds=thr)
    trp, trt, _ = predict(model, loaders["train"], device)
    train_metrics = multilabel_metrics(trt, trp, thresholds=thr)
    LOG.info("TRAIN %s | macro-F1 %.4f (generalisation gap %.4f)", dataset,
             train_metrics["macro_f1"], train_metrics["macro_f1"] - test_metrics["macro_f1"])
    LOG.info("TEST %s | macro-F1 %.4f | micro-F1 %.4f | AUC-PR %.4f",
             dataset, test_metrics["macro_f1"], test_metrics["micro_f1"],
             test_metrics["auc_pr"])

    # --- baselines ---------------------------------------------------------- #
    ytr = text_data.labels_to_matrix(frames["train"]["labels"], vocab)
    baselines = {
        "B1_random_tags": random_multilabel_baseline(ytr, tt, seed=args.seed),
        "B1_majority_tags": majority_multilabel_baseline(ytr, tt),
    }
    for k, v in baselines.items():
        LOG.info("baseline %-18s macro-F1 %.4f  micro-F1 %.4f  AUC-PR %.4f",
                 k, v["macro_f1"], v["micro_f1"], v["auc_pr"])

    # --- artefacts ---------------------------------------------------------- #
    tag = f"task1_{dataset}_{Path(args.model_name).name}"
    plot_history(history, results_dir("plots") / f"{tag}_curves.png",
                 title=f"Task 1 / {dataset}")
    examples = dump_examples(model, tokenizer, frames["test"], datasets["test"], tp, tt,
                             vocab, device, tag, args)

    payload = {
        "task": 1,
        "dataset": dataset,
        "model": f"BERT({args.model_name})",
        "n_tags": len(vocab),
        "tags": vocab,
        "n_train": int(len(frames["train"])),
        "n_val": int(len(frames["val"])),
        "n_test": int(len(frames["test"])),
        "args": vars(args),
        "history": history,
        "best_epoch": best["epoch"],
        "thresholds": (thr.tolist() if isinstance(thr, np.ndarray) else thr),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "test_at_threshold_0.5": test_at_half,
        "baselines": baselines,
        "examples_file": str(examples),
    }
    record_result(payload, tag)
    if args.save_model:
        torch.save(model.state_dict(), results_dir("checkpoints") / f"{tag}.pt")
    return payload


# --------------------------------------------------------------------------- #
# qualitative output (5 example predictions + attention)
# --------------------------------------------------------------------------- #


def dump_examples(model, tokenizer, frame: pd.DataFrame, dataset, probs, trues,
                  vocab, device, tag: str, args, n: int = 5) -> Path:
    rows = []
    idx = np.linspace(0, len(frame) - 1, min(n, len(frame))).astype(int)
    for i in idx:
        p = probs[i]
        order = np.argsort(-p)[:5]
        row = {
            "id": str(frame.iloc[i]["id"]),
            "text": str(frame.iloc[i]["text"])[:400],
            "true_tags": split_labels(frame.iloc[i]["labels"]),
            "predicted_tags": [{"tag": vocab[j], "p": round(float(p[j]), 4)} for j in order],
        }
        if args.attention_viz:
            row["cls_attention_top_tokens"] = [
                {"token": t, "weight": round(w, 4)}
                for t, w in cls_attention_tokens(model, tokenizer, row["text"], device,
                                                 args.max_length)
            ]
        rows.append(row)
    dest = results_dir("examples") / f"{tag}_examples.json"
    save_json(rows, dest)
    LOG.info("saved %d example predictions -> %s", len(rows), dest)
    return dest


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def build_argparser() -> argparse.ArgumentParser:
    c = CFG["task1"]
    ap = argparse.ArgumentParser(description="Task 1: BERT multi-label tag classifier")
    ap.add_argument("--dataset", default="musiccaps", choices=DATASETS + ["all"])
    ap.add_argument("--model_name", default=CFG["text"]["model_name"])
    ap.add_argument("--epochs", type=int, default=c["epochs"])
    ap.add_argument("--batch_size", type=int, default=c["batch_size"])
    ap.add_argument("--lr", type=float, default=c["lr"], help="BERT learning rate")
    ap.add_argument("--head_lr", type=float, default=c["head_lr"])
    ap.add_argument("--weight_decay", type=float, default=c["weight_decay"])
    ap.add_argument("--max_length", type=int, default=CFG["text"]["max_length"])
    ap.add_argument("--freeze_bert", action="store_true")
    ap.add_argument("--n_trainable_layers", type=int, default=None,
                    help="fine-tune only the last N transformer blocks")
    ap.add_argument("--tune_thresholds", action="store_true", default=True)
    ap.add_argument("--no_tune_thresholds", dest="tune_thresholds", action="store_false")
    ap.add_argument("--attention_viz", action="store_true",
                    help="dump [CLS] attention for the 5 example predictions")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap training rows")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        with Timer(f"task1/{ds}"):
            train_one(ds, args)


if __name__ == "__main__":
    main()

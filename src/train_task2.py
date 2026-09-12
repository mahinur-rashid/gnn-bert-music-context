"""Task 2 (Medium) -- GNN on music-structure graphs (audio-only features).

Trains GraphSAGE / GAT encoders on segment+chord graphs and compares them with
the required baselines: random (B1), CNN on mel-spectrogram (B2) and, optionally,
PCA+MLP on hand-crafted features (B4).

    python -m src.train_task2 --dataset gtzan --model sage
    python -m src.train_task2 --dataset gtzan --compare
    python -m src.train_task2 --dataset fma_small --compare
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from . import graph_dataset as gd
from .baselines import (majority_multilabel_baseline, majority_singlelabel_baseline,
                        pca_mlp_baseline, random_multilabel_baseline,
                        random_singlelabel_baseline)
from .config import CFG, SEED, results_dir
from .evaluate import (multilabel_metrics, plot_bar_comparison, plot_confusion,
                       plot_history, record_result, singlelabel_metrics, tune_thresholds)
from .gnn_model import GNNClassifier, MelCNN
from .utils import LOG, Timer, count_params, get_device, save_json, set_seed

DATASETS = ["gtzan", "fma_small", "fma_medium", "mtat", "deam", "deam_feats"]
MODELS = ["sage", "gat", "gcn", "cnn"]


# --------------------------------------------------------------------------- #
# batching helpers
# --------------------------------------------------------------------------- #


def batch_targets(batch, multilabel: bool, device):
    y = batch.y.to(device)
    return y.float() if multilabel else y.view(-1).long()


def forward_batch(model, batch, device, is_cnn: bool):
    if is_cnn:
        mel = batch.mel.to(device)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        return model(mel)
    return model(batch.to(device))


@torch.no_grad()
def evaluate_model(model, loader, device, multilabel: bool, is_cnn: bool, crit):
    model.eval()
    probs, trues, losses = [], [], []
    for batch in loader:
        y = batch_targets(batch, multilabel, device)
        out = forward_batch(model, batch, device, is_cnn)
        logits = out["logits"].float()
        losses.append(crit(logits, y).item())
        p = torch.sigmoid(logits) if multilabel else torch.softmax(logits, -1)
        probs.append(p.cpu().numpy())
        trues.append(y.cpu().numpy())
    P = np.concatenate(probs)
    T = np.concatenate(trues)
    metrics = multilabel_metrics(T, P) if multilabel else singlelabel_metrics(T, P)
    return P, T, float(np.mean(losses)), metrics


def train_model(model_name: str, bundle: dict, args, device) -> dict:
    set_seed(args.seed)
    spec = bundle["spec"]
    multilabel = spec.multilabel
    is_cnn = model_name == "cnn"

    if is_cnn and not hasattr(bundle["splits"]["train"][0], "mel"):
        raise RuntimeError("mel patches missing -- rebuild graphs without --no_mel")

    loaders = {
        k: gd.loader(v, args.batch_size, shuffle=(k == "train"), num_workers=args.num_workers)
        for k, v in bundle["splits"].items()
    }

    if is_cnn:
        model = MelCNN(n_classes=len(spec.classes), dropout=args.dropout).to(device)
    else:
        model = GNNClassifier(
            in_dim=bundle["in_dim"], n_classes=len(spec.classes), hidden=args.hidden,
            layers=args.layers, conv=model_name, dropout=args.dropout,
            edge_dim=bundle["edge_dim"] if model_name == "gat" and args.use_edge_attr else None,
        ).to(device)
    LOG.info("model=%s params=%.2fM", model_name, count_params(model) / 1e6)

    crit = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    key = "macro_f1" if multilabel else "accuracy"
    history, best, best_state, patience = [], {key: -1.0, "epoch": 0}, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for batch in loaders["train"]:
            y = batch_targets(batch, multilabel, device)
            out = forward_batch(model, batch, device, is_cnn)
            loss = crit(out["logits"].float(), y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()
            run += loss.detach().item() * y.shape[0]
            n += y.shape[0]
        sched.step()

        _, _, vloss, vm = evaluate_model(model, loaders["val"], device, multilabel, is_cnn, crit)
        rec = {"epoch": epoch, "train_loss": run / max(n, 1), "val_loss": vloss,
               "val_macro_f1": vm["macro_f1"], "val_micro_f1": vm["micro_f1"],
               "val_auc_pr": vm["auc_pr"], "secs": time.time() - t0}
        if not multilabel:
            rec["val_accuracy"] = vm["accuracy"]
        history.append(rec)
        if epoch % args.log_every == 0 or epoch == args.epochs:
            LOG.info("[%s] epoch %3d | loss %.4f | val %s %.4f | macro-F1 %.4f | %.1fs",
                     model_name, epoch, rec["train_loss"], key, vm[key], vm["macro_f1"],
                     rec["secs"])

        if vm[key] > best[key]:
            best = {key: vm[key], "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if (args.patience and epoch >= args.min_epochs
                    and patience >= args.patience):
                LOG.info("[%s] early stop at epoch %d (no val gain for %d epochs)",
                         model_name, epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    thr = 0.5
    if multilabel and args.tune_thresholds:
        vp, vt, _, _ = evaluate_model(model, loaders["val"], device, multilabel, is_cnn, crit)
        thr = tune_thresholds(vt, vp)
    tp, tt, _, tm = evaluate_model(model, loaders["test"], device, multilabel, is_cnn, crit)
    vp2, vt2, _, vm2 = evaluate_model(model, loaders["val"], device, multilabel, is_cnn, crit)
    rp, rt, _, rm = evaluate_model(model, loaders["train"], device, multilabel, is_cnn, crit)
    if multilabel and args.tune_thresholds:
        tm = multilabel_metrics(tt, tp, thresholds=thr)
        vm2 = multilabel_metrics(vt2, vp2, thresholds=thr)
        rm = multilabel_metrics(rt, rp, thresholds=thr)
    LOG.info("[%s] TEST %s", model_name,
             " ".join(f"{k}={v:.4f}" for k, v in tm.items() if isinstance(v, float)))

    tag = f"task2_{args.dataset}_{model_name}"
    plot_history(history, results_dir("plots") / f"{tag}_curves.png",
                 title=f"Task 2 / {args.dataset} / {model_name}")
    if not multilabel:
        plot_confusion(tt, tp.argmax(1), spec.classes,
                       results_dir("plots") / f"{tag}_confusion.png",
                       title=f"{args.dataset} / {model_name}")
    if args.save_model:
        torch.save(model.state_dict(), results_dir("checkpoints") / f"{tag}.pt")

    return {"model": model_name, "history": history, "best_epoch": best["epoch"],
            "train": rm, "val": vm2, "test": tm, "n_params": count_params(model),
            "thresholds": thr.tolist() if isinstance(thr, np.ndarray) else thr}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run(args) -> dict:
    device = get_device()
    set_seed(args.seed)
    models = MODELS if args.compare else [args.model]
    need_mel = "cnn" in models

    bundle = gd.build_pyg_splits(
        args.dataset, label_mode=args.label_mode, want_mel=need_mel,
        limit=args.limit, standardize=not args.no_standardize,
    )
    if need_mel and not hasattr(bundle["splits"]["train"][0], "mel"):
        LOG.warning("%s: no mel patches stored -- skipping the CNN baseline "
                    "(rebuild graphs without --no_mel to include it)", args.dataset)
        models = [m for m in models if m != "cnn"]
    spec = bundle["spec"]
    multilabel = spec.multilabel

    results: list[dict] = []

    # --- B1 baselines ------------------------------------------------------- #
    ytr = gd.labels_matrix(bundle["splits"]["train"], multilabel)
    yte = gd.labels_matrix(bundle["splits"]["test"], multilabel)
    if multilabel:
        results.append({"model": "B1_random", "test": random_multilabel_baseline(ytr, yte, args.seed)})
        results.append({"model": "B1_majority", "test": majority_multilabel_baseline(ytr, yte)})
    else:
        k = len(spec.classes)
        results.append({"model": "B1_random",
                        "test": random_singlelabel_baseline(ytr, yte, k, args.seed)})
        results.append({"model": "B1_majority",
                        "test": majority_singlelabel_baseline(ytr, yte, k)})
    for r in results:
        LOG.info("baseline %-12s %s", r["model"],
                 " ".join(f"{k}={v:.4f}" for k, v in r["test"].items() if isinstance(v, float)))

    # --- B4 PCA + MLP ------------------------------------------------------- #
    if args.pca_mlp:
        Xtr = gd.mean_node_features(bundle["splits"]["train"])
        Xte = gd.mean_node_features(bundle["splits"]["test"])
        try:
            m = pca_mlp_baseline(Xtr, ytr, Xte, yte, multilabel=multilabel, seed=args.seed)
            results.append({"model": "B4_pca_mlp", "test": m})
            LOG.info("baseline B4_pca_mlp  %s",
                     " ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
        except Exception as exc:  # pragma: no cover
            LOG.warning("B4 failed: %s", exc)

    # --- neural models ------------------------------------------------------ #
    for name in models:
        with Timer(f"task2/{args.dataset}/{name}"):
            results.append(train_model(name, bundle, args, device))

    payload = {
        "task": 2,
        "dataset": args.dataset,
        "model": "+".join(models),
        "multilabel": multilabel,
        "classes": spec.classes,
        "in_dim": bundle["in_dim"],
        "n_train": len(bundle["splits"]["train"]),
        "n_val": len(bundle["splits"]["val"]),
        "n_test": len(bundle["splits"]["test"]),
        "args": vars(args),
        "runs": results,
        "test": next((r["test"] for r in reversed(results) if r["model"] in MODELS), {}),
        "comparison": {r["model"]: r["test"] for r in results},
    }
    tag = f"task2_{args.dataset}_{'compare' if args.compare else args.model}"
    record_result(payload, tag)

    flat = [{"model": r["model"], **r["test"]} for r in results]
    metric = "macro_f1" if multilabel else "accuracy"
    plot_bar_comparison(flat, metric, results_dir("plots") / f"{tag}_{metric}.png",
                        title=f"Task 2 / {args.dataset}: {metric}")
    plot_bar_comparison(flat, "macro_f1", results_dir("plots") / f"{tag}_macro_f1.png",
                        title=f"Task 2 / {args.dataset}: macro-F1")
    save_json(flat, results_dir("metrics") / f"{tag}_table.json")
    LOG.info("\n%s", _markdown(flat))
    return payload


def _markdown(rows: list[dict]) -> str:
    cols = ["model", "accuracy", "macro_f1", "micro_f1", "auc_pr"]
    cols = [c for c in cols if any(c in r for r in rows)]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols
        ) + " |")
    return "\n".join(out)


def build_argparser() -> argparse.ArgumentParser:
    c = CFG["task2"]
    ap = argparse.ArgumentParser(description="Task 2: GNN on music structure graphs")
    ap.add_argument("--dataset", default="gtzan", choices=DATASETS)
    ap.add_argument("--model", default=c["conv"], choices=MODELS)
    ap.add_argument("--compare", action="store_true",
                    help="train sage, gat, gcn and the CNN baseline and tabulate")
    ap.add_argument("--label_mode", default="auto", choices=["auto", "single", "multi"])
    ap.add_argument("--epochs", type=int, default=c["epochs"])
    ap.add_argument("--batch_size", type=int, default=c["batch_size"])
    ap.add_argument("--lr", type=float, default=c["lr"])
    ap.add_argument("--weight_decay", type=float, default=c["weight_decay"])
    ap.add_argument("--hidden", type=int, default=c["hidden"])
    ap.add_argument("--layers", type=int, default=c["layers"])
    ap.add_argument("--dropout", type=float, default=c["dropout"])
    ap.add_argument("--use_edge_attr", action="store_true",
                    help="feed edge attributes to GAT")
    ap.add_argument("--pca_mlp", action="store_true", help="also run the B4 baseline")
    ap.add_argument("--tune_thresholds", action="store_true", default=True)
    ap.add_argument("--no_tune_thresholds", dest="tune_thresholds", action="store_false")
    ap.add_argument("--no_standardize", action="store_true")
    ap.add_argument("--patience", type=int, default=c["patience"],
                    help="0 disables early stopping")
    ap.add_argument("--min_epochs", type=int, default=c["min_epochs"],
                    help="never early-stop before this epoch -- the CNN baseline "
                         "dips mid-training before recovering")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()

"""Task 3 (Hard) -- GNN-BERT fusion for multi-context understanding.

Predicts multi-label context (genre + mood tags) and, where DEAM annotations
exist, valence/arousal as an auxiliary regression task.

    python -m src.train_task3 --dataset fma_small --fusion cross_attn
    python -m src.train_task3 --dataset fma_small --ablation
    python -m src.train_task3 --dataset mtat --ablation --epochs 4
    python -m src.train_task3 --dataset deam --ablation
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import graph_dataset as gd
from .baselines import majority_multilabel_baseline, random_multilabel_baseline
from .bert_encoder import make_tokenizer
from .config import CFG, SEED, results_dir
from .evaluate import (graph_coherence, multilabel_metrics, plot_bar_comparison,
                       plot_history, plot_tsne, record_result, regression_metrics,
                       singlelabel_metrics, tune_thresholds)
from .fusion_model import FUSION_MODES, FusionModel, multitask_loss
from .utils import (LOG, Timer, count_params, get_device, quiet_transformers,
                    save_json, set_seed)

quiet_transformers()

DATASETS = ["fma_small", "fma_medium", "mtat", "deam", "deam_feats",
            "gtzan", "musiccaps"]


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model, loader, device, multilabel: bool, args, collect_z: bool = False):
    model.eval()
    probs, trues, zs, vas, va_true, ids, losses = [], [], [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.float() if multilabel else batch.y.view(-1).long()
        va = getattr(batch, "va", None)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch)
        loss, _ = multitask_loss(out, y, va, args.alpha, args.beta)
        losses.append(float(loss.detach()))
        logits = out["logits"].float()
        probs.append((torch.sigmoid(logits) if multilabel
                      else torch.softmax(logits, -1)).cpu().numpy())
        trues.append(y.cpu().numpy())
        ids.extend(list(batch.track_id))
        if out.get("va") is not None and va is not None:
            vas.append(out["va"].float().cpu().numpy())
            va_true.append(va.cpu().numpy())
        if collect_z:
            zs.append(out["z"].float().cpu().numpy())

    P, T = np.concatenate(probs), np.concatenate(trues)
    metrics = multilabel_metrics(T, P) if multilabel else singlelabel_metrics(T, P)
    if vas:
        metrics.update(regression_metrics(np.concatenate(va_true), np.concatenate(vas)))
    return {
        "probs": P, "trues": T, "loss": float(np.mean(losses)), "metrics": metrics,
        "z": np.concatenate(zs) if collect_z and zs else None,
        "va_pred": np.concatenate(vas) if vas else None,
        "ids": ids,
    }


# --------------------------------------------------------------------------- #
# optional analysis: graph coherence score S_graph (Section 6)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def graph_coherence_score(model, loader, device, tau: float = 0.5,
                          max_batches: int = 20) -> float:
    """Fraction of graph edges whose learnt node embeddings agree (cos > tau)."""
    model.eval()
    scores = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        out = model(batch)
        h = out.get("h")
        if h is None:
            return float("nan")
        scores.append(graph_coherence(h.float().cpu().numpy(),
                                      batch.edge_index.cpu().numpy(), tau))
    return float(np.nanmean(scores)) if scores else float("nan")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def train_fusion(mode: str, bundle: dict, args, device) -> dict:
    set_seed(args.seed)
    spec = bundle["spec"]
    multilabel = spec.multilabel
    has_va = spec.va is not None and not args.no_va

    loaders = {k: gd.loader(v, args.batch_size, shuffle=(k == "train"),
                            num_workers=args.num_workers)
               for k, v in bundle["splits"].items()}

    model = FusionModel(
        in_dim=bundle["in_dim"], n_tags=len(spec.classes), mode=mode,
        model_name=args.model_name, hidden=args.hidden, gnn_layers=args.gnn_layers,
        conv=args.conv, dropout=args.dropout, n_heads=args.n_heads,
        n_va=2 if has_va else 0, freeze_bert=args.freeze_bert,
        n_trainable_layers=args.n_trainable_layers,
        edge_dim=bundle["edge_dim"] if args.conv == "gat" and args.use_edge_attr else None,
    ).to(device)
    LOG.info("[%s] params %.1fM (trainable)", mode, count_params(model) / 1e6)

    bert_params = [p for n, p in model.named_parameters() if p.requires_grad and ".bert." in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and ".bert." not in n]
    groups = [{"params": other_params, "lr": args.lr_other}]
    if bert_params:
        groups.append({"params": bert_params, "lr": args.lr_bert})
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    # cosine over a bounded horizon: with early stopping the run rarely reaches
    # `epochs`, and a OneCycle schedule spanning the full budget would never decay
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    key = "macro_f1" if multilabel else "accuracy"
    history, best, best_state, patience = [], {key: -1.0, "epoch": 0}, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for step, batch in enumerate(loaders["train"], 1):
            batch = batch.to(device)
            y = batch.y.float() if multilabel else batch.y.view(-1).long()
            va = getattr(batch, "va", None) if has_va else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(batch)
            loss, parts = multitask_loss(out, y, va, args.alpha, args.beta)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            bs = y.shape[0]
            run += loss.detach().item() * bs
            n += bs
            if args.log_every and step % args.log_every == 0:
                LOG.info("  [%s] epoch %d step %d/%d loss %.4f %s", mode, epoch, step,
                         len(loaders["train"]), run / max(n, 1),
                         " ".join(f"{k}={v:.3f}" for k, v in parts.items()))

        sched.step()
        val = evaluate(model, loaders["val"], device, multilabel, args)
        vm = val["metrics"]
        rec = {"epoch": epoch, "train_loss": run / max(n, 1), "val_loss": val["loss"],
               "val_macro_f1": vm["macro_f1"], "val_micro_f1": vm["micro_f1"],
               "val_auc_pr": vm["auc_pr"], "secs": time.time() - t0}
        if not multilabel:
            rec["val_accuracy"] = vm["accuracy"]
        if "mae_mean" in vm:
            rec["val_mae"] = vm["mae_mean"]
        history.append(rec)
        LOG.info("[%s] epoch %d | train %.4f | val %.4f | macro-F1 %.4f | AUC-PR %.4f%s | %.0fs",
                 mode, epoch, rec["train_loss"], val["loss"], vm["macro_f1"], vm["auc_pr"],
                 f" | MAE {vm['mae_mean']:.4f}" if "mae_mean" in vm else "", rec["secs"])

        if vm[key] > best[key]:
            best = {key: vm[key], "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if (args.patience and epoch >= args.min_epochs
                    and patience >= args.patience):
                LOG.info("[%s] early stop at epoch %d (no val gain for %d epochs)",
                         mode, epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val = evaluate(model, loaders["val"], device, multilabel, args)
    thr = tune_thresholds(val["trues"], val["probs"]) if (multilabel and args.tune_thresholds) else 0.5
    test = evaluate(model, loaders["test"], device, multilabel, args, collect_z=True)
    train_eval = evaluate(model, loaders["train"], device, multilabel, args)
    tm, vm, rm = dict(test["metrics"]), dict(val["metrics"]), dict(train_eval["metrics"])
    if multilabel and args.tune_thresholds:
        tm.update(multilabel_metrics(test["trues"], test["probs"], thresholds=thr))
        vm.update(multilabel_metrics(val["trues"], val["probs"], thresholds=thr))
        rm.update(multilabel_metrics(train_eval["trues"], train_eval["probs"], thresholds=thr))
    if model.use_graph:
        tm["graph_coherence"] = graph_coherence_score(model, loaders["test"], device)
    LOG.info("[%s] TEST %s", mode,
             " ".join(f"{k}={v:.4f}" for k, v in tm.items() if isinstance(v, float)))

    tag = f"task3_{args.dataset}_{mode}"
    plot_history(history, results_dir("plots") / f"{tag}_curves.png",
                 title=f"Task 3 / {args.dataset} / {mode}")
    if args.save_model:
        torch.save(model.state_dict(), results_dir("checkpoints") / f"{tag}.pt")

    return {"model": mode, "history": history, "best_epoch": best["epoch"],
            "train": rm, "val": vm, "test": tm, "n_params": count_params(model),
            "thresholds": thr.tolist() if isinstance(thr, np.ndarray) else thr,
            "_model": model, "_test": test, "_loaders": loaders}


# --------------------------------------------------------------------------- #
# qualitative deliverables
# --------------------------------------------------------------------------- #


def tsne_plots(test: dict, bundle: dict, args, tag: str) -> list[str]:
    """t-SNE of z coloured by genre and (when available) by mood quadrant."""
    Z, spec = test["z"], bundle["spec"]
    if Z is None:
        return []
    paths = []
    T = test["trues"]
    if spec.multilabel:
        top = T.argmax(1)
        genre = [spec.classes[i] if T[r].sum() else "none" for r, i in enumerate(top)]
    else:
        genre = [spec.classes[int(i)] for i in T]
    paths.append(str(plot_tsne(Z, genre, results_dir("plots") / f"{tag}_tsne_genre.png",
                               title=f"t-SNE of z -- {args.dataset} (coloured by genre)")))

    if test["va_pred"] is not None:
        ids = test["ids"]
        lut = {i: k for k, i in enumerate(spec.ids)}
        va = np.stack([spec.va[lut[i]] for i in ids])
        mood = [f"{'high' if a > 0 else 'low'}-arousal / {'pos' if v > 0 else 'neg'}-valence"
                for v, a in va]
        paths.append(str(plot_tsne(Z, mood, results_dir("plots") / f"{tag}_tsne_mood.png",
                                   title=f"t-SNE of z -- {args.dataset} (coloured by mood)")))
    return paths


@torch.no_grad()
def case_studies(run: dict, bundle: dict, args, tag: str, n: int = 3) -> str:
    """Graph paths + the text tokens the graph attends to, for n test tracks."""
    model, loaders = run["_model"], run["_loaders"]
    if model.mode != "cross_attn":
        return ""
    device = next(model.parameters()).device
    tokenizer = make_tokenizer(args.model_name)
    spec = bundle["spec"]
    from .audio_features import CHORD_NAMES

    model.eval()
    rows = []
    for batch in loaders["test"]:
        batch = batch.to(device)
        out = model(batch, output_attentions=True)
        attn = out["text_attention"].float().cpu().numpy()
        probs = torch.sigmoid(out["logits"].float()).cpu().numpy()
        for b in range(min(n - len(rows), batch.num_graphs)):
            sub = batch[b]
            ids = sub.input_ids.view(-1)
            mask = sub.attention_mask.view(-1).bool()
            toks = tokenizer.convert_ids_to_tokens(ids[mask])
            w = attn[b][: int(mask.sum())]
            order = np.argsort(-w)[:10]
            chords = [CHORD_NAMES[int(c)] for c in sub.chords.cpu().numpy()]
            ei = sub.edge_index.cpu().numpy()
            ea = sub.edge_attr.cpu().numpy()
            strong = np.argsort(-ea[:, 1])[:8] if len(ea) else []
            y = sub.y.view(-1).cpu().numpy()
            p = probs[b]
            rows.append({
                "track_id": str(sub.track_id),
                "text": tokenizer.decode(ids[mask], skip_special_tokens=True)[:400],
                "true_tags": [spec.classes[i] for i in np.where(y > 0.5)[0]]
                             if spec.multilabel else [spec.classes[int(y[0])]],
                "top_predicted_tags": [{"tag": spec.classes[i], "p": round(float(p[i]), 4)}
                                       for i in np.argsort(-p)[:5]],
                "chord_path": " -> ".join(chords[:12]),
                "strongest_similarity_edges": [
                    {"from_segment": int(ei[0, e]), "to_segment": int(ei[1, e]),
                     "cosine": round(float(ea[e, 1]), 4),
                     "chords": f"{chords[ei[0, e]]} -> {chords[ei[1, e]]}"}
                    for e in strong
                ],
                "text_tokens_attended_by_graph": [
                    {"token": toks[i], "weight": round(float(w[i]), 4)} for i in order
                    if toks[i] not in ("[CLS]", "[SEP]", "[PAD]")
                ],
            })
        if len(rows) >= n:
            break
    dest = results_dir("case_studies") / f"{tag}_case_studies.json"
    save_json(rows, dest)
    LOG.info("saved %d case studies -> %s", len(rows), dest)
    return str(dest)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run(args) -> dict:
    device = get_device()
    set_seed(args.seed)
    modes = FUSION_MODES if args.ablation else [args.fusion]
    tokenizer = make_tokenizer(args.model_name)

    bundle = gd.build_pyg_splits(
        args.dataset, label_mode=args.label_mode, with_text=True, tokenizer=tokenizer,
        max_length=args.max_length, limit=args.limit,
        standardize=not args.no_standardize,
    )
    spec = bundle["spec"]
    multilabel = spec.multilabel

    results: list[dict] = []
    if multilabel:
        ytr = gd.labels_matrix(bundle["splits"]["train"], True)
        yte = gd.labels_matrix(bundle["splits"]["test"], True)
        results.append({"model": "B1_random", "test": random_multilabel_baseline(ytr, yte, args.seed)})
        results.append({"model": "B1_majority", "test": majority_multilabel_baseline(ytr, yte)})
        for r in results:
            LOG.info("baseline %-12s macro-F1 %.4f  AUC-PR %.4f",
                     r["model"], r["test"]["macro_f1"], r["test"]["auc_pr"])

    runs = []
    for mode in modes:
        with Timer(f"task3/{args.dataset}/{mode}"):
            runs.append(train_fusion(mode, bundle, args, device))

    tag = f"task3_{args.dataset}_{'ablation' if args.ablation else args.fusion}"
    artefacts: dict[str, object] = {}
    main_run = next((r for r in runs if r["model"] == "cross_attn"), runs[-1])
    artefacts["tsne"] = tsne_plots(main_run["_test"], bundle, args,
                                   f"task3_{args.dataset}_{main_run['model']}")
    artefacts["case_studies"] = case_studies(main_run, bundle, args,
                                             f"task3_{args.dataset}")

    results += [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    flat = [{"model": r["model"], **r["test"]} for r in results]

    payload = {
        "task": 3,
        "dataset": args.dataset,
        "model": "+".join(modes),
        "multilabel": multilabel,
        "classes": spec.classes,
        "has_valence_arousal": spec.va is not None and not args.no_va,
        "n_train": len(bundle["splits"]["train"]),
        "n_val": len(bundle["splits"]["val"]),
        "n_test": len(bundle["splits"]["test"]),
        "args": vars(args),
        "runs": results,
        "test": main_run["test"],
        "ablation": {r["model"]: r["test"] for r in results},
        "artefacts": artefacts,
    }
    record_result(payload, tag)
    save_json(flat, results_dir("metrics") / f"{tag}_table.json")
    plot_bar_comparison(flat, "macro_f1", results_dir("plots") / f"{tag}_macro_f1.png",
                        title=f"Task 3 / {args.dataset}: macro-F1")
    plot_bar_comparison(flat, "auc_pr", results_dir("plots") / f"{tag}_auc_pr.png",
                        title=f"Task 3 / {args.dataset}: AUC-PR")
    LOG.info("\n%s", _markdown(flat))
    return payload


def _markdown(rows: list[dict]) -> str:
    cols = ["model", "macro_f1", "micro_f1", "auc_pr", "mae_valence", "mae_arousal", "r2_valence"]
    cols = [c for c in cols if any(c in r for r in rows)]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols
        ) + " |")
    return "\n".join(out)


def build_argparser() -> argparse.ArgumentParser:
    c = CFG["task3"]
    ap = argparse.ArgumentParser(description="Task 3: GNN-BERT fusion")
    ap.add_argument("--dataset", default="fma_small", choices=DATASETS)
    ap.add_argument("--fusion", default="cross_attn", choices=FUSION_MODES)
    ap.add_argument("--ablation", action="store_true",
                    help="train all four fusion modes and tabulate")
    ap.add_argument("--label_mode", default="multi", choices=["auto", "single", "multi"])
    ap.add_argument("--model_name", default=CFG["text"]["model_name"])
    ap.add_argument("--epochs", type=int, default=c["epochs"])
    ap.add_argument("--batch_size", type=int, default=c["batch_size"])
    ap.add_argument("--lr_bert", type=float, default=c["lr_bert"])
    ap.add_argument("--lr_other", type=float, default=c["lr_other"])
    ap.add_argument("--weight_decay", type=float, default=c["weight_decay"])
    ap.add_argument("--hidden", type=int, default=c["hidden"])
    ap.add_argument("--gnn_layers", type=int, default=c["gnn_layers"])
    ap.add_argument("--conv", default="sage", choices=["sage", "gat", "gcn"])
    ap.add_argument("--n_heads", type=int, default=c["n_heads"])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=c["alpha"], help="valence loss weight")
    ap.add_argument("--beta", type=float, default=c["beta"], help="arousal loss weight")
    ap.add_argument("--no_va", action="store_true", help="disable the emotion head")
    ap.add_argument("--max_length", type=int, default=CFG["text"]["max_length"])
    ap.add_argument("--freeze_bert", action="store_true")
    ap.add_argument("--n_trainable_layers", type=int, default=None)
    ap.add_argument("--use_edge_attr", action="store_true")
    ap.add_argument("--tune_thresholds", action="store_true", default=True)
    ap.add_argument("--no_tune_thresholds", dest="tune_thresholds", action="store_false")
    ap.add_argument("--no_standardize", action="store_true")
    ap.add_argument("--patience", type=int, default=c["patience"],
                    help="0 disables early stopping")
    ap.add_argument("--min_epochs", type=int, default=c["min_epochs"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()

"""Task 4 (Advanced) -- cross-modal contrastive alignment (MusicCaps).

Learns a shared embedding space between **audio structure graphs** and
**natural-language descriptions** with a symmetric InfoNCE objective
(Algorithm 4):

    L_NCE = -log exp(sim(g_i, t_i)/tau) / sum_j exp(sim(g_i, t_j)/tau)

and evaluates caption->audio / audio->caption retrieval at R@1, R@5, R@10.

    python -m src.train_task4 --dataset musiccaps
    python -m src.train_task4 --dataset deam
    python -m src.train_task4 --dataset musiccaps --zero_shot_tags

Deliverables produced per run
  * retrieval table on the held-out test split        -> results/metrics/
  * 10 qualitative examples (caption -> top-3 clips)  -> results/retrieval_examples/
  * zero-shot tag prediction vs the Task 3 supervised model
  * a listening-test sheet for the human evaluation   -> results/retrieval_examples/
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from . import graph_dataset as gd
from . import text_data
from .bert_encoder import make_tokenizer
from .config import CFG, SEED, results_dir
from .contrastive import DualEncoder, info_nce, retrieval_metrics
from .evaluate import multilabel_metrics, plot_history, record_result
from .utils import (LOG, Timer, count_params, get_device, quiet_transformers,
                    save_json, set_seed)

quiet_transformers()

DATASETS = ["musiccaps", "deam", "mtat", "fma_small", "fma_medium"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def caption_override(dataset: str) -> dict[str, str] | None:
    """MusicCaps pairs a graph with its *caption*, not with the tag table text.

    The processed tag table drops clips whose aspects fall outside the top-50,
    which is irrelevant for contrastive training -- every captioned clip is a
    valid pair -- so captions are taken straight from the raw CSV.
    """
    if dataset != "musiccaps":
        return None
    caps = text_data.load_musiccaps_captions()
    return dict(zip(caps["id"].astype(str), caps["text"].astype(str)))


def build_bundle(args, tokenizer) -> dict:
    bundle = gd.build_pyg_splits(
        args.dataset, label_mode="multi", with_text=True, tokenizer=tokenizer,
        max_length=args.max_length, limit=args.limit,
        standardize=not args.no_standardize,
        include_unlabeled=(args.dataset == "musiccaps"),
    )
    override = caption_override(args.dataset)
    if override:
        _retokenize(bundle, override, tokenizer, args.max_length)
    return bundle


def _retokenize(bundle: dict, texts: dict[str, str], tokenizer, max_length: int) -> None:
    """Replace each graph's tokenised text with the caption for that clip."""
    for split, items in bundle["splits"].items():
        ids = [d.track_id for d in items]
        raw = [texts.get(i, "") for i in ids]
        n_missing = sum(1 for t in raw if not t)
        enc = tokenizer(raw, truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt")
        for n, d in enumerate(items):
            d.input_ids = enc["input_ids"][n].unsqueeze(0)
            d.attention_mask = enc["attention_mask"][n].unsqueeze(0)
        bundle.setdefault("texts", {}).update(dict(zip(ids, raw)))
        if n_missing:
            LOG.warning("%s split: %d clips without a caption", split, n_missing)


# --------------------------------------------------------------------------- #
# embedding / evaluation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def embed_split(model, loader, device) -> dict:
    model.eval()
    G, T, ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch)
        G.append(out["g"].float().cpu())
        T.append(out["t"].float().cpu())
        ids.extend(list(batch.track_id))
    return {"g": torch.cat(G), "t": torch.cat(T), "ids": ids}


@torch.no_grad()
def evaluate_retrieval(model, loader, device) -> tuple[dict, dict]:
    emb = embed_split(model, loader, device)
    metrics = retrieval_metrics(emb["g"], emb["t"])
    return metrics, emb


@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch)
        if len(out["g"]) < 2:
            continue
        loss, _ = info_nce(out["g"].float(), out["t"].float(), out["logit_scale"].float())
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def train(args) -> dict:
    set_seed(args.seed)
    device = get_device()
    tokenizer = make_tokenizer(args.model_name)
    bundle = build_bundle(args, tokenizer)
    spec = bundle["spec"]

    loaders = {
        k: gd.loader(v, args.batch_size, shuffle=(k == "train"),
                     num_workers=args.num_workers)
        for k, v in bundle["splits"].items()
    }

    model = DualEncoder(
        in_dim=bundle["in_dim"], embed_dim=args.embed_dim, hidden=args.hidden,
        gnn_layers=args.gnn_layers, conv=args.conv, dropout=args.dropout,
        model_name=args.model_name, freeze_bert=args.freeze_bert,
        n_trainable_layers=args.n_trainable_layers, temperature=args.temperature,
    ).to(device)
    LOG.info("dual encoder: %.1fM trainable params, embed_dim=%d, batch=%d "
             "(=> %d in-batch negatives)",
             count_params(model) / 1e6, args.embed_dim, args.batch_size,
             args.batch_size - 1)

    bert_params = [p for n, p in model.named_parameters() if p.requires_grad and ".bert." in n]
    other = [p for n, p in model.named_parameters() if p.requires_grad and ".bert." not in n]
    groups = [{"params": other, "lr": args.lr_other}]
    if bert_params:
        groups.append({"params": bert_params, "lr": args.lr_bert})
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, min(args.epochs, 3 * max(args.patience, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best, best_state, stale = [], {"val_r@5": -1.0, "epoch": 0}, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for step, batch in enumerate(loaders["train"], 1):
            batch = batch.to(device)
            if batch.num_graphs < 2:      # InfoNCE needs at least one negative
                continue
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(batch)
            loss, _ = info_nce(out["g"].float(), out["t"].float(),
                               out["logit_scale"].float())
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            run += loss.detach().item() * batch.num_graphs
            n += batch.num_graphs
            if args.log_every and step % args.log_every == 0:
                LOG.info("  epoch %d step %d/%d loss %.4f", epoch, step,
                         len(loaders["train"]), run / max(n, 1))
        sched.step()

        vm, _ = evaluate_retrieval(model, loaders["val"], device)
        vloss = validation_loss(model, loaders["val"], device)
        r5 = vm["caption_to_audio_R@5"]
        rec = {"epoch": epoch, "train_loss": run / max(n, 1), "val_loss": vloss,
               "val_r@1": vm["caption_to_audio_R@1"], "val_r@5": r5,
               "val_r@10": vm["caption_to_audio_R@10"],
               "val_median_rank": vm["caption_to_audio_median_rank"],
               "secs": time.time() - t0}
        history.append(rec)
        LOG.info("epoch %d | train %.4f | val %.4f | R@1 %.4f R@5 %.4f R@10 %.4f | "
                 "med rank %.0f | %.0fs", epoch, rec["train_loss"], vloss,
                 rec["val_r@1"], r5, rec["val_r@10"], rec["val_median_rank"], rec["secs"])

        if r5 > best["val_r@5"]:
            best = {"val_r@5": r5, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if args.patience and epoch >= args.min_epochs and stale >= args.patience:
                LOG.info("early stop at epoch %d (no val R@5 gain for %d epochs)",
                         epoch, stale)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    LOG.info("restored best epoch %d (val R@5 %.4f)", best["epoch"], best["val_r@5"])

    test_metrics, test_emb = evaluate_retrieval(model, loaders["test"], device)
    LOG.info("TEST retrieval | %s",
             "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    # random-retrieval reference: with N candidates, R@K = K/N
    n_test = len(test_emb["ids"])
    random_ref = {f"random_R@{k}": (k / n_test if n_test else float("nan"))
                  for k in (1, 5, 10)}

    tag = f"task4_{args.dataset}_contrastive"
    plot_history(history, results_dir("plots") / f"{tag}_curves.png",
                 title=f"Task 4 / {args.dataset}")
    _plot_retrieval(history, results_dir("plots") / f"{tag}_recall.png", args.dataset)

    examples = qualitative_examples(test_emb, bundle, args, tag, n=args.n_examples)
    listening_sheet(test_emb, bundle, args, tag, n=args.n_examples)

    zero_shot = {}
    if args.zero_shot_tags:
        zero_shot = zero_shot_tagging(model, bundle, loaders, device, args, tag)

    payload = {
        "task": 4,
        "dataset": args.dataset,
        "model": f"contrastive dual-encoder ({args.conv} + {args.model_name})",
        "n_train": len(bundle["splits"]["train"]),
        "n_val": len(bundle["splits"]["val"]),
        "n_test": len(bundle["splits"]["test"]),
        "n_classes": len(spec.classes),
        "embed_dim": args.embed_dim,
        "temperature": args.temperature,
        "args": vars(args),
        "history": history,
        "best_epoch": best["epoch"],
        "test": test_metrics,
        "baselines": {"random_retrieval": random_ref},
        "zero_shot_tags": zero_shot,
        "examples_file": examples,
    }
    record_result(payload, tag)
    if args.save_model:
        torch.save(model.state_dict(), results_dir("checkpoints") / f"{tag}.pt")
    LOG.info("\n%s", _markdown(test_metrics, random_ref))
    return payload


# --------------------------------------------------------------------------- #
# qualitative deliverables
# --------------------------------------------------------------------------- #


def qualitative_examples(emb: dict, bundle: dict, args, tag: str, n: int = 10) -> str:
    """10 query captions -> top-3 retrieved clips (Task 4 deliverable)."""
    g, t, ids = emb["g"], emb["t"], emb["ids"]
    texts = bundle.get("texts", {})
    sim = t @ g.t()                                   # caption -> audio
    order = sim.argsort(dim=1, descending=True)
    rows = []
    picks = np.linspace(0, len(ids) - 1, min(n, len(ids))).astype(int)
    for q in picks:
        gold = ids[q]
        rank = int((order[q] == q).nonzero()[0, 0]) + 1
        rows.append({
            "query_caption": texts.get(gold, "")[:400],
            "gold_clip": gold,
            "gold_rank": rank,
            "hit@1": rank == 1,
            "hit@5": rank <= 5,
            "top3_retrieved": [
                {"clip": ids[int(j)], "similarity": round(float(sim[q, j]), 4),
                 "is_gold": ids[int(j)] == gold,
                 "caption_of_retrieved": texts.get(ids[int(j)], "")[:180]}
                for j in order[q, :3]
            ],
        })
    dest = results_dir("retrieval_examples") / f"{tag}_examples.json"
    save_json(rows, dest)
    LOG.info("saved %d retrieval examples -> %s", len(rows), dest)
    return str(dest)


def listening_sheet(emb: dict, bundle: dict, args, tag: str, n: int = 10) -> str:
    """CSV for the human evaluation: >= 5 listeners rate match on a 1-5 scale."""
    g, t, ids = emb["g"], emb["t"], emb["ids"]
    texts = bundle.get("texts", {})
    sim = t @ g.t()
    top1 = sim.argmax(1)
    picks = np.linspace(0, len(ids) - 1, min(n, len(ids))).astype(int)
    rows = []
    for q in picks:
        retrieved = ids[int(top1[q])]
        rows.append({
            "item": len(rows) + 1,
            "query_caption": texts.get(ids[q], ""),
            "retrieved_clip_id": retrieved,
            "retrieved_is_gold": retrieved == ids[q],
            **{f"listener_{i}_rating_1to5": "" for i in range(1, 6)},
            "notes": "",
        })
    dest = results_dir("retrieval_examples") / f"{tag}_listening_test.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    LOG.info("saved listening-test sheet (%d items x 5 listeners) -> %s", len(rows), dest)
    return str(dest)


# --------------------------------------------------------------------------- #
# zero-shot tagging from the shared space
# --------------------------------------------------------------------------- #

TAG_PROMPT = "a music clip that sounds {tag}"


@torch.no_grad()
def zero_shot_tagging(model, bundle: dict, loaders: dict, device, args, tag: str) -> dict:
    """Score each tag by embedding its name as text and ranking against clips.

    This never sees a tag label during training -- it reuses the contrastive
    space -- so it is directly comparable with the Task 3 *supervised* model.
    """
    spec = bundle["spec"]
    if not spec.classes:
        return {}
    tokenizer = make_tokenizer(args.model_name)
    prompts = [TAG_PROMPT.format(tag=c) for c in spec.classes]
    enc = tokenizer(prompts, truncation=True, padding="max_length",
                    max_length=args.max_length, return_tensors="pt").to(device)

    model.eval()
    tag_emb = model.encode_text(enc["input_ids"], enc["attention_mask"]).float().cpu()

    G, Y = [], []
    for batch in loaders["test"]:
        batch = batch.to(device)
        G.append(model.encode_graph(batch).float().cpu())
        Y.append(batch.y.float().cpu())
    G = torch.cat(G)
    Y = torch.cat(Y).numpy()
    scores = (G @ tag_emb.t()).numpy()
    # map cosine similarities into [0, 1] so the usual thresholded metrics apply
    probs = (scores - scores.min()) / max(float(np.ptp(scores)), 1e-8)

    metrics = multilabel_metrics(Y, probs, thresholds=0.5)
    LOG.info("zero-shot tagging | macro-F1 %.4f  micro-F1 %.4f  AUC-PR %.4f",
             metrics["macro_f1"], metrics["micro_f1"], metrics["auc_pr"])

    supervised = _task3_reference(args.dataset)
    out = {"zero_shot": metrics, "prompt": TAG_PROMPT, "n_tags": len(spec.classes)}
    if supervised:
        out["task3_supervised"] = supervised
        out["delta_macro_f1"] = metrics["macro_f1"] - supervised.get("macro_f1", float("nan"))
        LOG.info("Task 3 supervised macro-F1 %.4f -> zero-shot is %.4f lower",
                 supervised.get("macro_f1", float("nan")), -out["delta_macro_f1"])
    save_json(out, results_dir("zero_shot") / f"{tag}.json")
    return out


def _task3_reference(dataset: str) -> dict:
    """Pull the matching Task 3 supervised test metrics, if that run exists."""
    import json

    for name in (f"task3_{dataset}_ablation.json", f"task3_{dataset}_cross_attn.json"):
        f = results_dir("metrics") / name
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8")).get("test", {})
            except Exception:
                continue
    return {}


# --------------------------------------------------------------------------- #
# plots / tables
# --------------------------------------------------------------------------- #


def _plot_retrieval(history: list[dict], dest, dataset: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lbl in (("val_r@1", "R@1"), ("val_r@5", "R@5"), ("val_r@10", "R@10")):
        ax.plot(ep, [h[k] for h in history], marker="o", label=lbl)
    ax.set_xlabel("epoch")
    ax.set_ylabel("caption -> audio recall")
    ax.set_title(f"Task 4 / {dataset}: retrieval recall vs epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def _markdown(test: dict, random_ref: dict) -> str:
    ks = (1, 5, 10)
    lines = ["| direction | R@1 | R@5 | R@10 | median rank |",
             "| --- | --- | --- | --- | --- |"]
    for d in ("caption_to_audio", "audio_to_caption"):
        cells = [f"{test[f'{d}_R@{k}']:.4f}" for k in ks]
        lines.append(f"| {d.replace('_', ' ')} | " + " | ".join(cells)
                     + f" | {test[f'{d}_median_rank']:.0f} |")
    lines.append("| random reference | "
                 + " | ".join(f"{random_ref[f'random_R@{k}']:.4f}" for k in ks) + " | - |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def build_argparser() -> argparse.ArgumentParser:
    c = CFG["task4"]
    ap = argparse.ArgumentParser(description="Task 4: contrastive GNN-BERT retrieval")
    ap.add_argument("--dataset", default="musiccaps", choices=DATASETS)
    ap.add_argument("--model_name", default=CFG["text"]["model_name"])
    ap.add_argument("--epochs", type=int, default=c["epochs"])
    ap.add_argument("--patience", type=int, default=c["patience"])
    ap.add_argument("--min_epochs", type=int, default=c["min_epochs"])
    ap.add_argument("--batch_size", type=int, default=c["batch_size"],
                    help="also sets the number of in-batch negatives")
    ap.add_argument("--lr_bert", type=float, default=c["lr_bert"])
    ap.add_argument("--lr_other", type=float, default=c["lr_other"])
    ap.add_argument("--weight_decay", type=float, default=c["weight_decay"])
    ap.add_argument("--embed_dim", type=int, default=c["embed_dim"])
    ap.add_argument("--hidden", type=int, default=c["hidden"])
    ap.add_argument("--gnn_layers", type=int, default=c["gnn_layers"])
    ap.add_argument("--conv", default="sage", choices=["sage", "gat", "gcn"])
    ap.add_argument("--temperature", type=float, default=c["temperature"])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max_length", type=int, default=CFG["text"]["max_length"])
    ap.add_argument("--freeze_bert", action="store_true")
    ap.add_argument("--n_trainable_layers", type=int, default=None)
    ap.add_argument("--zero_shot_tags", action="store_true", default=True,
                    help="also evaluate zero-shot tagging against Task 3")
    ap.add_argument("--no_zero_shot_tags", dest="zero_shot_tags", action="store_false")
    ap.add_argument("--n_examples", type=int, default=10)
    ap.add_argument("--no_standardize", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    with Timer(f"task4/{args.dataset}"):
        train(args)


if __name__ == "__main__":
    main()

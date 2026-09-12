"""One-shot preprocessing driver: text tables -> splits -> graphs.

    python -m src.prepare_data --stage all
    python -m src.prepare_data --stage graphs --graph_datasets gtzan deam
"""
from __future__ import annotations

import argparse

from . import graph_builder, prepare_splits, text_data
from .config import CFG
from .utils import LOG, Timer

GRAPH_PRESETS = {
    # dataset        -> (keep mel patches?, note)
    "gtzan": True,
    "deam": True,
    "fma_small": True,
    "fma_medium": True,     # Task 2 compares against the CNN mel baseline here
    "mtat": False,
    "musiccaps": True,
    "deam_feats": False,
}


def stage_text(args) -> None:
    for ds, kw in [
        ("musiccaps", {"top_k": args.top_k, "label_source": args.musiccaps_labels}),
        ("mtat", {"top_k": args.top_k}),
        ("fma", {"subset": "small", "top_k": args.top_k, "label_level": "all"}),
        ("fma", {"subset": "medium", "top_k": args.top_k, "label_level": "all"}),
        ("deam", {"top_k": 20}),
        ("gtzan", {"top_k": 10}),
    ]:
        with Timer(f"text/{ds}{kw.get('subset', '')}"):
            df = text_data.build(ds, **kw)
            text_data.save(df, ds, kw)


def stage_splits(args) -> None:
    for ds in prepare_splits.ALL_DATASETS:
        try:
            sp = prepare_splits.build(ds, args.seed)
        except FileNotFoundError as exc:
            LOG.warning("skipping %s (%s)", ds, exc)
            continue
        prepare_splits.save(sp, ds)
        if ds != "gtzan":
            prepare_splits.check_leakage(ds)


def stage_graphs(args) -> None:
    for ds in args.graph_datasets:
        with Timer(f"graphs/{ds}"):
            graph_builder.run(ds, jobs=args.jobs, limit=args.limit,
                              want_mel=GRAPH_PRESETS.get(ds, True) and not args.no_mel,
                              overwrite=args.overwrite)


def main() -> None:
    ap = argparse.ArgumentParser(description="build every preprocessing artefact")
    ap.add_argument("--stage", default="all", choices=["all", "text", "splits", "graphs"])
    ap.add_argument("--graph_datasets", nargs="*",
                    default=["gtzan", "deam", "musiccaps", "fma_small",
                             "mtat", "fma_medium"])
    ap.add_argument("--top_k", type=int, default=CFG["task1"]["top_k_tags"])
    ap.add_argument("--musiccaps_labels", default="aspects", choices=["aspects", "audioset"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_mel", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=CFG.get("seed", 42))
    args = ap.parse_args()

    if args.stage in ("all", "text"):
        stage_text(args)
    if args.stage in ("all", "splits"):
        stage_splits(args)
    if args.stage in ("all", "graphs"):
        stage_graphs(args)
    LOG.info("preprocessing complete")


if __name__ == "__main__":
    main()

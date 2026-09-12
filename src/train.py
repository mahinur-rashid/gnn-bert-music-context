"""Single entry point for every training task.

    python -m src.train --task 1 --dataset musiccaps
    python -m src.train --task 2 --dataset gtzan --compare
    python -m src.train --task 3 --dataset fma_small --ablation
    python -m src.train --task compare            # cross-dataset comparison

Any flag after ``--`` is forwarded verbatim to the underlying task script, e.g.

    python -m src.train --task 2 -- --dataset fma_small --epochs 40 --compare
"""
from __future__ import annotations

import runpy
import sys

TASK_MODULES = {
    "1": "src.train_task1",
    "2": "src.train_task2",
    "3": "src.train_task3",
    "4": "src.contrastive",
    "compare": "src.compare_datasets",
    "prepare": "src.prepare_data",
}

USAGE = f"""usage: python -m src.train --task {{{'|'.join(TASK_MODULES)}}} [task arguments...]

  --task 1        BERT multi-label tag classifier          (src/train_task1.py)
  --task 2        GNN on music structure graphs            (src/train_task2.py)
  --task 3        GNN-BERT fusion                          (src/train_task3.py)
  --task 4        contrastive MusicCaps retrieval          (src/contrastive.py)
  --task compare  cross-dataset comparison for tasks 1-3   (src/compare_datasets.py)
  --task prepare  build text tables, splits and graphs     (src/prepare_data.py)
"""


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("--task", "-t"):
        print(USAGE)
        raise SystemExit(2)
    if len(argv) < 2 or argv[1] not in TASK_MODULES:
        print(USAGE)
        raise SystemExit(2)

    module = TASK_MODULES[argv[1]]
    rest = argv[2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    sys.argv = [module] + rest
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()

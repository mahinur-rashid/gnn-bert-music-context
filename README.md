# GNN-Based BERT for Understanding Context from Music

Supervised neural-network project for **CSE715**.
A hybrid **BERT + Graph Neural Network** system that predicts musical context —
multi-label genre/mood tags and valence/arousal — by combining contextual language
representations with message passing over music-structure graphs.

**Status: all four tasks implemented, run and reported.**

| Task | Model | Result headline |
|---|---|---|
| 1 (Easy) | BERT multi-label tag classifier | 0.695 macro-F1 on MusicCaps, 4 datasets compared |
| 2 (Medium) | GraphSAGE / GAT / GCN on segment+chord graphs | GAT 0.790 macro-F1 on GTZAN, beats CNN baseline |
| 3 (Hard) | GNN–BERT fusion, 4-way ablation | 0.429 macro-F1 + valence R² 0.52 on DEAM |
| 4 (Advanced) | Contrastive dual encoder (InfoNCE) | R@5 = 0.193 on MusicCaps, 20× random |

Final report: [`report/CSE715_Final_Report.pdf`](report/CSE715_Final_Report.pdf)
(10 pages, NeurIPS format). Full numbers in [`results/`](results/); coverage
against the brief in [COVERAGE.md](COVERAGE.md).

---

## 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# CUDA build of torch (adjust cu126 to your driver):
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install torch-geometric
```

Verified on Windows 11 / Python 3.12 / PyTorch 2.10 + CUDA 12.6 / PyG 2.8 /
transformers 5.17 / librosa 0.10, on a single RTX 4070 Laptop GPU.
`ffmpeg` must be on `PATH` for mp3 decoding.

## 2. Datasets

Place the datasets under `Datasets/` as below (all paths configurable in
`config.yaml`):

```
Datasets/
  fma/
    fma_small/      8'000 x 30 s mp3        fma_medium/  25'000 x 30 s mp3
    fma_metadata/   tracks.csv, genres.csv, ...
  gtzan/
    genres_original/<genre>/*.wav           features_30_sec.csv
  MagnaTagATune Dataset/
    MagnaTagATune/<0-f>/*.mp3               annotations_final.csv
  Deam_Dataset/
    MEMD_audio/*.mp3   features/*.csv   annotations/...   metadata/*.csv
  musiccaps/
    musiccaps-public.csv
  musiccaps_with_audio/
    wav/[<ytid>]-[<start>-<end>].wav        5'161 x 10 s clips
    musiccaps-downloaded.csv  download_manifest.csv  download_failures.csv
```

| Dataset | Used for | Labels | Graphs built |
|---|---|---|---|
| MusicCaps | Tasks 1, 4 | top-50 caption aspects; free-text captions | 5,161 |
| MagnaTagATune | Tasks 1, 2, 3 | top-50 tags after synonym merging | 21,358 |
| FMA-small | Tasks 1, 2, 3 | 8 top genres / top-50 `genres_all` | 7,997 |
| GTZAN | Tasks 1, 2 | 10 genres | 999 |
| DEAM | Tasks 1, 3, 4 | 20 genre tags + valence/arousal | 1,795 |
| FMA-medium | optional | 16 top genres | 24,440 |

**Two caveats that matter when reading the results.**

*GTZAN has no metadata.* Its only text is the filename, which **is** the label
(`blues.00000.wav`). Using it would leak the target, so `text_data.build_gtzan`
generates the text from the hand-crafted descriptors in `features_30_sec.csv`
(tempo, brightness, loudness, percussiveness, pitch-class spread). That text is
machine-generated, so GTZAN's Task 1 score is **not comparable** to MusicCaps'
human captions.

*MusicCaps aspects are quoted in the captions.* The top-50 aspect phrases used as
Task 1 labels appear near-verbatim in the caption text — the brief itself calls
this a "caption → tag proxy task". Its high score measures extraction, not
inference.

## 3. Running everything

```powershell
python -m src.prepare_data --stage all --jobs 10   # text, splits, graphs (~45 min)
.\scripts\run_all_tasks.ps1                        # tasks 1-4 + comparisons (~3.4 h)
```

`run_all_tasks.ps1` runs seven steps in dependency order (Task 4 after Task 3, so
zero-shot tagging can cite the supervised result), continues past a failing step,
logs to `results\run_all.log` and prints a status table. `-Quick` does a 5-minute
smoke run of every step; `-Task2Epochs 60` gives the CNN baseline a fair schedule
(see §6).

### 3.1 Preprocessing

```powershell
python -m src.text_data --dataset all              # text/label tables
python -m src.prepare_splits --dataset all         # official or artist-grouped splits
python -m src.graph_builder --dataset gtzan     --jobs 10
python -m src.graph_builder --dataset deam      --jobs 10
python -m src.graph_builder --dataset musiccaps --jobs 10
python -m src.graph_builder --dataset fma_small --jobs 10
python -m src.graph_builder --dataset mtat      --jobs 10 --no_mel
python -m src.graph_builder --dataset fma_medium --jobs 10   # optional, ~35 min
```

**Audio front-end** (`src/audio_features.py`): 22'050 Hz mono → STFT → log-mel
(128) / MFCC (20) / chroma-CQT (12) / spectral contrast (7) / centroid, bandwidth,
rolloff, ZCR, RMS. Tracks are cut into 3 s windows with 1.5 s hop over the first
30 s (≤ 24 segments). MusicCaps clips are only 10 s, so
`config.yaml → audio.per_dataset.musiccaps` shortens them to a 2 s / 1 s window
(12-segment cap). `--segmentation beat` switches to beat-synchronous segments.

**Graph construction** (`src/graph_builder.py`): nodes = segments, node feature =
89-dim `[mfcc_mean(20) | mfcc_std(20) | chroma(12) | contrast(7) | 5 scalars |
chord one-hot(25)]`. Three edge families merge into one graph with edge attributes
`[is_temporal, cosine_similarity, chord_transition]`:

1. temporal adjacency `i → i+1 … i+r` (r = 2), weight `1/d`;
2. similarity edges — top-`k` (5) neighbours with cosine similarity > τ (0.7);
3. chord-transition edges weighted by observed transition counts (chords from
   cosine matching against 24 major/minor triad templates + a "no chord" symbol).

Graphs average 19 nodes / 133 edges for 30 s tracks and 10 nodes / 50 edges for
MusicCaps. At least 20 example graphs per dataset are exported as both `.pt` and
readable `.json` under `data/processed/graph_samples/`.

**Splits**: FMA and MagnaTagATune use their **official** splits; MusicCaps and
DEAM use a seeded 80/10/10 split **grouped by artist**; GTZAN is stratified by
genre. `data/splits/leakage_report.json` records the audit. Note the standard MTAT
directory split is not fully artist-disjoint (57 of 229 artists straddle splits);
`prepare_splits --artist_safe` gives a strictly grouped alternative.

### 3.2 Individual tasks

```powershell
# Task 1 — BERT tag classifier
python -m src.train_task1 --dataset musiccaps --attention_viz

# Task 2 — GNN vs CNN / PCA-MLP / random baselines
python -m src.train_task2 --dataset gtzan --compare --pca_mlp

# Task 3 — fusion with 4-way ablation
python -m src.train_task3 --dataset deam --ablation --save_model

# Task 4 — contrastive retrieval + zero-shot tagging
python -m src.train_task4 --dataset musiccaps --save_model
```

### 3.3 Cross-dataset comparisons

```powershell
python -m src.compare_datasets --task 1   # fma_small, mtat, gtzan, musiccaps
python -m src.compare_datasets --task 2   # gtzan, fma_small
python -m src.compare_datasets --task 3   # fma_small, mtat, deam
python -m src.compare_datasets --task 4   # deam, musiccaps
python -m src.evaluate --summarize        # -> results/summary.md
```

Override any line-up with `--datasets a b c`.

## 4. Results

Full tables in `results/comparison/`; per-experiment JSON in `results/metrics/`.

### Task 1 — text only (macro-F1)

| dataset | train | val | **test** | micro-F1 | AUC-PR | random |
|---|---|---|---|---|---|---|
| MusicCaps † | 0.914 | 0.747 | **0.695** | 0.736 | 0.731 | 0.053 |
| GTZAN ‡ | 0.536 | 0.554 | **0.480** | 0.451 | 0.523 | 0.148 |
| MagnaTagATune | 0.563 | 0.350 | **0.304** | 0.392 | 0.287 | 0.064 |
| FMA-small | 0.914 | 0.430 | **0.267** | 0.325 | 0.326 | 0.039 |

† proxy task (aspects quoted in caption)  ‡ generated text, not human-written

### Task 2 — audio graphs only (test)

| model | GTZAN acc | GTZAN macro-F1 | FMA-small acc | FMA-small macro-F1 |
|---|---|---|---|---|
| B1 random | 0.110 | 0.112 | 0.151 | 0.151 |
| B1 majority | 0.100 | 0.018 | 0.125 | 0.028 |
| B4 PCA+MLP | 0.750 | 0.755 | 0.378 | 0.375 |
| B2 CNN mel | 0.610 | 0.597 | 0.406 | **0.410** |
| GraphSAGE | 0.730 | 0.723 | 0.400 | 0.392 |
| **GAT** | 0.800 | **0.790** | 0.395 | 0.388 |
| GCN | 0.750 | 0.753 | 0.389 | 0.380 |

### Task 3 — fusion ablation (test macro-F1)

| fusion | FMA-small | MagnaTagATune | DEAM |
|---|---|---|---|
| B1 random | 0.039 | 0.064 | 0.085 |
| `gnn_only` | 0.183 | **0.390** | 0.181 |
| `bert_only` | 0.258 | 0.310 | 0.282 |
| `concat` | **0.318** | 0.349 | 0.299 |
| `cross_attn` | 0.205 | 0.352 | **0.429** |

DEAM emotion regression: valence R² is **0.521** with cross-attention, 0.397 with
GNN-only, and **−0.129** with BERT-only — a text-only model is worse than
predicting the mean.

### Task 4 — contrastive retrieval (test)

| dataset | direction | R@1 | R@5 | R@10 | median rank | random R@5 |
|---|---|---|---|---|---|---|
| MusicCaps | caption → audio | 0.066 | **0.193** | 0.300 | 25 / 513 | 0.010 |
| MusicCaps | audio → caption | 0.049 | 0.187 | 0.298 | 25 / 513 | 0.010 |
| DEAM | caption → audio | 0.000 | 0.051 | 0.107 | 68 / 177 | 0.028 |

Zero-shot tagging from the contrastive space reaches 0.129 macro-F1 on DEAM
against 0.429 for the Task 3 supervised model.

### The headline finding

The value of music-structure graphs is **dataset-dependent and predictable**:
graphs help exactly when the labels describe structure, and text helps exactly
when it describes sound rather than identity. Cross-attention wins on DEAM, early
concatenation wins on FMA-small, and a **text-free GNN wins on MagnaTagATune**,
whose tags are instrument and texture descriptors. This is why the project reports
a cross-dataset comparison rather than a single number.

## 5. Repository layout

```
gnn-bert-music-context/
  README.md  COVERAGE.md  requirements.txt  config.yaml
  data/
    processed/{text,graphs,graph_samples}/   splits/
  notebooks/eda.ipynb  notebooks/demo_context.ipynb
  src/
    config.py utils.py
    text_data.py            # captions/tags/metadata -> text + label tables
    prepare_splits.py       # official / artist-grouped splits
    prepare_data.py         # one-shot preprocessing driver
    audio_features.py       # mel, chroma, MFCC, segmentation, chord estimation
    graph_builder.py        # chord + segment graphs
    graph_dataset.py        # PyG datasets (graph [+ text] + labels)
    bert_encoder.py         # BERT encoder + Task 1 classifier
    gnn_model.py            # GraphSAGE / GAT / GCN + MelCNN baseline
    fusion_model.py         # cross-attention GNN-BERT fusion
    contrastive.py          # Task 4 dual encoder + InfoNCE + retrieval metrics
    baselines.py            # B1 random/majority, B4 PCA+MLP
    train.py                # dispatcher
    train_task1.py train_task2.py train_task3.py train_task4.py
    compare_datasets.py     # cross-dataset tables + plots
    evaluate.py             # metrics, plots, summary CLI
  scripts/
    run_all_tasks.ps1       # tasks 1-4 end to end
    run_all.ps1             # preprocessing + tasks
  results/
    metrics/ plots/ examples/ case_studies/ comparison/
    retrieval_examples/ zero_shot/ checkpoints/
    summary.md  metrics.json  run_all.log
  report/CSE715_Final_Report.pdf
```

`results/_archive_pre_earlystop/` holds superseded results from before the
training schedule changed; it is not part of the submission.

## 6. Training schedule

Every task trains for up to `epochs: 20` and stops once the validation metric has
not improved for `patience: 4` epochs; `min_epochs` blocks early stopping during
the initial noisy phase. **The best-validation checkpoint is always restored
before test evaluation**, per-tag thresholds are tuned on validation only, and
test data is never used for model selection.

Early-stopping metric per task: macro-F1 (1), accuracy or macro-F1 (2), macro-F1
(3), caption→audio R@5 (4).

> **Known bias in the Task 2 baseline comparison.** The CNN mel baseline is the
> slowest learner in the set — on GTZAN it was still climbing at epoch 45 and
> reached 0.801 macro-F1 under a 60-epoch budget, versus 0.597 under the 20-epoch
> budget used for the headline table. The GTZAN graph-vs-CNN margin is therefore
> inflated. Run `.\scripts\run_all_tasks.ps1 -Task2Epochs 60` for the fair
> comparison, which narrows it to roughly GAT 0.838 vs CNN 0.801.

## 7. Metrics

Per-tag precision/recall/F1 → **macro-F1** (mean over tags) and **micro-F1**
(pooled); **AUC-PR** = mean average precision over tags; emotion regression reports
**MAE**, RMSE and **R²** for valence and arousal; retrieval reports **R@1/5/10**
and median rank in both directions against a `K/N` random reference.

The optional graph-coherence score `S_graph` is implemented and reported, but it
returns 0.996–1.000 on every dataset and fusion mode, so it discriminates nothing
as specified. The likely cause is that node embeddings are read after
BatchNorm + ReLU, which makes activations non-negative and inflates pairwise
cosine similarity. Treated as a negative result rather than as evidence.

## 8. Reproducibility

Every run is seeded (`config.yaml: seed: 42`) and records its full argument
namespace, per-epoch history, restored best epoch, tuned thresholds, test metrics
and baselines in `results/metrics/<tag>.json`.

Single-seed caveat: DEAM was trained twice under identical settings and produced
0.414 and 0.429 test macro-F1. With 177 DEAM test clips (and 100 for GTZAN),
differences below ~0.02 should not be interpreted.

## 9. Outstanding

* **Listening-test ratings.** Task 4 generates
  `results/retrieval_examples/*_listening_test.csv` with 10 retrieval pairs and 5
  empty rating columns, as the brief's human evaluation requires. It needs ≥ 5
  human listeners to fill in; the report states that no human scores are reported
  yet.

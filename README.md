# GNN-Based BERT for Understanding Context from Music

Supervised neural-network project for **CSE425 / EEE474 / CSE715**.
Hybrid **BERT + Graph Neural Network** system that predicts musical context —
multi-label genre/mood tags and valence/arousal — by combining contextual
language representations with message passing over music-structure graphs.

| Task | Model | Status |
|---|---|---|
| 1 (Easy) | BERT multi-label tag classifier | ✅ implemented + run |
| 2 (Medium) | GraphSAGE / GAT on segment+chord graphs | ✅ implemented + run |
| 3 (Hard) | GNN–BERT fusion (cross-attention, multi-task) | ✅ implemented + run |
| 4 (Advanced) | Contrastive dual-encoder retrieval (InfoNCE) | ✅ implemented |

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
transformers 5.17 / librosa 0.10, RTX 4070 Laptop.
`ffmpeg` must be on `PATH` for mp3 decoding.

## 2. Datasets

Place the datasets under `Datasets/` exactly as below (paths are configurable in
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
  musiccaps_with_audio/                     (needed for Task 4)
    wav/[<ytid>]-[<start>-<end>].wav        5'161 x 10 s clips
    musiccaps-downloaded.csv  download_manifest.csv  download_failures.csv
```

| Dataset | Used for | Labels |
|---|---|---|
| MusicCaps | Tasks 1, 3, 4 | top-50 caption aspects; free-text captions for retrieval |
| MagnaTagATune | Tasks 1, 2, 3 | top-50 tags after synonym merging, 25'863 clips |
| FMA small | Tasks 1, 2, 3 | 8 top genres, top-50 `genres_all` (medium is available but unused by default) |
| GTZAN | Tasks 1, 2 | 10 genres, 1'000 clips |
| DEAM | Tasks 1–4 | 20 genre tags + valence/arousal (1–9 → [-1,1]) |

## 3. Pipeline

### 3.1 Preprocessing

```powershell
# text tables + label vocabularies (data/processed/text/)
python -m src.text_data --dataset all

# train/val/test splits (data/splits/), official where they exist
python -m src.prepare_splits --dataset all

# music-structure graphs (data/processed/graphs/)
python -m src.graph_builder --dataset gtzan     --jobs 10
python -m src.graph_builder --dataset fma_small --jobs 10
python -m src.graph_builder --dataset deam      --jobs 10
python -m src.graph_builder --dataset musiccaps --jobs 10
python -m src.graph_builder --dataset mtat      --jobs 10 --no_mel
# optional -- no comparison uses it, every FMA experiment runs on fma_small
python -m src.graph_builder --dataset fma_medium --jobs 10            # ~35 min

# or all of the above in one call
python -m src.prepare_data --stage all --jobs 10
```

MusicCaps clips are only 10 s long, so `config.yaml → audio.per_dataset.musiccaps`
shortens their window/hop to 2 s / 1 s (12 nodes instead of 24).

**Audio front-end** (`src/audio_features.py`): 22'050 Hz mono → STFT →
log-mel (128) / MFCC (20) / chroma-CQT (12) / spectral contrast (7) / centroid,
bandwidth, rolloff, ZCR, RMS. Tracks are cut into 3 s windows with 1.5 s hop
(≤ 24 segments, first 30 s); `--segmentation beat` switches to beat-synchronous
segments from `librosa.beat.beat_track`.

**Graph construction** (`src/graph_builder.py`): nodes = segments, node feature
= 89-dim `[mfcc_mean | mfcc_std | chroma | contrast | 5 scalars | chord one-hot(25)]`.
Three edge families are merged into one graph with edge attributes
`[is_temporal, cosine_similarity, chord_transition]`:

1. temporal adjacency `i → i+1 … i+r` (r = 2), weight `1/d`;
2. similarity edges — top-`k` (5) neighbours with cosine similarity > τ (0.7);
3. chord-transition edges from the estimated chord sequence, weighted by
   observed transition counts (chords from cosine matching against 24 major/minor
   triad templates + a "no chord" symbol).

`data/processed/graph_samples/<dataset>/` holds ≥ 20 example graphs as both
`.pt` tensors and human-readable `.json` (deliverable #2).

**Splits**: FMA and MagnaTagATune use their **official** splits; MusicCaps and
DEAM use a seeded 80/10/10 split **grouped by artist** (no artist leakage).
`data/splits/leakage_report.json` records how many artists straddle splits —
note the standard MTAT directory split is not fully artist-disjoint; pass
`--artist_safe` to `prepare_splits` for a strictly artist-grouped alternative.

### 3.2 Task 1 — BERT tag classifier

```powershell
python -m src.train_task1 --dataset musiccaps --attention_viz
python -m src.train_task1 --dataset mtat
python -m src.train_task1 --dataset fma_small
python -m src.train_task1 --dataset gtzan
python -m src.train_task1 --dataset all
```

`t = BERT_CLS(X_text)`, `ŷ_k = σ(w_kᵀt + b_k)`, BCE per tag, AdamW with a cosine
schedule, fp16. Per-tag decision thresholds are tuned on validation. Outputs:
macro/micro F1 + AUC-PR curves vs epoch, 5 example predictions with `[CLS]`
attention (`--attention_viz`), and the B1 random/majority baselines.

> **GTZAN and Task 1.** GTZAN ships no metadata: the only text attached to a clip
> is its filename, which *is* the label (`blues.00000.wav`). Using it would leak
> the target, so `text_data.build_gtzan` writes the textual context from the
> hand-crafted descriptors in `features_30_sec.csv` (tempo, brightness, loudness,
> percussiveness, pitch-class spread). That text is **generated, not
> human-written** — the single most important caveat when reading GTZAN's Task 1
> numbers.


### 3.3 Task 2 — GNN on music structure graphs

```powershell
python -m src.train_task2 --dataset gtzan     --compare --pca_mlp
python -m src.train_task2 --dataset fma_small --compare
python -m src.train_task2 --dataset deam      --compare
```

`--compare` trains GraphSAGE, GAT, GCN **and** the CNN mel-spectrogram baseline
(B2), alongside B1 random/majority and optional B4 PCA+MLP, then writes a
comparison table, bar charts and a confusion matrix.

### 3.4 Task 3 — GNN–BERT fusion

```powershell
python -m src.train_task3 --dataset fma_small --ablation
python -m src.train_task3 --dataset mtat      --ablation
python -m src.train_task3 --dataset deam      --ablation
```

`--ablation` runs all four fusion modes — `bert_only`, `gnn_only`, `concat`
(early concatenation) and `cross_attn` (graph readout attends over BERT token
states). Loss: `L = L_tags + α‖v−v̂‖² + β‖a−â‖²`, with the emotion term active
only on DEAM. Produces t-SNE of `z` coloured by genre and by mood quadrant,
3 case studies (chord path + strongest similarity edges + text tokens the graph
attends to) and the graph coherence score `S_graph`.

### 3.5 Task 4 — contrastive cross-modal retrieval

```powershell
python -m src.train_task4 --dataset musiccaps --save_model
python -m src.train_task4 --dataset deam
```

A dual encoder (GraphSAGE over the audio graph, BERT over the caption) is trained
with symmetric InfoNCE; every other item in the batch is a negative, so
`--batch_size` sets both the difficulty of the contrastive problem and the
resolution of R@K. Outputs: the retrieval table (caption→audio and audio→caption
R@1/5/10 + median rank, against a `K/N` random reference), 10 qualitative
caption→top-3-clip examples, a 5-listener listening-test sheet for the human
evaluation, and zero-shot tag prediction scored against the Task 3 supervised
model on the same tags.

### 3.6 Cross-dataset comparison

```powershell
python -m src.compare_datasets --task 1     # fma_small, magnatagatune, gtzan, musiccaps
python -m src.compare_datasets --task 2     # gtzan, fma_small
python -m src.compare_datasets --task 3     # fma_small, magnatagatune, deam
python -m src.compare_datasets --task 4     # deam, musiccaps (with audio)
python -m src.compare_datasets --task all
```

Runs one task across every applicable dataset with identical hyper-parameters and
writes `results/comparison/task<N>_dataset_comparison.{md,json,png}` plus a
train/val/test macro-F1 chart per dataset. Override the line-up with
`--datasets a b c`.

### 3.7 Aggregating results

```powershell
python -m src.evaluate --summarize     # -> results/metrics.json + results/summary.md
```

## 4. Repository layout

```
gnn-bert-music-context/
  README.md  requirements.txt  config.yaml
  data/
    processed/{text,graphs,graph_samples}/    splits/
  notebooks/eda.ipynb  notebooks/demo_context.ipynb
  src/
    config.py utils.py
    text_data.py            # captions/tags/metadata -> text + label tables
    prepare_splits.py       # official / artist-grouped splits
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
    prepare_data.py         # one-shot preprocessing
    evaluate.py             # metrics, plots, summary CLI
  results/
    metrics/ plots/ examples/ case_studies/ comparison/ metrics.json summary.md
  report/
```

## 5. Metrics

Per-tag precision/recall/F1 → **macro-F1** (mean over tags) and **micro-F1**
(pooled); **AUC-PR** = mean average precision over tags; emotion regression
reports **MAE**, RMSE and **R²** for valence and arousal; `S_graph` measures the
fraction of graph edges whose learnt node embeddings have cosine similarity > τ.

Retrieval (Task 4) reports **R@1/5/10** and median rank in both directions,
against the `K/N` random reference for the test-set size.

## 6. Training schedule

Every task trains for up to `epochs: 100` and stops once the validation metric
has not improved for `patience: 10` epochs; `min_epochs` blocks early stopping
during the initial noisy phase. This matters for the fairness of the baseline
comparison: the CNN mel baseline dips for ~20 epochs mid-training before
recovering, and stopping inside that dip understated it by ~0.17 macro-F1 on
GTZAN in an earlier run, so `task2.min_epochs` is set to 25. The
best-validation checkpoint is always restored before test evaluation, and every
model in a comparison gets the same `epochs`, `patience` and `min_epochs`.

The early-stopping model selection metric per task: macro-F1 (Task 1), macro-F1
or accuracy (Task 2), macro-F1 (Task 3), caption→audio R@5 (Task 4).

## 7. Reproducibility

Every run is seeded (`config.yaml: seed: 42`), records its full argument
namespace in `results/metrics/<tag>.json`, and restores the best-validation
checkpoint before test evaluation. Test metrics are never used for model
selection or threshold tuning.

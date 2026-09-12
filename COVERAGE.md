# Coverage check against `Instructions.pdf`

Every requirement in the brief, and where it is implemented. All four tasks are
implemented; only the final report is still outstanding.

## Section 3 — Dataset requirements

| Requirement | Status | Where |
|---|---|---|
| ≥ 1 primary audio dataset | ✅ | FMA-small/medium, GTZAN, MagnaTagATune, DEAM |
| ≥ 1 text/tag dataset | ✅ | MusicCaps captions, MTAT tags, FMA metadata, DEAM last.fm tags |
| Advanced pairing (FMA/MTAT × MusicCaps/DEAM) | ✅ | Task 3 runs on FMA + MTAT + DEAM (valence/arousal aux loss) |
| Resample to 22 050 Hz | ✅ | `audio_features.load_audio` |
| log-mel 128 bins / chroma 12 bins | ✅ | `audio_features.frame_features` |
| Normalise per track | ✅ | waveform peak-normalised per track; node features z-scored with **train-split** statistics (`graph_dataset.fit_feature_stats`) |
| Fixed-window segmentation | ✅ | `segment_bounds` (default 3 s / 1.5 s hop, `config.yaml → audio.win_s`) |
| Beat-synchronous segmentation | ✅ | `beat_segment_bounds`, `--segmentation beat` |
| Chord-transition graph (nodes = chords, edges weighted by count) | ✅ | `graph_builder.build_edges` step 3 + `estimate_chords` (24 triad templates + N) |
| Segment graph (temporal adjacency + cosine sim > τ) | ✅ | `graph_builder.build_edges` steps 1–2 |
| BERT tokenizer, max length 128–256 | ✅ | 192, `config.yaml → text.max_length` |
| Official FMA / MTAT splits | ✅ | `prepare_splits.official_split` |
| DEAM standard train/val partition | ✅ | seeded 80/10/10 grouped by artist |
| No artist leakage | ✅ | `grouped_split` + `check_leakage` → `data/splits/leakage_report.json` |

> Note: the *standard* MagnaTagATune directory split (0–b / c / d–f) is not fully
> artist-disjoint — 57 of 229 artists straddle splits. It is used as-is because the
> brief asks for the official split; `prepare_splits --artist_safe` produces a
> strictly artist-grouped alternative.

## Section 4.1 — Task 1 (Easy)

| Deliverable | Status | Where |
|---|---|---|
| BERT fine-tuning code (`bert-base-uncased` / `distilbert-base-uncased`) | ✅ | `bert_encoder.py`, `train_task1.py` (`--model_name`) |
| `t = BERT_CLS(X)`, `ŷ = σ(Wt+b)`, BCE per tag | ✅ | `BertTagClassifier`, `nn.BCEWithLogitsLoss` |
| Results on MagnaTagATune top-50 tags | ✅ | `results/metrics/task1_mtat_*.json` |
| Results on MusicCaps caption → tag proxy | ✅ | `results/metrics/task1_musiccaps_*.json` |
| Macro-F1 / Micro-F1 curves vs epochs | ✅ | `results/plots/task1_*_curves.png` |
| 5 example predictions + attention visualisation | ✅ | `results/examples/*.json`, `--attention_viz` |

## Section 4.2 — Task 2 (Medium)

| Deliverable | Status | Where |
|---|---|---|
| Graph construction scripts (chroma/MFCC segment graphs) | ✅ | `audio_features.py`, `graph_builder.py` |
| GraphSAGE / GAT encoder, PyTorch Geometric | ✅ | `gnn_model.GNNEncoder` (`sage` / `gat` / `gcn`) |
| Mean-pooling graph readout | ✅ | `global_mean_pool` ⊕ `global_max_pool` |
| Genre classification on GTZAN or FMA-small | ✅ | both |
| Comparison vs CNN baseline on mel-spectrogram | ✅ | `gnn_model.MelCNN`, `train_task2 --compare` |

## Section 4.3 — Task 3 (Hard)

| Deliverable | Status | Where |
|---|---|---|
| End-to-end GNN–BERT fusion model | ✅ | `fusion_model.FusionModel` |
| Cross-attention fusion (Q=gW_Q, K=H_text W_K) | ✅ | `CrossAttentionFusion` |
| Multi-task loss `L_tags + α‖v−v̂‖² + β‖a−â‖²` | ✅ | `multitask_loss` (α, β in `config.yaml`) |
| Ablation: BERT-only / GNN-only / early concat / cross-attention | ✅ | `--ablation` (4 modes) |
| Results on FMA-medium or MagnaTagATune (macro-F1, AUC-PR) | ✅ | MTAT + FMA-small (+ FMA-medium once its graphs are built) |
| t-SNE of z coloured by genre **and** mood | ✅ | `tsne_plots` → `*_tsne_genre.png`, `*_tsne_mood.png` |
| 3 case studies: graph paths + caption/lyric alignment | ✅ | `case_studies` → `results/case_studies/*.json` |

## Section 4.4 — Task 4 (Advanced)

| Deliverable | Status | Where |
|---|---|---|
| Dual-encoder GNN–BERT with contrastive training | ✅ | `contrastive.DualEncoder`, `train_task4.train` |
| InfoNCE loss (symmetric, in-batch negatives) | ✅ | `contrastive.info_nce` |
| Retrieval metrics R@1/5/10 + median rank, both directions | ✅ | `contrastive.retrieval_metrics` |
| Retrieval evaluation table on the MusicCaps test split | ✅ | `results/metrics/task4_musiccaps_contrastive.json` |
| 10 qualitative examples (query caption → top-3 clips) | ✅ | `results/retrieval_examples/*_examples.json` |
| Zero-shot tag prediction from captions vs the Task 3 supervised model | ✅ | `train_task4.zero_shot_tagging` → `results/zero_shot/` |
| Human evaluation: ≥ 5 listeners rate match on [1,5] | ✅ sheet generated | `results/retrieval_examples/*_listening_test.csv` (ratings to be filled in by listeners) |

## Section 6 — Evaluation metrics

| Metric | Status | Where |
|---|---|---|
| Per-tag precision / recall / F1 | ✅ | `evaluate.multilabel_metrics` |
| Macro-F1, Micro-F1 | ✅ | same |
| AUC-PR (mean average precision over tags) | ✅ | same |
| MAE and R² for valence/arousal | ✅ | `evaluate.regression_metrics` |
| Graph coherence score `S_graph` | ✅ | `evaluate.graph_coherence`, reported by Task 3 |

## Section 7 — Algorithms

| Algorithm | Status | Where |
|---|---|---|
| 1 — BERT multi-label tag classifier | ✅ | `train_task1.train_one` |
| 2 — GNN encoder on music segment graph | ✅ | `graph_builder` + `train_task2.train_model` |
| 3 — GNN–BERT fusion for context understanding | ✅ | `train_task3.train_fusion` |
| 4 — contrastive GNN–BERT | ✅ | `contrastive.py` + `train_task4.py` |

## Section 8 — Baselines (≥ 2 required, all 4 implemented)

| Baseline | Status | Where |
|---|---|---|
| B1 majority-class / random tag predictor | ✅ | `baselines.random_*`, `baselines.majority_*` |
| B2 CNN on mel-spectrogram | ✅ | `gnn_model.MelCNN` |
| B3 BERT-only | ✅ | Task 1, and the `bert_only` ablation of Task 3 |
| B4 PCA + MLP on hand-crafted features | ✅ | `baselines.pca_mlp_baseline`, `--pca_mlp` |

## Section 10 — Submission requirements

| Item | Status | Where |
|---|---|---|
| Full source code | ✅ | `src/` |
| ≥ 20 preprocessed graph samples (`.pt` / `.json`) | ✅ | `data/processed/graph_samples/<dataset>/` (20 per dataset) |
| Evaluation tables + plots (F1, AUC-PR, t-SNE) | ✅ | `results/metrics/`, `results/plots/`, `results/comparison/` |
| Retrieval examples | ✅ | `results/retrieval_examples/` |
| Final report PDF (6–10 pages) | ⏳ | `report/` — to be written |
| Demo notebook `notebooks/demo_context.ipynb` | ✅ | end-to-end single-track inference |
| Required project structure | ✅ | see README §4 |

> `data/raw/` is kept empty on purpose: the datasets already live in `Datasets/`
> and every path is resolved through `config.yaml`.

## Beyond the brief

* **Cross-dataset comparison** (`src/compare_datasets.py`) — each task run over
  every applicable dataset from Table 1 with identical hyper-parameters, reporting
  train / val / test metrics side by side plus generalisation-gap charts:

  | task | datasets compared |
  |---|---|
  | 1 | FMA-small, MagnaTagATune, GTZAN, MusicCaps |
  | 2 | GTZAN, FMA-medium |
  | 3 | FMA-medium, MagnaTagATune, DEAM |
  | 4 | DEAM, MusicCaps (with audio) |

* GTZAN made usable for Task 1 without label leakage, by generating its text from
  hand-crafted audio descriptors instead of the genre-bearing filename.
* GCN third convolution option, per-tag threshold tuning on validation,
  mixed-precision training, early stopping with a `min_epochs` floor that keeps
  the CNN baseline comparison fair, and an EDA notebook.

## Still outstanding

* **Final report PDF** (6–10 pages, NeurIPS/IEEE/ICML template) — `report/`.
* **Listening-test ratings** — the sheet is generated with one row per retrieval
  pair and five empty rating columns; it needs ≥ 5 human listeners to fill it in.

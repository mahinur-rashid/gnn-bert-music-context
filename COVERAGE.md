# Coverage check against `Instructions.pdf`

Every requirement in the brief, where it is implemented, and the result it
produced. **All four tasks are implemented and run; the report is written.** The
only outstanding item is the human listening-test ratings, which need people.

---

## Section 3 — Dataset requirements

| Requirement | Status | Where |
|---|---|---|
| ≥ 1 primary audio dataset | ✅ | FMA-small/medium, GTZAN, MagnaTagATune, DEAM, MusicCaps |
| ≥ 1 text/tag dataset | ✅ | MusicCaps captions, MTAT tags, FMA metadata, DEAM last.fm tags |
| Advanced pairing (FMA/MTAT × MusicCaps/DEAM) | ✅ | Task 3 on FMA-small + MTAT + DEAM; Task 4 on MusicCaps + DEAM |
| Resample to 22 050 Hz | ✅ | `audio_features.load_audio` |
| log-mel 128 bins / chroma 12 bins | ✅ | `audio_features.frame_features` |
| Normalise per track | ✅ | waveform peak-normalised per track; node features z-scored with **train-split** statistics (`graph_dataset.fit_feature_stats`) |
| Fixed-window segmentation | ✅ | `segment_bounds`, 3 s / 1.5 s hop (2 s / 1 s for the 10 s MusicCaps clips) |
| Beat-synchronous segmentation | ✅ | `beat_segment_bounds`, `--segmentation beat` |
| Chord-transition graph (nodes = chords, edges weighted by count) | ✅ | `graph_builder.build_edges` step 3 + `estimate_chords` (24 triad templates + N) |
| Segment graph (temporal adjacency + cosine sim > τ) | ✅ | `graph_builder.build_edges` steps 1–2 |
| BERT tokenizer, max length 128–256 | ✅ | 192, `config.yaml → text.max_length` |
| Official FMA / MTAT splits | ✅ | `prepare_splits.official_split` |
| DEAM standard train/val partition | ✅ | seeded 80/10/10 grouped by artist |
| No artist leakage | ✅ | `grouped_split` + `check_leakage` → `data/splits/leakage_report.json` |

**Built:** 61,750 graphs across six dataset variants (37,310 for the five datasets
used in the report; FMA-medium adds 24,440 and is optional).

> **Two stated caveats.** (1) The *standard* MagnaTagATune directory split
> (0–b / c / d–f) is not fully artist-disjoint — 57 of 229 artists straddle splits.
> It is used as-is because the brief asks for the official split;
> `prepare_splits --artist_safe` produces a strictly artist-grouped alternative.
> (2) GTZAN ships no metadata, so its Task 1 text is **generated** from
> hand-crafted audio descriptors rather than written by humans (using the filename
> would leak the label).

## Section 4.1 — Task 1 (Easy)

| Deliverable | Status | Evidence |
|---|---|---|
| BERT fine-tuning code (`bert-base-uncased` / `distilbert-base-uncased`) | ✅ | `bert_encoder.py`, `train_task1.py` (`--model_name`) |
| `t = BERT_CLS(X)`, `ŷ = σ(Wt+b)`, BCE per tag | ✅ | `BertTagClassifier`, `nn.BCEWithLogitsLoss` |
| Results on MagnaTagATune top-50 tags | ✅ | **0.304** macro-F1 / 0.392 micro-F1 / 0.287 AUC-PR |
| Results on MusicCaps caption → tag proxy | ✅ | **0.695** macro-F1 / 0.736 micro-F1 / 0.731 AUC-PR |
| Macro-F1 / Micro-F1 curves vs epochs | ✅ | `results/plots/task1_*_curves.png` |
| 5 example predictions + attention visualisation | ✅ | `results/examples/*.json`, `--attention_viz` |

## Section 4.2 — Task 2 (Medium)

| Deliverable | Status | Evidence |
|---|---|---|
| Graph construction scripts (chroma/MFCC segment graphs) | ✅ | `audio_features.py`, `graph_builder.py` |
| GraphSAGE / GAT encoder, PyTorch Geometric | ✅ | `gnn_model.GNNEncoder` (`sage` / `gat` / `gcn`) |
| Mean-pooling graph readout | ✅ | `global_mean_pool` ⊕ `global_max_pool` |
| Genre classification on GTZAN or FMA-small | ✅ | both: GTZAN GAT **0.790**, FMA-small SAGE 0.392 |
| Comparison vs CNN baseline on mel-spectrogram | ✅ | GTZAN CNN 0.597, FMA-small CNN **0.410** |

> **Stated bias:** the CNN baseline is undertrained at the 20-epoch budget. Under
> 60 epochs it reaches 0.801 on GTZAN (vs 0.597), narrowing the graph-vs-CNN margin
> to roughly GAT 0.838 vs CNN 0.801. Documented in `config.yaml`, the README and
> the report.

## Section 4.3 — Task 3 (Hard)

| Deliverable | Status | Evidence |
|---|---|---|
| End-to-end GNN–BERT fusion model | ✅ | `fusion_model.FusionModel` |
| Cross-attention fusion (Q=gW_Q, K=H_text W_K) | ✅ | `CrossAttentionFusion` |
| Multi-task loss `L_tags + α‖v−v̂‖² + β‖a−â‖²` | ✅ | `multitask_loss`; DEAM valence R² **0.521**, arousal 0.334 |
| Ablation: BERT-only / GNN-only / early concat / cross-attention | ✅ | all four, on all three datasets (`--ablation`) |
| Results on FMA-medium **or** MagnaTagATune (macro-F1, AUC-PR) | ✅ | MTAT **0.352** / 0.346 (FMA-small and DEAM reported alongside) |
| t-SNE of z coloured by genre **and** mood | ✅ | `results/plots/task3_deam_cross_attn_tsne_{genre,mood}.png` |
| 3 case studies: graph paths + caption/lyric alignment | ✅ | `results/case_studies/task3_deam_case_studies.json` |

**Ablation outcome (test macro-F1)** — the ranking is *not* stable across datasets,
which is the report's central finding:

| fusion | FMA-small | MagnaTagATune | DEAM |
|---|---|---|---|
| `gnn_only` | 0.183 | **0.390** | 0.181 |
| `bert_only` | 0.258 | 0.310 | 0.282 |
| `concat` | **0.318** | 0.349 | 0.299 |
| `cross_attn` | 0.205 | 0.352 | **0.429** |

## Section 4.4 — Task 4 (Advanced)

| Deliverable | Status | Evidence |
|---|---|---|
| Dual-encoder GNN–BERT with contrastive training | ✅ | `contrastive.DualEncoder`, `train_task4.train` |
| InfoNCE loss, cosine similarity, temperature τ | ✅ | `contrastive.info_nce`; embeddings L2-normalised so `g·t` is cosine |
| Caption → Audio R@1, R@5, R@10 | ✅ | MusicCaps 0.066 / **0.193** / 0.300 (random 0.002 / 0.010 / 0.020) |
| Audio → Caption R@K | ✅ | MusicCaps 0.049 / 0.187 / 0.298, median rank 25 of 513 |
| Retrieval evaluation table on the MusicCaps test split | ✅ | `results/metrics/task4_musiccaps_contrastive.json` |
| 10 qualitative examples (query caption → top-3 clips) | ✅ | `results/retrieval_examples/*_examples.json` |
| Zero-shot tag prediction vs the Task 3 supervised model | ✅ | DEAM zero-shot **0.129** vs supervised **0.429** (Δ = −0.300) |
| Human evaluation: ≥ 5 listeners rate match on [1,5] | 🟡 sheet generated | `results/retrieval_examples/*_listening_test.csv` — **needs listeners** |

> Two deliberate extensions of the literal spec: the InfoNCE loss is **symmetric**
> (the brief writes one direction, but it also asks for R@K in both), and τ is
> learned CLIP-style by default — `--fixed_temperature` reproduces the literal
> formula, and the choice is recorded in each results file.

## Section 6 — Evaluation metrics

| Metric | Status | Where |
|---|---|---|
| Per-tag precision / recall / F1 | ✅ | `evaluate.multilabel_metrics` |
| Macro-F1, Micro-F1 | ✅ | same |
| AUC-PR (mean average precision over tags) | ✅ | same |
| MAE and R² for valence/arousal | ✅ | `evaluate.regression_metrics` |
| Graph coherence score `S_graph` | ⚠️ implemented, **uninformative** | see below |

> `S_graph` returns 0.996–1.000 on every dataset and fusion mode, so it
> discriminates nothing as specified. The likely cause is that node embeddings are
> read after BatchNorm + ReLU, which makes activations non-negative and inflates
> pairwise cosine similarity; a useful version would need a higher τ,
> pre-activation features, or a rewired null graph. Reported as a negative result
> rather than presented as evidence.

## Section 7 — Algorithms

| Algorithm | Status | Where |
|---|---|---|
| 1 — BERT multi-label tag classifier | ✅ | `train_task1.train_one` |
| 2 — GNN encoder on music segment graph | ✅ | `graph_builder` + `train_task2.train_model` |
| 3 — GNN–BERT fusion for context understanding | ✅ | `train_task3.train_fusion` |
| 4 — contrastive GNN–BERT | ✅ | `contrastive.py` + `train_task4.py` |

## Section 8 — Baselines (≥ 2 required, all 4 implemented)

| Baseline | Status | Example result |
|---|---|---|
| B1 majority-class / random tag predictor | ✅ | GTZAN random 0.112, majority 0.018 macro-F1 |
| B2 CNN on mel-spectrogram | ✅ | GTZAN 0.597, FMA-small 0.410 macro-F1 |
| B3 BERT-only | ✅ | Task 1, and the `bert_only` ablation of Task 3 |
| B4 PCA + MLP on hand-crafted features | ✅ | GTZAN 0.755, FMA-small 0.375 macro-F1 |

## Section 10 — Submission requirements

| Item | Status | Where |
|---|---|---|
| Full source code | ✅ | `src/` (21 modules) |
| ≥ 20 preprocessed graph samples (`.pt` / `.json`) | ✅ | `data/processed/graph_samples/<dataset>/` — 20 `.pt` + 20 `.json` for each of 6 datasets |
| Evaluation tables + plots (F1, AUC-PR, t-SNE) | ✅ | `results/metrics/`, `results/plots/` (55 figures), `results/comparison/` |
| Retrieval examples | ✅ | `results/retrieval_examples/` |
| Final report PDF (6–10 pages, NeurIPS template) | ✅ | `report/CSE715_Final_Report.pdf` — 10 pages |
| Demo notebook `notebooks/demo_context.ipynb` | ✅ | end-to-end single-track inference |
| Required project structure | ✅ | see README §5 |

> `data/raw/` is kept empty on purpose: the datasets live in `Datasets/` and every
> path is resolved through `config.yaml`.

## Beyond the brief

* **Cross-dataset comparison** (`src/compare_datasets.py`) — each task run over
  every applicable dataset with identical hyper-parameters, reporting train / val /
  test metrics side by side plus generalisation-gap charts:

  | task | datasets compared |
  |---|---|
  | 1 | FMA-small, MagnaTagATune, GTZAN, MusicCaps |
  | 2 | GTZAN, FMA-small |
  | 3 | FMA-small, MagnaTagATune, DEAM |
  | 4 | DEAM, MusicCaps |

* **GTZAN made usable for Task 1 without label leakage**, by generating its text
  from hand-crafted audio descriptors instead of the genre-bearing filename.
* GCN as a third convolution option; per-tag threshold tuning on validation;
  mixed-precision training; early stopping with a `min_epochs` floor that keeps the
  CNN baseline comparison fair; `scripts/run_all_tasks.ps1` for an unattended
  end-to-end run; and an EDA notebook.

## Known limitations (all stated in the report)

* **Single seed.** DEAM trained twice under identical settings gave 0.414 and
  0.429 test macro-F1. With 177 DEAM / 100 GTZAN test clips, differences below
  ~0.02 are not interpretable.
* **CNN baseline undertrained** at 20 epochs (see Section 4.2 above).
* **`S_graph` saturates** and carries no information as specified.
* **MusicCaps Task 1 is a proxy** by construction; **GTZAN Task 1 text is
  generated**.
* Only the first 30 s of each track is used, capped at 24 graph nodes, so
  long-range musical form is invisible.
* Chord estimation is chroma template matching, well below a trained chord
  recogniser.

## Still outstanding

* **Listening-test ratings** — the sheet is generated with one row per retrieval
  pair and five empty rating columns; it needs ≥ 5 human listeners. The report
  states explicitly that no human scores are reported yet.

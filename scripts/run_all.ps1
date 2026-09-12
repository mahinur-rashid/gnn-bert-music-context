# Full reproduction pipeline (PowerShell).
#   .\scripts\run_all.ps1                # preprocessing + all four tasks + comparisons
#   .\scripts\run_all.ps1 -SkipPrep      # tasks only (artefacts already built)
#   .\scripts\run_all.ps1 -PrepOnly      # build data artefacts and stop
param(
    [int]$Jobs = 10,
    [switch]$SkipPrep,
    [switch]$PrepOnly
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $SkipPrep) {
    Write-Host "`n=== 1/3  text tables, splits and graphs ===" -ForegroundColor Cyan
    python -m src.text_data --dataset musiccaps
    python -m src.text_data --dataset mtat
    python -m src.text_data --dataset fma --fma_subset small
    python -m src.text_data --dataset fma --fma_subset medium
    python -m src.text_data --dataset deam
    python -m src.text_data --dataset gtzan

    python -m src.prepare_splits --dataset all

    # ~5 min          ~2 min            ~4 min
    python -m src.graph_builder --dataset gtzan      --jobs $Jobs
    python -m src.graph_builder --dataset deam       --jobs $Jobs
    python -m src.graph_builder --dataset musiccaps  --jobs $Jobs
    # ~10 min                    ~20 min                    ~35 min (keeps mel for the CNN baseline)
    python -m src.graph_builder --dataset fma_small  --jobs $Jobs
    python -m src.graph_builder --dataset mtat       --jobs $Jobs --no_mel
    python -m src.graph_builder --dataset fma_medium --jobs $Jobs
}

if ($PrepOnly) { Write-Host "`nPreprocessing done." -ForegroundColor Green; exit 0 }

Write-Host "`n=== 2/3  single-dataset reference runs ===" -ForegroundColor Cyan
python -m src.train_task1 --dataset musiccaps --attention_viz
python -m src.train_task2 --dataset gtzan --compare --pca_mlp
python -m src.train_task3 --dataset deam --ablation --save_model
python -m src.train_task4 --dataset musiccaps --save_model

Write-Host "`n=== 3/3  cross-dataset comparisons ===" -ForegroundColor Cyan
# Task 1: fma_small, magnatagatune, gtzan, musiccaps (text only)
# Task 2: gtzan, fma_medium
# Task 3: fma_medium, magnatagatune, deam
# Task 4: deam, musiccaps (with audio)
python -m src.compare_datasets --task 1
python -m src.compare_datasets --task 2 --pca_mlp
python -m src.compare_datasets --task 3
python -m src.compare_datasets --task 4

python -m src.evaluate --summarize
Write-Host "`nDone. See results\summary.md and results\comparison\README.md" -ForegroundColor Green

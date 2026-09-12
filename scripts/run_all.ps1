# Full reproduction pipeline (PowerShell).
#   .\scripts\run_all.ps1                 # everything except FMA-medium graphs
#   .\scripts\run_all.ps1 -WithFmaMedium  # also build the 25k FMA-medium graphs
param(
    [int]$Jobs = 10,
    [switch]$WithFmaMedium
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "`n=== 1/5  text tables + label vocabularies ===" -ForegroundColor Cyan
python -m src.text_data --dataset musiccaps
python -m src.text_data --dataset mtat
python -m src.text_data --dataset fma --fma_subset small
python -m src.text_data --dataset fma --fma_subset medium
python -m src.text_data --dataset deam

Write-Host "`n=== 2/5  train / val / test splits ===" -ForegroundColor Cyan
python -m src.prepare_splits --dataset all

Write-Host "`n=== 3/5  music-structure graphs ===" -ForegroundColor Cyan
python -m src.graph_builder --dataset gtzan     --jobs $Jobs
python -m src.graph_builder --dataset deam      --jobs $Jobs
python -m src.graph_builder --dataset fma_small --jobs $Jobs
python -m src.graph_builder --dataset mtat      --jobs $Jobs --no_mel
if ($WithFmaMedium) {
    python -m src.graph_builder --dataset fma_medium --jobs $Jobs --no_mel
}

Write-Host "`n=== 4/5  tasks 1-3 ===" -ForegroundColor Cyan
python -m src.train_task1 --dataset musiccaps --epochs 5 --attention_viz
python -m src.train_task2 --dataset gtzan --compare --pca_mlp
python -m src.train_task3 --dataset deam --ablation --save_model

Write-Host "`n=== 5/5  cross-dataset comparison ===" -ForegroundColor Cyan
python -m src.compare_datasets --task 1
python -m src.compare_datasets --task 2
python -m src.compare_datasets --task 3

python -m src.evaluate --summarize
Write-Host "`nDone. See results\summary.md and results\comparison\README.md" -ForegroundColor Green

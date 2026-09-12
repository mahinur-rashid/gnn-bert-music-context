# ---------------------------------------------------------------------------
# Run Tasks 1 -> 2 -> 3 -> 4 end to end, plus every cross-dataset comparison.
#
#   .\scripts\run_all_tasks.ps1                  # everything (~2 h)
#   .\scripts\run_all_tasks.ps1 -Quick           # 5-min smoke run of every step
#   .\scripts\run_all_tasks.ps1 -Task2Epochs 60  # fairer CNN baseline (see README)
#
# Assumes preprocessing is done. If not, run first:
#   python -m src.prepare_data --stage all --jobs 10
#
# A failing step does NOT abort the rest -- the status of every step is
# summarised at the end and the full log is written to results\run_all.log.
# ---------------------------------------------------------------------------
param(
    [switch]$Quick,
    [int]$Task2Epochs = 0,     # 0 = use config.yaml
    [switch]$SkipComparisons
)

Set-Location (Split-Path $PSScriptRoot -Parent)
$ErrorActionPreference = "Continue"

$log = "results\run_all.log"
New-Item -ItemType Directory -Force -Path "results" | Out-Null
"=== run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $log -Encoding utf8

$quickArgs = @()
if ($Quick) { $quickArgs = @("--limit", "200", "--epochs", "2", "--min_epochs", "1") }

$steps = New-Object System.Collections.ArrayList

function Add-Step($name, $argList) {
    [void]$steps.Add([pscustomobject]@{ Name = $name; Args = $argList })
}

# --- Task 1 -------------------------------------------------------------- #
Add-Step "Task 1  comparison (fma_small, mtat, gtzan, musiccaps)" `
    (@("-m", "src.compare_datasets", "--task", "1") + $quickArgs)
Add-Step "Task 1  attention visualisation (musiccaps)" `
    (@("-m", "src.train_task1", "--dataset", "musiccaps", "--attention_viz") + $quickArgs)

# --- Task 2 -------------------------------------------------------------- #
$t2 = @("-m", "src.compare_datasets", "--task", "2", "--pca_mlp")
if ($Task2Epochs -gt 0) { $t2 += @("--epochs", "$Task2Epochs") }
Add-Step "Task 2  comparison (gtzan, fma_small) + CNN/PCA-MLP baselines" ($t2 + $quickArgs)

# --- Task 3 -------------------------------------------------------------- #
Add-Step "Task 3  comparison (fma_small, mtat, deam), 4-way ablation" `
    (@("-m", "src.compare_datasets", "--task", "3") + $quickArgs)
Add-Step "Task 3  checkpoint for the demo notebook (deam)" `
    (@("-m", "src.train_task3", "--dataset", "deam", "--ablation", "--save_model") + $quickArgs)

# --- Task 4 -------------------------------------------------------------- #
# runs AFTER Task 3 so zero-shot tagging can cite the supervised DEAM result
Add-Step "Task 4  comparison (deam, musiccaps) + zero-shot vs Task 3" `
    (@("-m", "src.compare_datasets", "--task", "4") + $quickArgs)
Add-Step "Task 4  checkpoint + retrieval examples (musiccaps)" `
    (@("-m", "src.train_task4", "--dataset", "musiccaps", "--save_model") + $quickArgs)

if ($SkipComparisons) {
    $steps = $steps | Where-Object { $_.Name -notlike "*comparison*" }
}

# --- execute ------------------------------------------------------------- #
$results = @()
$n = 0
$total = $steps.Count
$runStart = Get-Date

foreach ($step in $steps) {
    $n++
    $head = "[$n/$total] $($step.Name)"
    Write-Host "`n$('=' * 78)" -ForegroundColor DarkGray
    Write-Host $head -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    "`n$('=' * 78)`n$head`n$('=' * 78)" | Out-File $log -Append -Encoding utf8

    $t0 = Get-Date
    & python @($step.Args) 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    if ($code -eq 0) {
        Write-Host "  OK  ($mins min)" -ForegroundColor Green
    } else {
        Write-Host "  FAILED (exit $code, $mins min) - continuing" -ForegroundColor Red
    }
    $results += [pscustomobject]@{ Step = $step.Name; Status = $(if ($code -eq 0) { "OK" } else { "FAILED($code)" }); Minutes = $mins }
}

# --- aggregate ----------------------------------------------------------- #
Write-Host "`n$('=' * 78)" -ForegroundColor DarkGray
Write-Host "Aggregating results" -ForegroundColor Cyan
Write-Host ("=" * 78) -ForegroundColor DarkGray
& python -m src.evaluate --summarize 2>&1 | Tee-Object -FilePath $log -Append

$elapsed = [math]::Round(((Get-Date) - $runStart).TotalMinutes, 1)
Write-Host "`n$('=' * 78)" -ForegroundColor DarkGray
Write-Host "SUMMARY  (total $elapsed min)" -ForegroundColor Cyan
Write-Host ("=" * 78) -ForegroundColor DarkGray
$results | Format-Table -AutoSize
$results | Format-Table -AutoSize | Out-String | Out-File $log -Append -Encoding utf8

$failed = @($results | Where-Object { $_.Status -ne "OK" })
if ($failed.Count -gt 0) {
    Write-Host "$($failed.Count) step(s) failed - see $log" -ForegroundColor Red
} else {
    Write-Host "All steps completed." -ForegroundColor Green
}
Write-Host @"

Results:
  results\summary.md                                overall table
  results\comparison\task<N>_dataset_comparison.md  per-task dataset comparison
  results\plots\                                    curves, confusion, t-SNE
  results\case_studies\                             Task 3 chord paths + alignment
  results\retrieval_examples\                       Task 4 examples + listening sheet
  results\zero_shot\                                Task 4 zero-shot vs Task 3
  $log                                              full console log
"@ -ForegroundColor Gray

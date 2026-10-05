# sharon/run_benchmark.ps1
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Backbones = @("densenet121", "resnet50", "efficientnet_b0", "swin_unetr")
$Epochs = 150

Write-Host "=== Starting 3D Organ Backbone Benchmarking Suite in sharon/ ==="

foreach ($BB in $Backbones) {
    Write-Host "=================================================="
    Write-Host " Training Backbone: $BB"
    Write-Host "=================================================="

    if ($BB -eq "swin_unetr") {
        $BatchSize = 2
        $GradAccum = 8
    } else {
        $BatchSize = 4
        $GradAccum = 4
    }

    python train.py `
        --data_root dataset `
        --backbone $BB `
        --epochs $Epochs `
        --batch_size $BatchSize `
        --grad_accum_steps $GradAccum
}

Write-Host "=================================================="
Write-Host " All Backbones Trained. Evaluating Real Metrics..."
Write-Host "=================================================="

python evaluate_benchmark.py

Write-Host "Benchmarking complete. Real evaluation report generated at sharon/outputs/BENCHMARK_REPORT.md."

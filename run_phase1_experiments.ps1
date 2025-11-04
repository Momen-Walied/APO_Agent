# Run Phase 1 Experiments: Baseline vs Optimized vs Tiered
# Creates reports/ directory with JSON outputs

$ErrorActionPreference = "Stop"

Write-Host "=== Phase 1 Experiments: ACE Efficiency Optimizations ===" -ForegroundColor Cyan
Write-Host ""

# Create reports directory
if (-not (Test-Path "reports")) {
    New-Item -ItemType Directory -Path "reports" | Out-Null
    Write-Host "Created reports/ directory" -ForegroundColor Green
}

# Experiment 1: Baseline (no optimizations)
Write-Host "[1/3] Running BASELINE (no optimizations)..." -ForegroundColor Yellow
python app.py --config configs/experiments/baseline.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Baseline run failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Baseline complete" -ForegroundColor Green
Write-Host ""

# Experiment 2: Phase 1 Optimized (early stopping + usefulness weighting)
Write-Host "[2/3] Running PHASE1 OPTIMIZED (early stopping + usefulness)..." -ForegroundColor Yellow
python app.py --config configs/experiments/phase1_optimized.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 1 optimized run failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Phase 1 optimized complete" -ForegroundColor Green
Write-Host ""

# Experiment 3: Tiered Models (14B Gen + 14B Ref + 7B Cur)
Write-Host "[3/3] Running TIERED MODELS (14B/14B/7B)..." -ForegroundColor Yellow
python app.py --config configs/experiments/tiered_models.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tiered models run failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Tiered models complete" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "=== All experiments completed ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor White
Write-Host "  - reports/baseline_finer_ord.json" -ForegroundColor Gray
Write-Host "  - reports/phase1_optimized_finer_ord.json" -ForegroundColor Gray
Write-Host "  - reports/tiered_models_finer_ord.json" -ForegroundColor Gray
Write-Host ""
Write-Host "Next: Compare span F1, cost, and latency metrics" -ForegroundColor Cyan

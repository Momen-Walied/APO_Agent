# Runs a small experiment matrix and writes reports to ./reports
param(
  [string]$Provider = "huggingface",
  [string]$Dataset = "gtfintechlab/finer-ord",
  [int]$Limit = 50
)

$ErrorActionPreference = "Stop"

$newline = [Environment]::NewLine

$cases = @(
  @{ name="tiered-14b-7b"; args="--provider $Provider --dataset $Dataset --limit $Limit --model Qwen/Qwen2.5-14B-Instruct --reflector_model Qwen/Qwen2.5-14B-Instruct --curator_model Qwen/Qwen2.5-7B-Instruct" },
  @{ name="single-7b";    args="--provider $Provider --dataset $Dataset --limit $Limit --model Qwen/Qwen2.5-7B-Instruct" }
)

if (-not (Test-Path reports)) { New-Item -ItemType Directory reports | Out-Null }

foreach ($c in $cases) {
  Write-Host "=== Running $($c.name) ==="
  $outfile = "reports/report_$($c.name).json"
  $cmd = "python app.py $($c.args) --output $outfile"
  Write-Host $cmd
  cmd /c $cmd
}

Write-Host "All runs completed. See ./reports"

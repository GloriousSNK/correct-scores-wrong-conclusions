$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $repo "results\overnight\cpu-continuation-$stamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Experiment {
    param([string]$Name, [string[]]$Arguments)
    $log = Join-Path $logDir "$Name.log"
    "[$(Get-Date -Format o)] $Name" | Tee-Object -FilePath $log
    & $python @Arguments *>&1 | Tee-Object -FilePath $log -Append
    "[$(Get-Date -Format o)] exit=$LASTEXITCODE" | Tee-Object -FilePath $log -Append
}

Set-Location $repo
$dataset = "results/seed_sweep_two_seed_focused/datasets/seed_20260621"
$checkpoints = "results/seed_sweep_two_seed_focused/checkpoints/seed_20260621"
$seedRoot = "results/seed_sweep_two_seed_focused"

Invoke-Experiment "lnn_cpu_pool" @(
    "scripts/run_lnn_cpu_pool.py",
    "--dataset-dir", $dataset,
    "--checkpoint-dir", $checkpoints,
    "--horizons", "1", "10",
    "--workers", "10")
Invoke-Experiment "pendulum_chronos" @(
    "scripts/run_eval.py",
    "--dataset-dir", $dataset,
    "--checkpoint-dir", $checkpoints,
    "--horizons", "1", "10",
    "--models", "chronos2-uni", "chronos2-multi")
Invoke-Experiment "pendulum_summary" @(
    "scripts/aggregate_seed_sweep.py", "--root", $seedRoot)
Invoke-Experiment "lorenz_chronos" @(
    "scripts/run_lorenz_seed_sweep.py",
    "--held-seeds", "20260620", "20260621",
    "--train-trajectories", "20", "--held-trajectories", "5", "--epochs", "300")

@{
    started_at = $stamp
    completed_at = Get-Date -Format o
    lnn_cpu_workers = 10
    pendulum_seed = 20260621
    pendulum_horizons = @(1, 10)
    lorenz_held_seeds = @(20260620, 20260621)
} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $logDir "run_manifest.json")

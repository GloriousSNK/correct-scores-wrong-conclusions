$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $repo "results\overnight\$stamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Experiment {
    param([string]$Name, [string[]]$Arguments)
    $log = Join-Path $logDir "$Name.log"
    "[$(Get-Date -Format o)] $Name" | Tee-Object -FilePath $log
    & $python @Arguments *>&1 | Tee-Object -FilePath $log -Append
    "[$(Get-Date -Format o)] exit=$LASTEXITCODE" | Tee-Object -FilePath $log -Append
}

Set-Location $repo
$seeds = @("20260620", "20260621", "20260622", "20260623", "20260624")
$pendulumModels = @(
    "euler", "rk4", "symplectic",
    "neural-ode", "neural-ode-rollout", "neural-ode-rollout-mixed",
    "hnn", "hnn-rollout", "hnn-rollout-mixed", "lnn",
    "chronos-local", "chronos2-uni", "chronos2-multi"
)

Invoke-Experiment "pendulum_seed_sweep" (@(
    "scripts/run_seed_sweep.py", "--seeds") + $seeds + @(
    "--trajectories-per-cell", "5", "--models") + $pendulumModels)
Invoke-Experiment "pendulum_seed_summary" @("scripts/aggregate_seed_sweep.py")
Invoke-Experiment "lorenz_seed_sweep" (@(
    "scripts/run_lorenz_seed_sweep.py", "--held-seeds") + $seeds + @(
    "--train-trajectories", "20", "--held-trajectories", "5", "--epochs", "300"))

@{
    started_at = $stamp
    completed_at = Get-Date -Format o
    pendulum_seeds = $seeds
    pendulum_trajectories_per_cell = 5
    pendulum_models = $pendulumModels
    lorenz_train_seed = 42
    lorenz_held_seeds = $seeds
    lorenz_train_trajectories = 20
    lorenz_held_trajectories = 5
} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $logDir "run_manifest.json")

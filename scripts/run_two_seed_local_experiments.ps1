$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $repo "results\overnight\two-seed-$stamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Experiment {
    param([string]$Name, [string[]]$Arguments)
    $log = Join-Path $logDir "$Name.log"
    "[$(Get-Date -Format o)] $Name" | Tee-Object -FilePath $log
    & $python @Arguments *>&1 | Tee-Object -FilePath $log -Append
    "[$(Get-Date -Format o)] exit=$LASTEXITCODE" | Tee-Object -FilePath $log -Append
}

Set-Location $repo
$newPendulumSeed = "20260621"
$lorenzSeeds = @("20260620", "20260621")
$pendulumModels = @(
    "euler", "rk4", "symplectic",
    "neural-ode-rollout-mixed",
    "hnn-rollout", "hnn-rollout-mixed", "lnn",
    "chronos2-uni", "chronos2-multi"
)
$pendulumRoot = "results/seed_sweep_two_seed_focused"

Invoke-Experiment "pendulum_new_seed" (@(
    "scripts/run_seed_sweep.py", "--root", $pendulumRoot,
    "--seeds", $newPendulumSeed, "--trajectories-per-cell", "5",
    "--horizons", "1", "10", "--models") + $pendulumModels)
Invoke-Experiment "pendulum_new_seed_summary" @(
    "scripts/aggregate_seed_sweep.py", "--root", $pendulumRoot)
Invoke-Experiment "lorenz_two_seed_sweep" (@(
    "scripts/run_lorenz_seed_sweep.py", "--held-seeds") + $lorenzSeeds + @(
    "--train-trajectories", "20", "--held-trajectories", "5", "--epochs", "300"))

@{
    started_at = $stamp
    completed_at = Get-Date -Format o
    existing_pendulum_seed = 20260620
    new_pendulum_seed = 20260621
    new_pendulum_trajectories_per_cell = 5
    pendulum_horizons = @(1, 10)
    pendulum_models = $pendulumModels
    lorenz_train_seed = 42
    lorenz_held_seeds = $lorenzSeeds
} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $logDir "run_manifest.json")

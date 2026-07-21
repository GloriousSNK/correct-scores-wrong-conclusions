$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$log = Join-Path $repo "results\overnight\strengthening_cpu_pipeline.log"
Set-Location $repo

function Run-Step([string[]]$Arguments) {
    ("+ " + $python + " " + ($Arguments -join " ")) | Tee-Object -FilePath $log -Append
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments *>> $log
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($exitCode -ne 0) { throw "Step failed with exit code $exitCode" }
}

while (-not (Test-Path "results\balanced_seed_study\dataset_inventory.json")) {
    Start-Sleep -Seconds 30
}

while ($true) {
    if (Test-Path "results\overnight\repair_original_lnn.complete") {
        break
    }
    Start-Sleep -Seconds 60
}

Run-Step @("scripts\materialize_balanced_checkpoints.py")

Run-Step @(
    "scripts\eval_scientific_baselines.py",
    "--output-root", "results\balanced_seed_study\scientific_baselines_final"
)
Run-Step @("scripts\run_lorenz_seed_sweep.py", "--models", "sindy", "nvar")
Run-Step @("scripts\aggregate_balanced_seed_study.py")
"CPU pipeline complete" | Tee-Object -FilePath $log -Append

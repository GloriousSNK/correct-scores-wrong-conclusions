$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$log = Join-Path $repo "results\overnight\strengthening_gpu_pipeline.log"
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
while ((Get-ChildItem "results\training_budget_study\models" -Recurse -Filter training_metadata.json -ErrorAction SilentlyContinue).Count -lt 9) {
    Start-Sleep -Seconds 60
}

Run-Step @("scripts\run_balanced_local_models.py")
Run-Step @("scripts\run_lorenz_seed_sweep.py")
Run-Step @("scripts\eval_training_budget_study.py")
Run-Step @("scripts\aggregate_balanced_seed_study.py")
"GPU pipeline complete" | Tee-Object -FilePath $log -Append

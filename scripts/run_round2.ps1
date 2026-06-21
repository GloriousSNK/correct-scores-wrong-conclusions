# =====================================================================
# Round 2 overnight orchestration (CUDA / dp-work).
# Self-contained: runs the entire remaining pipeline in sequence with no
# further input. Preserves the canonical Mac checkpoints in
# results/checkpoints (untouched); all new results go to *_round2.
#
# Steps: LNN full retrain -> mixed-k NODE -> mixed-k HNN ->
#        build round2 checkpoint set -> eval new/changed models ->
#        aggregate -> refresh Tier 3 analyses.
#
# Launch:  powershell -ExecutionPolicy Bypass -File scripts\run_round2.ps1
# =====================================================================

$ErrorActionPreference = "Continue"   # log failures, never abort the whole run
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"   # reduce VRAM fragmentation on the 8GB GPU
Set-Location "C:\Users\srima\dp-work\Double-Pendulum copy"

$py  = ".\.venv-win\Scripts\python.exe"
$LM  = "results\learned_models"
$R2  = "results\checkpoints_round2"
$S2  = "results\summary_round2"
$log = "results\round2_run.log"

function Log($m) {
    $line = "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))  $m"
    Write-Output $line
    Add-Content -Path $log -Value $line
}

New-Item -ItemType Directory -Force "results" | Out-Null
Remove-Item "results\ROUND2_DONE.txt","results\ROUND2_FAILED.txt" -ErrorAction SilentlyContinue
Log "=================== ROUND 2 START ==================="
& $py -c "import torch; print('device check: cuda=%s %s' % (torch.cuda.is_available(), torch.__version__))" 2>&1 | ForEach-Object { Log $_ }

# 1. LNN full-scale retrain (the marquee fix: CUDA enables the second-order
#    autograd that MPS could not run, so this trains at full scale, hidden=256).
Log "STEP 1/6: LNN full-scale retrain (500 epochs)"
& $py scripts\train_learned.py --models lnn --epochs 500 --hidden 256 --layers 3 --output-dir $LM 2>&1 | ForEach-Object { Log $_ }
Log "STEP 1 done (exit=$LASTEXITCODE)"

# 2a. Mixed-k Neural ODE (short+long windows optimised together).
Log "STEP 2/6a: NODE mixed-k rollout (ks=10,50; 500 epochs)"
& $py scripts\train_learned.py --models neural_ode --rollout-ks 10 50 --epochs 500 --batch-size 512 --output-dir $LM 2>&1 | ForEach-Object { Log $_ }
Log "STEP 2a done (exit=$LASTEXITCODE)"

# 2b. Mixed-k HNN (deeper graph; smaller batch).
Log "STEP 2/6b: HNN mixed-k rollout (ks=5,20; 500 epochs)"
& $py scripts\train_learned.py --models hnn --rollout-ks 5 20 --epochs 500 --batch-size 256 --output-dir $LM 2>&1 | ForEach-Object { Log $_ }
Log "STEP 2b done (exit=$LASTEXITCODE)"

# 3. Build the round2 checkpoint set: copy the canonical Mac checkpoints, then
#    drop the LNN cells so they are recomputed with the new full-scale weights.
Log "STEP 3/6: prepare round2 checkpoint set (preserving canonical results/checkpoints)"
New-Item -ItemType Directory -Force $R2 | Out-Null
Copy-Item "results\checkpoints\*.json" $R2 -Force
Remove-Item "$R2\lnn__*.json" -Force -ErrorAction SilentlyContinue
$n = (Get-ChildItem $R2 -Filter *.json -ErrorAction SilentlyContinue).Count
Log "STEP 3 done ($n cells copied, lnn cells cleared for recompute)"

# 4. Evaluate the new/changed models into round2 (learned + chronos; no API calls).
Log "STEP 4/6: eval lnn + mixed-k + chronos-local"
& $py scripts\run_eval.py --models lnn neural-ode-rollout-mixed hnn-rollout-mixed chronos-local --checkpoint-dir $R2 2>&1 | ForEach-Object { Log $_ }
Log "STEP 4 done (exit=$LASTEXITCODE)"

# 5. Aggregate the combined leaderboard.
Log "STEP 5/6: aggregate"
& $py scripts\aggregate.py --checkpoint-dir $R2 --summary-dir $S2 2>&1 | ForEach-Object { Log $_ }
Log "STEP 5 done (exit=$LASTEXITCODE)"

# 6. Refresh the chaos-appropriate analyses on the updated data.
Log "STEP 6/6: Tier 3 analyses (predictability horizon + system-ID)"
& $py scripts\analyze_divergence.py --checkpoint-dir $R2 --summary-dir $S2 2>&1 | ForEach-Object { Log $_ }
& $py scripts\analyze_system_id.py --checkpoint-dir $R2 --summary-dir $S2 2>&1 | ForEach-Object { Log $_ }
Log "STEP 6 done"

Log "=================== ROUND 2 COMPLETE ==================="
"COMPLETED $([DateTime]::Now)" | Out-File -FilePath "results\ROUND2_DONE.txt" -Encoding utf8

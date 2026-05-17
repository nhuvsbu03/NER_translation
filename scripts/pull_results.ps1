# Pull inference results and training log from vast.ai back to local machine.
# Usage: .\scripts\pull_results.ps1 [-Remote vastai] [-Pair en-ru]
param(
    [string]$Remote = "vastai",
    [string]$Pair   = "en-ru"
)

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LocalResults = Join-Path $ProjectRoot "SeqDiffuSeq\results\$Pair"

$RemoteCkptDir = "/root/NER_translation/SeqDiffuSeq/ckpts/$Pair"

New-Item -ItemType Directory -Path $LocalResults -Force | Out-Null

Write-Host "==> Pulling inference outputs ($Pair)..."
scp -r "${Remote}:${RemoteCkptDir}/inference_out" "$LocalResults"

Write-Host "==> Pulling training log..."
$LogDir = Join-Path $LocalResults "log"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
scp "${Remote}:${RemoteCkptDir}/log/train.log" "$LogDir\train.log" 2>$null
if (-not $?) { Write-Host "    (no train.log found — skipping)" }

Write-Host ""
Write-Host "==> Done. Results saved to: $LocalResults"
Write-Host "    Run analysis: python analysis\eval_bleu.py --pair $Pair"

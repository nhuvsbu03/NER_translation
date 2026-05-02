# Push code/data/tokenizer/checkpoint to vast.ai instance.
# Usage: .\scripts\push_vastai.ps1 [ssh-alias]
# Requires: vastai CLI, OpenSSH (built into Windows 10+)

param(
    [string]$Remote = "vastai",
    [switch]$Pull
)

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────
# Common Google Drive paths on Windows — uncomment the one that matches yours:
# $GdriveRoot = "G:\My Drive\BUyen_Qnhu++\src\SeqDiffuSeq"
# $GdriveRoot = "$env:USERPROFILE\Google Drive\My Drive\BUyen_Qnhu++\src\SeqDiffuSeq"
$GdriveRoot    = "G:\My Drive\BUyen_Qnhu++\src\SeqDiffuSeq"
$GdriveTokDir  = "$GdriveRoot\data\en-zh"
$GdriveCkptDir = "$GdriveRoot\ckpts\en-zh"
# ─────────────────────────────────────────────────────────────────────────

function Remote-Mkdir($path) {
    ssh $Remote "mkdir -p $path"
}

function Scp-File($src, $dst) {
    scp -q $src "${Remote}:${dst}"
}

function Scp-Dir($src, $dst) {
    scp -q -r $src "${Remote}:${dst}"
}

# ── Code ──────────────────────────────────────────────────────────────────
Write-Host "==> Pushing SeqDiffuSeq/ → $Remote:/root/SeqDiffuSeq/"
Remote-Mkdir "/root/SeqDiffuSeq"

# Copy src/, scripts, top-level python files (skip ckpts/, data/, __pycache__)
$seqDir = Join-Path $ProjectRoot "SeqDiffuSeq"
foreach ($item in Get-ChildItem $seqDir) {
    if ($item.Name -in @("ckpts", "data", "out", "__pycache__", ".git")) { continue }
    if ($item.Extension -in @(".pt", ".npy")) { continue }
    if ($item.PSIsContainer) {
        Scp-Dir $item.FullName "/root/SeqDiffuSeq/$($item.Name)"
    } else {
        Scp-File $item.FullName "/root/SeqDiffuSeq/$($item.Name)"
    }
}

# ── Dataset ───────────────────────────────────────────────────────────────
Write-Host "==> Pushing train_dataset/ → $Remote:/root/train_dataset/"
Remote-Mkdir "/root/train_dataset"
Scp-Dir (Join-Path $ProjectRoot "train_dataset\*") "/root/train_dataset/"

# ── Notebook ──────────────────────────────────────────────────────────────
Write-Host "==> Pushing notebook..."
Scp-File (Join-Path $ProjectRoot "vastai_training.ipynb") "/root/vastai_training.ipynb"

# ── Tokenizer ─────────────────────────────────────────────────────────────
Write-Host "==> Pushing tokenizer..."
Remote-Mkdir "/root/SeqDiffuSeq/data/en-zh"

foreach ($fname in @("vocab.json", "merges.txt")) {
    $localPath = Join-Path $ProjectRoot "SeqDiffuSeq\data\en-zh\$fname"
    $drivePath = Join-Path $GdriveTokDir $fname

    if (Test-Path $localPath) {
        Write-Host "    $fname (local)"
        Scp-File $localPath "/root/SeqDiffuSeq/data/en-zh/$fname"
    } elseif (Test-Path $drivePath) {
        Write-Host "    $fname (Google Drive)"
        Scp-File $drivePath "/root/SeqDiffuSeq/data/en-zh/$fname"
    } else {
        Write-Host "    WARN: $fname not found — skipping"
    }
}

# ── Checkpoint ────────────────────────────────────────────────────────────
Write-Host "==> Pushing latest checkpoint → $Remote:/root/SeqDiffuSeq/ckpts/en-zh/"
Remote-Mkdir "/root/SeqDiffuSeq/ckpts/en-zh"

$ckptSearchDirs = @(
    (Join-Path $ProjectRoot "SeqDiffuSeq\ckpts\en-zh"),
    $GdriveCkptDir
)

$latestCkpt = $null
foreach ($dir in $ckptSearchDirs) {
    if (Test-Path $dir) {
        $found = Get-ChildItem $dir -Filter "model*.pt" | Where-Object { $_.Name -notlike "*ema*" } | Sort-Object Name | Select-Object -Last 1
        if ($found) { $latestCkpt = $found.FullName; break }
    }
}
if ($latestCkpt) {
    Write-Host "    $(Split-Path $latestCkpt -Leaf)"
    Scp-File $latestCkpt "/root/SeqDiffuSeq/ckpts/en-zh/$(Split-Path $latestCkpt -Leaf)"
} else {
    Write-Host "    WARN: no model*.pt found — skipping"
}

$latestNpy = $null
foreach ($dir in $ckptSearchDirs) {
    if (Test-Path $dir) {
        $found = Get-ChildItem $dir -Filter "alpha_cumprod_step_*.npy" | Sort-Object Name | Select-Object -Last 1
        if ($found) { $latestNpy = $found.FullName; break }
    }
}
if ($latestNpy) {
    Write-Host "    $(Split-Path $latestNpy -Leaf)"
    Scp-File $latestNpy "/root/SeqDiffuSeq/ckpts/en-zh/$(Split-Path $latestNpy -Leaf)"
} else {
    Write-Host "    WARN: no alpha_cumprod_step_*.npy found — skipping"
}

# ── Pull results ──────────────────────────────────────────────────────────
if ($Pull) {
    $localResults = Join-Path $ProjectRoot "SeqDiffuSeq\results"
    New-Item -ItemType Directory -Path $localResults -Force | Out-Null
    Write-Host "==> Pulling results from $Remote..."
    scp -r "${Remote}:/root/SeqDiffuSeq/ckpts/en-zh/inference_out/*" $localResults
}

Write-Host "==> Done."

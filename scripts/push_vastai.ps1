# Push tokenizer + checkpoint to vast.ai. Code is cloned from git on the instance.
# Usage: .\scripts\push_vastai.ps1 [ssh-alias]

param(
    [string]$Remote = "vastai",
    [string]$GitRepo = "https://github.com/nhuvsbu03/NER_translation.git",
    [switch]$Pull
)

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────
$GdriveRoot    = "G:\My Drive\BUyen_Qnhu++\src\SeqDiffuSeq"
$GdriveTokDir  = "$GdriveRoot\data\en-zh"
$GdriveCkptDir = "$GdriveRoot\ckpts\en-zh"
# -------------------------------------------------------------------------

function Remote-Run($cmd) {
    ssh $Remote $cmd
}

function Scp-File($src, $dst) {
    scp -q "$src" "${Remote}:$dst"
}

# ── Setup instance --------------------------------------------------------
Write-Host "==> Installing git on instance..."
Remote-Run "apt-get install -y -q git"

Write-Host "==> Cloning repo on instance..."
Remote-Run "if [ -d /root/NER_translation/.git ]; then cd /root/NER_translation && git pull; else rm -rf /root/NER_translation && git clone $GitRepo /root/NER_translation; fi"

# ── Tokenizer -------------------------------------------------------------
Write-Host "==> Pushing tokenizer..."
Remote-Run "mkdir -p /root/NER_translation/SeqDiffuSeq/data/en-zh"

foreach ($fname in @("vocab.json", "merges.txt")) {
    $localPath = Join-Path $ProjectRoot "SeqDiffuSeq\data\en-zh\$fname"
    $drivePath = Join-Path $GdriveTokDir $fname

    if (Test-Path $localPath) {
        Write-Host "    $fname (local)"
        Scp-File $localPath "/root/NER_translation/SeqDiffuSeq/data/en-zh/$fname"
    } elseif (Test-Path $drivePath) {
        Write-Host "    $fname (Google Drive)"
        Scp-File $drivePath "/root/NER_translation/SeqDiffuSeq/data/en-zh/$fname"
    } else {
        Write-Host "    WARN: $fname not found -- skipping"
    }
}

# ── Checkpoint ------------------------------------------------------------
Write-Host "==> Pushing latest checkpoint..."
Remote-Run "mkdir -p /root/NER_translation/SeqDiffuSeq/ckpts/en-zh"

$ckptSearchDirs = @(
    (Join-Path $ProjectRoot "SeqDiffuSeq\ckpts\en-zh"),
    $GdriveCkptDir
)

$latestCkpt = $null
foreach ($dir in $ckptSearchDirs) {
    if (Test-Path $dir) {
        $found = Get-ChildItem $dir -Filter "model*.pt" -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -notlike "*ema*" } | Sort-Object Name | Select-Object -Last 1
        if ($found) { $latestCkpt = $found.FullName; break }
    }
}
if ($latestCkpt) {
    $ckptName = Split-Path $latestCkpt -Leaf
    Write-Host "    $ckptName"
    Scp-File $latestCkpt "/root/NER_translation/SeqDiffuSeq/ckpts/en-zh/$ckptName"
} else {
    Write-Host "    WARN: no model*.pt found -- skipping"
}

$latestNpy = $null
foreach ($dir in $ckptSearchDirs) {
    if (Test-Path $dir) {
        $found = Get-ChildItem $dir -Filter "alpha_cumprod_step_*.npy" -ErrorAction SilentlyContinue |
                 Sort-Object Name | Select-Object -Last 1
        if ($found) { $latestNpy = $found.FullName; break }
    }
}
if ($latestNpy) {
    $npyName = Split-Path $latestNpy -Leaf
    Write-Host "    $npyName"
    Scp-File $latestNpy "/root/NER_translation/SeqDiffuSeq/ckpts/en-zh/$npyName"
} else {
    Write-Host "    WARN: no alpha_cumprod_step_*.npy found -- skipping"
}

# ── Pull results ----------------------------------------------------------
if ($Pull) {
    $localResults = Join-Path $ProjectRoot "SeqDiffuSeq\results"
    New-Item -ItemType Directory -Path $localResults -Force | Out-Null
    Write-Host "==> Pulling results from $Remote..."
    scp -r "${Remote}:/root/NER_translation/SeqDiffuSeq/ckpts/en-zh/inference_out" "$localResults"
}

Write-Host "==> Done."

# Start vast.ai instance, wait until ready, update ~/.ssh/config Host "vastai".
# Usage (PowerShell): .\scripts\start_vastai.ps1
# Requires vastai CLI: pip install vastai

$INSTANCE_ID = "36014201"
$SSH_ALIAS   = "vastai"
$SSH_KEY     = "$env:USERPROFILE\.ssh\id_ed25519"
$SSH_CONFIG  = "$env:USERPROFILE\.ssh\config"

Write-Host "==> Starting instance $INSTANCE_ID..."
vastai start instance $INSTANCE_ID

Write-Host "==> Waiting for instance to be ready (checking every 10s)..."
$SSH_URL = ""
while ($true) {
    try {
        $SSH_URL = vastai ssh-url $INSTANCE_ID 2>$null
    } catch {}
    if ($SSH_URL -match "^ssh://") { break }
    Write-Host "    not ready yet, retrying..."
    Start-Sleep -Seconds 10
}

# Parse ssh://root@1.2.3.4:12345
$parsed = $SSH_URL -replace "ssh://root@", ""
$HOST_IP, $PORT = $parsed -split ":"

Write-Host "==> Instance ready: $SSH_URL"
Write-Host "    Host: $HOST_IP  Port: $PORT"

# Update or create ~/.ssh/config
if (Test-Path $SSH_CONFIG) {
    $content = Get-Content $SSH_CONFIG -Raw

    if ($content -match "(?ms)^Host $SSH_ALIAS\b.*?(?=^Host |\z)") {
        # Replace existing block
        $newBlock = "Host $SSH_ALIAS`n    HostName $HOST_IP`n    Port $PORT`n    User root`n    IdentityFile $SSH_KEY`n    StrictHostKeyChecking no`n"
        $content = $content -replace "(?ms)^Host $SSH_ALIAS\b.*?(?=^Host |\z)", $newBlock
        Set-Content $SSH_CONFIG $content
        Write-Host "==> Updated $SSH_CONFIG"
    } else {
        # Append new block
        Add-Content $SSH_CONFIG "`nHost $SSH_ALIAS`n    HostName $HOST_IP`n    Port $PORT`n    User root`n    IdentityFile $SSH_KEY`n    StrictHostKeyChecking no"
        Write-Host "==> Added Host $SSH_ALIAS to $SSH_CONFIG"
    }
} else {
    # Create config from scratch
    New-Item -ItemType File -Path $SSH_CONFIG -Force | Out-Null
    Set-Content $SSH_CONFIG "Host $SSH_ALIAS`n    HostName $HOST_IP`n    Port $PORT`n    User root`n    IdentityFile $SSH_KEY`n    StrictHostKeyChecking no"
    Write-Host "==> Created $SSH_CONFIG"
}

Write-Host ""
Write-Host "==> Ready. Connect with:"
Write-Host "    ssh $SSH_ALIAS"
Write-Host ""
Write-Host "==> Then push your code:"
Write-Host "    bash scripts/push_vastai.sh $SSH_ALIAS"

# Create a new vast.ai instance with a persistent volume.
# Usage (PowerShell): .\scripts\create_vastai.ps1

$IMAGE       = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"
$DISK_GB     = 30
$VOLUME_GB   = 50
$MOUNT_PATH  = "/workspace"
$VOLUME_LABEL = "seqdiffuseq-ckpts"

Write-Host "==> Searching for cheapest RTX 3090/4090 offer..."
$offersJson = vastai search offers `
  "gpu_name in [RTX_3090,RTX_4090] num_gpus=1 cuda_max_good>=11.8 disk_space>=$DISK_GB inet_up>=100 verified=true rentable=true" `
  --order 'dph_total' --raw 2>$null
$offers = $offersJson | ConvertFrom-Json
if (-not $offers -or $offers.Count -eq 0) {
    Write-Error "No matching offers found. Try relaxing filters on vast.ai website."
    exit 1
}
$OFFER_ID = $offers[0].id
Write-Host "    Found offer ID: $OFFER_ID"

Write-Host "==> Searching for cheapest volume location..."
$volJson = vastai search volumes "disk_space>=$VOLUME_GB verified=true" --order 'price_per_gb' --raw 2>$null
$volumes = $volJson | ConvertFrom-Json
if (-not $volumes -or $volumes.Count -eq 0) {
    Write-Error "No volume offers found."
    exit 1
}
$VOLUME_OFFER_ID = $volumes[0].id
Write-Host "    Found volume offer ID: $VOLUME_OFFER_ID"

Write-Host "==> Creating instance..."
$resultJson = vastai create instance $OFFER_ID `
  --image $IMAGE `
  --disk $DISK_GB `
  --jupyter --jupyter-lab --ssh --direct `
  --create-volume $VOLUME_OFFER_ID `
  --volume-size $VOLUME_GB `
  --mount-path $MOUNT_PATH `
  --volume-label $VOLUME_LABEL `
  --raw 2>$null
$result = $resultJson | ConvertFrom-Json
$INSTANCE_ID = $result.new_contract

if (-not $INSTANCE_ID) {
    Write-Error "Instance creation failed. Response: $resultJson"
    exit 1
}

Write-Host ""
Write-Host "==> Instance created! ID: $INSTANCE_ID"

# Auto-update INSTANCE_ID in both start scripts
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$shScript  = Join-Path $scriptDir "start_vastai.sh"
$ps1Script = Join-Path $scriptDir "start_vastai.ps1"

(Get-Content $shScript)  -replace 'INSTANCE_ID=\d+', "INSTANCE_ID=$INSTANCE_ID"  | Set-Content $shScript
(Get-Content $ps1Script) -replace '\$INSTANCE_ID\s*=\s*"\d+"', "`$INSTANCE_ID = `"$INSTANCE_ID`"" | Set-Content $ps1Script

Write-Host "==> Auto-updated INSTANCE_ID in start_vastai.sh and start_vastai.ps1"
Write-Host ""
Write-Host "==> Next: commit the updated scripts, then run:"
Write-Host "    .\scripts\start_vastai.ps1"

#!/usr/bin/env bash
# Create a new vast.ai instance with a persistent volume.
# Searches for cheapest RTX 3090/4090 with >=24 GB VRAM, creates a 50 GB
# volume at /workspace for checkpoints, then prints the instance ID to put
# into start_vastai.sh.
#
# Usage: ./scripts/create_vastai.sh
set -euo pipefail

IMAGE="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"
DISK_GB=30          # ephemeral instance disk (code + data)
VOLUME_GB=50        # persistent volume (checkpoints)
MOUNT_PATH="/workspace"
VOLUME_LABEL="seqdiffuseq-ckpts"

echo "==> Searching for cheapest RTX 3090/4090 offer…"
OFFER_ID=$(vastai search offers \
  "gpu_name in [RTX_3090,RTX_4090] num_gpus=1 cuda_max_good>=11.8 disk_space>=$DISK_GB inet_up>=100 verified=true rentable=true" \
  --order 'dph_total' \
  --raw 2>/dev/null \
  | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id']) if offers else exit(1)")

if [[ -z "$OFFER_ID" ]]; then
  echo "ERROR: No matching offers found. Try relaxing filters on vast.ai website."
  exit 1
fi
echo "    Found offer ID: $OFFER_ID"

echo "==> Searching for cheapest volume location…"
VOLUME_OFFER_ID=$(vastai search volumes \
  "disk_space>=$VOLUME_GB verified=true" \
  --order 'price_per_gb' \
  --raw 2>/dev/null \
  | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id']) if offers else exit(1)")

if [[ -z "$VOLUME_OFFER_ID" ]]; then
  echo "ERROR: No volume offers found."
  exit 1
fi
echo "    Found volume offer ID: $VOLUME_OFFER_ID"

echo "==> Creating instance…"
RESULT=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --jupyter \
  --jupyter-lab \
  --ssh \
  --direct \
  --create-volume "$VOLUME_OFFER_ID" \
  --volume-size "$VOLUME_GB" \
  --mount-path "$MOUNT_PATH" \
  --volume-label "$VOLUME_LABEL" \
  --raw 2>/dev/null)

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_contract',''))")

if [[ -z "$INSTANCE_ID" ]]; then
  echo "ERROR: Instance creation failed. Response:"
  echo "$RESULT"
  exit 1
fi

echo ""
echo "==> Instance created! ID: $INSTANCE_ID"
echo ""
echo "==> Next: update INSTANCE_ID in both start scripts:"
echo "    scripts/start_vastai.sh  → INSTANCE_ID=$INSTANCE_ID"
echo "    scripts/start_vastai.ps1 → \$INSTANCE_ID = \"$INSTANCE_ID\""
echo ""
echo "==> Then run:"
echo "    ./scripts/start_vastai.sh"

# Auto-update start_vastai.sh in the same repo
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

sed -i '' "s/^INSTANCE_ID=.*/INSTANCE_ID=$INSTANCE_ID/" "$SCRIPT_DIR/start_vastai.sh"
sed -i '' "s/^\\\$INSTANCE_ID = .*/\$INSTANCE_ID = \"$INSTANCE_ID\"/" "$SCRIPT_DIR/start_vastai.ps1"
echo "==> Auto-updated INSTANCE_ID in start_vastai.sh and start_vastai.ps1"

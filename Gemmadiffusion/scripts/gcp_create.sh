#!/usr/bin/env bash
# Create (or start, if it already exists) a GCE VM with a single L4 GPU for
# DiffusionGemma W4A16 inference. Mirrors scripts/start_vastai.sh but for GCP.
#
# Prerequisites (one-time):
#   brew install --cask google-cloud-sdk
#   gcloud init                                  # pick/create a project, set billing
#   gcloud services enable compute.googleapis.com
#   # If NVIDIA_L4_GPUS quota is 0 in your region, request an increase first:
#   # console.cloud.google.com -> IAM & Admin -> Quotas -> filter "NVIDIA L4 GPUs"
#
# Usage: ./scripts/gcp_create.sh
set -euo pipefail

INSTANCE_NAME=${INSTANCE_NAME:-diffgemma-l4}
ZONE=${ZONE:-us-central1-a}
MACHINE_TYPE=${MACHINE_TYPE:-g2-standard-8}   # 1x L4 (24GB), 8 vCPU, 32GB RAM
IMAGE_FAMILY=${IMAGE_FAMILY:-common-cu121-debian-11-py310}
IMAGE_PROJECT=${IMAGE_PROJECT:-ml-images}     # Deep Learning VM — CUDA/drivers preinstalled
BOOT_DISK_SIZE=${BOOT_DISK_SIZE:-100GB}
SSH_ALIAS=gcp

echo "==> Checking for existing instance $INSTANCE_NAME in $ZONE..."
if gcloud compute instances describe "$INSTANCE_NAME" --zone "$ZONE" &>/dev/null; then
    echo "==> Instance exists — starting it..."
    gcloud compute instances start "$INSTANCE_NAME" --zone "$ZONE"
else
    echo "==> Creating instance $INSTANCE_NAME ($MACHINE_TYPE, L4 GPU, $ZONE)..."
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone "$ZONE" \
        --machine-type "$MACHINE_TYPE" \
        --accelerator="type=nvidia-l4,count=1" \
        --maintenance-policy=TERMINATE \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type=pd-ssd
fi

echo "==> Waiting for SSH to come up..."
until gcloud compute ssh "$INSTANCE_NAME" --zone "$ZONE" --command="echo ready" &>/dev/null; do
    echo "    not ready yet, retrying..."
    sleep 10
done

echo ""
echo "==> Ready. Connect with:"
echo "    gcloud compute ssh $INSTANCE_NAME --zone $ZONE"
echo ""
echo "==> Then push your code:"
echo "    ./scripts/gcp_push.sh"
echo ""
echo "==> IMPORTANT — this instance bills per-second while running."
echo "    Stop it when done: ./scripts/gcp_stop.sh"

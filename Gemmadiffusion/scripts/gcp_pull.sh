#!/usr/bin/env bash
# Pull results/ back from the GCP instance. Mirrors pull_results.ps1.
# Usage: ./scripts/gcp_pull.sh
set -euo pipefail

INSTANCE_NAME=${INSTANCE_NAME:-diffgemma-l4}
ZONE=${ZONE:-us-central1-a}
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Pulling results from $INSTANCE_NAME:/root/Gemmadiffusion/results/ ..."
gcloud compute scp --recurse --zone "$ZONE" \
    "$INSTANCE_NAME:/root/Gemmadiffusion/results" "$LOCAL_DIR/"

echo "==> Done. Results in $LOCAL_DIR/results/"

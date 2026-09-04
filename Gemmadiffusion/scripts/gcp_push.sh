#!/usr/bin/env bash
# Push local Gemmadiffusion/ to the GCP instance. Mirrors push_vastai.ps1.
# Usage: ./scripts/gcp_push.sh
set -euo pipefail

INSTANCE_NAME=${INSTANCE_NAME:-diffgemma-l4}
ZONE=${ZONE:-us-central1-a}
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Pushing $LOCAL_DIR to $INSTANCE_NAME:/root/Gemmadiffusion/ ..."
gcloud compute scp --recurse \
    --zone "$ZONE" \
    --compress \
    "$LOCAL_DIR" "$INSTANCE_NAME:/root/" \
    2> >(grep -v '^Warning: Permanently added' >&2) \
    || true

# scp --recurse copies the directory itself; the excludes rsync gave us on
# vast.ai aren't available here, so drop heavy/untracked dirs after copying.
gcloud compute ssh "$INSTANCE_NAME" --zone "$ZONE" --command="
    rm -rf /root/Gemmadiffusion/data /root/Gemmadiffusion/results /root/Gemmadiffusion/.git
"

echo "==> Pushed. Next: run setup —"
echo "    gcloud compute ssh $INSTANCE_NAME --zone $ZONE --command='bash /root/Gemmadiffusion/scripts/gcp_setup.sh'"

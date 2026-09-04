#!/usr/bin/env bash
# Stop the GCP instance to halt billing (keeps the disk — restart with gcp_create.sh).
# Usage: ./scripts/gcp_stop.sh
#        ./scripts/gcp_stop.sh --delete   # also deletes the instance + disk entirely
set -euo pipefail

INSTANCE_NAME=${INSTANCE_NAME:-diffgemma-l4}
ZONE=${ZONE:-us-central1-a}

if [[ "${1:-}" == "--delete" ]]; then
    echo "==> Deleting instance $INSTANCE_NAME (irreversible — disk is destroyed too)..."
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE"
else
    echo "==> Stopping instance $INSTANCE_NAME (disk persists, compute billing stops)..."
    gcloud compute instances stop "$INSTANCE_NAME" --zone "$ZONE"
fi

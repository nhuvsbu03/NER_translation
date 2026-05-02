#!/usr/bin/env bash
# Start vast.ai instance, wait until ready, update ~/.ssh/config Host "vastai".
# Usage: ./scripts/start_vastai.sh
set -euo pipefail

INSTANCE_ID=36014085
SSH_ALIAS=vastai

echo "==> Starting instance $INSTANCE_ID…"
vastai start instance "$INSTANCE_ID"

echo "==> Waiting for instance to be ready (checking every 10s)…"
SSH_URL=""
while true; do
    SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
    if [[ "$SSH_URL" == ssh://* ]]; then
        break
    fi
    echo "    not ready yet, retrying…"
    sleep 10
done

# Parse ssh://root@1.2.3.4:12345
HOST=$(echo "$SSH_URL" | sed 's|ssh://root@||' | cut -d: -f1)
PORT=$(echo "$SSH_URL" | cut -d: -f3)

echo "==> Instance ready: $SSH_URL"
echo "    Host: $HOST  Port: $PORT"

# Update ~/.ssh/config — replace the vastai block
SSH_CONFIG="$HOME/.ssh/config"
if grep -q "^Host $SSH_ALIAS" "$SSH_CONFIG" 2>/dev/null; then
    # Replace HostName and Port lines inside the vastai block
    awk -v alias="$SSH_ALIAS" -v host="$HOST" -v port="$PORT" '
        /^Host / { in_block = ($2 == alias) }
        in_block && /^[[:space:]]+HostName / { print "    HostName " host; next }
        in_block && /^[[:space:]]+Port /     { print "    Port " port; next }
        { print }
    ' "$SSH_CONFIG" > "$SSH_CONFIG.tmp" && mv "$SSH_CONFIG.tmp" "$SSH_CONFIG"
    echo "==> Updated $SSH_CONFIG"
else
    # Append new block
    cat >> "$SSH_CONFIG" <<EOF

Host $SSH_ALIAS
    HostName $HOST
    Port $PORT
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
EOF
    echo "==> Added Host $SSH_ALIAS to $SSH_CONFIG"
fi

echo ""
echo "==> Ready. Connect with:"
echo "    ssh $SSH_ALIAS"
echo ""
echo "==> Then push your code:"
echo "    ./scripts/push_vastai.sh $SSH_ALIAS"

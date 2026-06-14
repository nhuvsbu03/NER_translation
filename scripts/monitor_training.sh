#!/usr/bin/env bash
# Monitor training and auto-restart on crash. Also cleans disk before it fills.
# Runs in a SEPARATE tmux session ('monitor') so it survives training restarts.
#
# Usage (from /root/NER_translation):
#   bash scripts/monitor_training.sh [en-ru]
#
# Launch automatically via train_en_ru.sh, or manually:
#   tmux new-session -d -s monitor "bash /root/NER_translation/scripts/monitor_training.sh en-ru"

PAIR="${1:-en-ru}"
PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/$PAIR"
LOG_DIR="$CKPT_DIR/log"
MONITOR_LOG="$LOG_DIR/monitor.log"

DISK_WARN_PCT=75   # log a warning
DISK_CRIT_PCT=88   # trigger cleanup
CHECK_INTERVAL=60  # seconds between checks
RESTART_COOLDOWN=180  # seconds to wait before restarting after a crash

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

cleanup_disk() {
    log "--- Disk cleanup start ---"
    # Keep only the latest model checkpoint (only needed for resuming)
    local latest_model
    latest_model=$(ls "$CKPT_DIR"/model*.pt 2>/dev/null | grep -v ema | sort | tail -1 || true)
    if [ -n "$latest_model" ]; then
        ls "$CKPT_DIR"/model*.pt 2>/dev/null | grep -v ema | sort | grep -v "$latest_model" | xargs rm -f 2>/dev/null || true
        log "Kept model checkpoint: $(basename "$latest_model"), deleted older ones"
    fi
    # Keep only the latest optimizer state (paired with model checkpoint, ~350MB each)
    ls "$CKPT_DIR"/opt*.pt 2>/dev/null | sort | head -n -1 | xargs rm -f 2>/dev/null || true
    # Keep only last 2 EMA checkpoints (~785MB each) — prevents disk-full panic-deletion of best EMA
    ls "$CKPT_DIR"/ema_0.9999_*.pt 2>/dev/null | sort | head -n -2 | xargs rm -f 2>/dev/null || true
    # Keep only last 3 noise schedule files
    ls "$CKPT_DIR"/alpha_cumprod_step_*.npy 2>/dev/null | sort | head -n -3 | xargs rm -f 2>/dev/null || true
    ls "$CKPT_DIR"/loss_count_*.npy         2>/dev/null | sort | head -n -3 | xargs rm -f 2>/dev/null || true
    ls "$CKPT_DIR"/loss_step_*.npy          2>/dev/null | sort | head -n -3 | xargs rm -f 2>/dev/null || true
    local pct
    pct=$(df /root | tail -1 | awk '{gsub(/%/,"",$5); print $5}')
    log "--- Disk cleanup done: ${pct}% used ---"
}

log "=== Monitor started for $PAIR (PID $$) ==="
log "    Disk warn: ${DISK_WARN_PCT}%  |  Disk cleanup: ${DISK_CRIT_PCT}%  |  Check every: ${CHECK_INTERVAL}s"

LAST_RESTART=0

while true; do
    NOW=$(date +%s)

    # ── Disk check ────────────────────────────────────────────────────────────
    DISK_PCT=$(df /root | tail -1 | awk '{gsub(/%/,"",$5); print $5}' || echo 0)
    if [ "$DISK_PCT" -ge "$DISK_CRIT_PCT" ]; then
        log "CRITICAL: disk at ${DISK_PCT}% — running cleanup"
        cleanup_disk
    elif [ "$DISK_PCT" -ge "$DISK_WARN_PCT" ]; then
        log "WARNING: disk at ${DISK_PCT}%"
    fi

    # ── Collapse detection: grad_norm early stopping ───────────────────────────
    if [ -f "$LOG_DIR/train.log" ]; then
        CURRENT_STEP=$(grep "| step " "$LOG_DIR/train.log" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
        if [ "${CURRENT_STEP:-0}" -ge 20000 ] && [ ! -f "$LOG_DIR/COLLAPSE_FLAG" ]; then
            LOW_COUNT=$(grep "| grad_norm |" "$LOG_DIR/train.log" 2>/dev/null | tail -10 \
                | awk '{v=$3+0; if(v < 0.005) c++} END {print c+0}')
            if [ "${LOW_COUNT:-0}" -ge 8 ]; then
                log "COLLAPSE DETECTED: grad_norm < 0.005 in ${LOW_COUNT}/10 recent readings — killing training"
                tmux kill-session -t train 2>/dev/null || true
                echo "COLLAPSE_DETECTED $(date '+%Y-%m-%d %H:%M:%S') step=${CURRENT_STEP}" > "$LOG_DIR/COLLAPSE_FLAG"
                log "Training killed. Best checkpoint in $CKPT_DIR. Pull results before instance expires."
            fi
        fi
    fi

    # ── Process check ─────────────────────────────────────────────────────────
    if ! pgrep -f "python3.*main.py" > /dev/null 2>&1; then
        if [ -f "$LOG_DIR/COLLAPSE_FLAG" ]; then
            log "Training not running (collapse detected — not restarting)"
        else
            SINCE=$(( NOW - LAST_RESTART ))
            if [ "$SINCE" -lt "$RESTART_COOLDOWN" ]; then
                log "Training not running (cooldown: ${SINCE}/${RESTART_COOLDOWN}s — waiting)"
            else
                log "Training process dead — auto-restarting..."
                cd "$PROJECT_ROOT"
                bash scripts/train_en_ru.sh >> "$MONITOR_LOG" 2>&1 || true
                LAST_RESTART=$(date +%s)
                log "Restart issued."
            fi
        fi
    fi

    sleep "$CHECK_INTERVAL"
done

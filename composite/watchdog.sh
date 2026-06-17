#!/usr/bin/env bash
# watchdog.sh — Keeps the pipeline running overnight.
# If run_pipeline.sh exits with a non-zero code, waits 30s then restarts.
# Run inside a tmux session: tmux new -s train 'bash ~/model/composite/watchdog.sh'

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG_LOG="${SCRIPT_DIR}/logs/watchdog.log"
mkdir -p "${SCRIPT_DIR}/logs"

# Memory fragmentation fix for V100s
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] $*" | tee -a "${WATCHDOG_LOG}"
}

ATTEMPT=0
MAX_ATTEMPTS=5

log "Watchdog started. Will run pipeline up to ${MAX_ATTEMPTS} attempts."

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    log "========== ATTEMPT ${ATTEMPT}/${MAX_ATTEMPTS} =========="

    bash "${SCRIPT_DIR}/run_pipeline.sh"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "Pipeline COMPLETED SUCCESSFULLY on attempt ${ATTEMPT}. Watchdog exiting."
        exit 0
    fi

    log "Pipeline exited with code ${EXIT_CODE}."

    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        log "Max attempts (${MAX_ATTEMPTS}) reached. Stopping."
        exit 1
    fi

    log "Waiting 30s before restart..."
    sleep 30
    log "Restarting pipeline..."
done

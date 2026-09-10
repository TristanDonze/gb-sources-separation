#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$REPO_ROOT/experiments/evaluation_nohup.log"
PID_FILE="$REPO_ROOT/experiments/evaluation_nohup.pid"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ "${1:-}" == "--worker" ]]; then
    cd "$REPO_ROOT"
    "$PYTHON" -u experiments/prepare_protocol_results.py
    "$PYTHON" -u experiments/prepare_wrong_k_results.py
    exit 0
fi

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(<"$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Evaluation already running with PID $OLD_PID"
        exit 1
    fi
fi

nohup "$0" --worker >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"
echo "Evaluation started with PID $PID"
echo "Log: $LOG_FILE"

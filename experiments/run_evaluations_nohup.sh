#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$REPO_ROOT/experiments/evaluation_nohup.log"
PID_FILE="$REPO_ROOT/experiments/evaluation_nohup.pid"
DEFAULT_OUTPUT_DIR="$REPO_ROOT/experiments/results/run_20260903_170728_epoch_23_medoid"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON=("$REPO_ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run --frozen python)
else
    echo "No project Python found: create .venv or install uv" >&2
    exit 1
fi

if [[ "${1:-}" == "--worker" ]]; then
    OUTPUT_DIR="$2"
    cd "$REPO_ROOT"
    "${PYTHON[@]}" -u experiments/prepare_protocol_results.py --output-dir "$OUTPUT_DIR"
    "${PYTHON[@]}" -u experiments/prepare_wrong_k_results.py --output-dir "$OUTPUT_DIR"
    exit 0
fi

OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(<"$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Evaluation already running with PID $OLD_PID"
        exit 1
    fi
fi

nohup "$0" --worker "$OUTPUT_DIR" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"
echo "Evaluation started with PID $PID"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"

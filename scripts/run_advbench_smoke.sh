#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-advbench_smoke_depth5_honeypot}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set in this terminal."
    exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set in this terminal."
    exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
cd "$PROJECT_DIR"
"$PYTHON_BIN" scripts/run_advbench_50.py \
    --limit 1 \
    --profile pilot-honeypot-depth5 \
    --run-name "$RUN_NAME" \
    --conditions no_defense cot_only \
    --job-attempts 1

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT="${1:-5}"
START="${2:-0}"
RUN_NAME="${3:-meaningful_pilot}"

for KEY_NAME in OPENAI_API_KEY OPENROUTER_API_KEY; do
    if [[ -z "${!KEY_NAME:-}" ]]; then
        echo "$KEY_NAME is not set."
        exit 2
    fi
done

export WANDB_MODE="${WANDB_MODE:-offline}"

cd "$PROJECT_DIR"

python3 scripts/run_benchmark_subset.py \
    --limit "$LIMIT" \
    --start "$START" \
    --run-name "$RUN_NAME"

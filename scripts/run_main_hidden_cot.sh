#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GOAL="${1:-}"
TARGET_PREFIX="${2:-}"
RUN_NAME="${3:-main_hidden_cot}"

if [[ -z "$GOAL" || -z "$TARGET_PREFIX" ]]; then
    echo 'Usage: scripts/run_main_hidden_cot.sh "objective" "objective-specific target prefix" [run name]'
    exit 2
fi

for KEY_NAME in OPENAI_API_KEY OPENROUTER_API_KEY; do
    if [[ -z "${!KEY_NAME:-}" ]]; then
        echo "$KEY_NAME is not set."
        exit 2
    fi
done

export WANDB_MODE="${WANDB_MODE:-offline}"

cd "$PROJECT_DIR"

python3 main.py \
    --profile main-hidden-cot \
    --goal "$GOAL" \
    --target-str "$TARGET_PREFIX" \
    --output "results/${RUN_NAME}.jsonl" \
    -v

echo "Experiment trace: $PROJECT_DIR/results/${RUN_NAME}.jsonl"

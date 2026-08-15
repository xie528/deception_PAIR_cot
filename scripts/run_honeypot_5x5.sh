#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
python3 scripts/run_clearharm_20.py \
    --limit 5 \
    --profile pilot-honeypot-depth5 \
    --run-name clearharm_5_depth5_honeypot \
    --conditions no_defense cot_only \
    --job-attempts 3

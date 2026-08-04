#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set in this terminal."
    exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
cd "$PROJECT_DIR"
python3 scripts/run_clearharm_20.py \
    --limit 10 \
    --profile pilot-strong-depth10 \
    --run-name clearharm_10_depth10_ablation \
    --job-attempts 3

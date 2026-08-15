#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-clearharm_10_depth5_honeypot}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set in this terminal."
    exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set in this terminal."
    exit 2
fi

# A path may be supplied as the first argument. Otherwise, try the repository
# copy first and the original local Pilot0 layout second.
if [[ $# -ge 1 ]]; then
    DATASET="$1"
elif [[ -f "$PROJECT_DIR/data/ClearHarm_development.json" ]]; then
    DATASET="$PROJECT_DIR/data/ClearHarm_development.json"
else
    DATASET="$PROJECT_DIR/../Pilot0/data/ClearHarm_development.json"
fi

if [[ ! -f "$DATASET" ]]; then
    echo "ClearHarm dataset not found: $DATASET"
    echo "Usage: bash scripts/run_honeypot_10x5.sh /path/to/ClearHarm_development.json"
    exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
cd "$PROJECT_DIR"
"$PYTHON_BIN" scripts/run_clearharm_20.py \
    --dataset "$DATASET" \
    --limit 10 \
    --profile pilot-honeypot-depth5 \
    --run-name "$RUN_NAME" \
    --conditions no_defense cot_only \
    --job-attempts 3

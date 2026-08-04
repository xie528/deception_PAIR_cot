#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOAL="${1:-}"
TARGET_PREFIX="${2:-Sure, here is the requested information}"
RUN_NAME="${3:-pilot_strong}"
DEFENSE_MODE="${4:-with_hidden_cot}"
PROFILE="${5:-pilot-strong}"

if [[ -z "$GOAL" ]]; then
    echo 'Usage: scripts/run_pilot_strong.sh "objective" ["target prefix"] [run name] [with_hidden_cot|without_hidden_cot] [pilot-strong|pilot-strong-smoke]'
    exit 2
fi
if [[ "$PROFILE" != "pilot-strong" && "$PROFILE" != "pilot-strong-smoke" ]]; then
    echo "Invalid profile: $PROFILE"
    exit 2
fi
if [[ "$DEFENSE_MODE" != "with_hidden_cot" && "$DEFENSE_MODE" != "without_hidden_cot" ]]; then
    echo "Invalid defense mode: $DEFENSE_MODE"
    exit 2
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set."
    exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
cd "$PROJECT_DIR"
python3 main.py \
    --profile "$PROFILE" \
    --goal "$GOAL" \
    --target-str "$TARGET_PREFIX" \
    --defense-mode "$DEFENSE_MODE" \
    --output "results/${RUN_NAME}.jsonl" \
    -v

echo "Experiment trace: $PROJECT_DIR/results/${RUN_NAME}.jsonl"

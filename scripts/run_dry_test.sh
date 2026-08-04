#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

python3 -m unittest tests.test_hidden_cot_defense

python3 main.py \
    --dry-run-hidden-cot \
    --n-streams 2 \
    --n-iterations 3 \
    --max-regenerations 3 \
    --output results/pair_hidden_cot_dry_test.jsonl \
    --overwrite

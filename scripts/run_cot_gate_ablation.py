#!/usr/bin/env python3
"""Replay a PAIR trace with the Hidden-CoT judge/gate removed."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judges import GPTJudge
from reasoning_target import OpenAICompatibleReasoningTarget


def load_trace(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--goal", required=True)
    parser.add_argument(
        "--target-prefix",
        default="Sure, here is the requested information",
    )
    parser.add_argument("--target-model", default="qwen/qwen3-8b")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    args = parser.parse_args()

    source_rows = load_trace(args.input)
    target = OpenAICompatibleReasoningTarget(
        model=args.target_model,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        temperature=0.0,
        max_tokens=512,
        reasoning_max_tokens=1024,
    )

    generations = [
        target.generate_with_reasoning(row["prompt"], regeneration_count=0)
        for row in source_rows
    ]
    responses = [generation.response for generation in generations]

    judge_args = SimpleNamespace(
        judge_max_n_tokens=200,
        judge_temperature=0.0,
        judge_model=args.judge_model,
        goal=args.goal,
        target_str=args.target_prefix,
    )
    scores = GPTJudge(judge_args).score(
        [row["prompt"] for row in source_rows],
        responses,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for source, generation, score in zip(source_rows, generations, scores):
            json.dump(
                {
                    "iteration": source["iteration"],
                    "stream": source["stream"],
                    "prompt": source["prompt"],
                    "released_response": generation.response,
                    "reasoning": generation.reasoning,
                    "pair_score": score,
                    "ablation": "no_cot_judge_or_regeneration",
                    "source_pair_score": source["pair_score"],
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")

    print(f"Wrote {len(source_rows)} paired records to {args.output}")


if __name__ == "__main__":
    main()

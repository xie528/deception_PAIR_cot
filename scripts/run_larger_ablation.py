#!/usr/bin/env python3
"""Run a larger, two-condition Hidden-CoT PAIR pilot."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_INDICES = "0,3,9,12,14"
CONDITIONS = ("with_hidden_cot", "without_hidden_cot")


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_dir / "data" / "harmful_behaviors_custom.csv",
    )
    parser.add_argument(
        "--indices",
        default=DEFAULT_INDICES,
        help="Comma-separated zero-based CSV row indices.",
    )
    parser.add_argument("--run-name", default="large_hidden_cot_ablation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    rows = load_rows(args.dataset)
    indices = [int(value.strip()) for value in args.indices.split(",")]
    selected = [(index, rows[index]) for index in indices]
    run_dir = project_dir / "results" / args.run_name

    jobs = []
    for position, (source_index, row) in enumerate(selected):
        # Alternate order to reduce systematic provider/time-order effects.
        condition_order = CONDITIONS if position % 2 == 0 else CONDITIONS[::-1]
        for condition in condition_order:
            output = run_dir / condition / f"objective_{source_index:03d}.jsonl"
            jobs.append((source_index, row, condition, output))

    existing = [str(output) for *_, output in jobs if output.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to mix this run with existing traces:\n"
            + "\n".join(existing)
        )
    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in this terminal")

    manifest = {
        "profile": "pilot-strong-large",
        "dataset": str(args.dataset),
        "indices": indices,
        "n_objectives": len(selected),
        "n_streams": 3,
        "n_iterations": 5,
        "conditions": list(CONDITIONS),
        "objectives": [
            {
                "source_index": index,
                "goal": row["goal"],
                "target": row["target"],
                "category": row["category"],
            }
            for index, row in selected
        ],
    }

    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        print("\nDry run only; no files created and no API calls made.")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    for number, (source_index, row, condition, output) in enumerate(jobs, 1):
        command = [
            sys.executable,
            str(project_dir / "main.py"),
            "--profile",
            "pilot-strong-large",
            "--goal",
            row["goal"],
            "--target-str",
            row["target"],
            "--defense-mode",
            condition,
            "--category",
            row["category"],
            "--index",
            str(source_index),
            "--output",
            str(output),
            "-v",
        ]
        print(
            f"\n[{number}/{len(jobs)}] objective={source_index} "
            f"condition={condition}\noutput={output}",
            flush=True,
        )
        if args.dry_run:
            print("command:", " ".join(command))
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=project_dir, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent
    default_dataset = (
        project_dir.parent / "Pilot0" / "data" / "ClearHarm_development.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--objective-key", default="question")
    parser.add_argument(
        "--target-prefix",
        default="Sure, here is the requested information",
    )
    parser.add_argument("--run-name", default="meaningful_pilot")
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent

    with args.dataset.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    selected = records[args.start : args.start + args.limit]
    if not selected:
        raise ValueError("No benchmark records selected")

    for offset, record in enumerate(selected):
        source_index = args.start + offset
        objective = str(record[args.objective_key])
        target_prefix = str(
            record.get("target_prefix", args.target_prefix)
        )
        output = (
            project_dir
            / "results"
            / f"{args.run_name}_{source_index:03d}.jsonl"
        )
        command = [
            sys.executable,
            str(project_dir / "main.py"),
            "--profile",
            "meaningful-pilot",
            "--goal",
            objective,
            "--target-str",
            target_prefix,
            "--output",
            str(output),
            "-v",
        ]
        print(
            f"\nRunning objective {offset + 1}/{len(selected)} "
            f"(source index {source_index})"
        )
        subprocess.run(command, cwd=project_dir, check=True)


if __name__ == "__main__":
    main()

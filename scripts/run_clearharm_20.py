#!/usr/bin/env python3
"""Run a resumable two-condition PAIR experiment on ClearHarm."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment_profiles import PROFILES


CONDITIONS = ("with_hidden_cot", "without_hidden_cot")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_DIR.parent / "Pilot0" / "data" / "ClearHarm_development.json",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--run-name", default="clearharm_20_additive_ablation")
    parser.add_argument("--profile", default="pilot-strong-large")
    parser.add_argument("--job-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    project_dir = PROJECT_DIR
    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.profile not in PROFILES:
        raise ValueError(f"Unknown profile: {args.profile}")
    profile = PROFILES[args.profile]
    selected = records[args.start : args.start + args.limit]
    if len(selected) != args.limit:
        raise ValueError(
            f"Requested {args.limit} rows from {args.start}, found {len(selected)}"
        )
    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in this terminal")

    run_dir = project_dir / "results" / args.run_name
    manifest = {
        "dataset": str(args.dataset),
        "start": args.start,
        "limit": args.limit,
        "profile": args.profile,
        "comparison": {
            "without_hidden_cot": "output gate only",
            "with_hidden_cot": "output gate plus Hidden-CoT gate",
        },
        "n_streams": profile["n_streams"],
        "n_iterations": profile["n_iterations"],
        "attacker_strategy": (
            "role-playing"
            if profile["n_streams"] == 1
            else "PAIR round-robin strategy prompts"
        ),
        "objectives": selected,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; no files created and no API calls made.")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("Existing run manifest does not match this invocation")
        else:
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

    jobs = []
    for position, record in enumerate(selected):
        source_index = args.start + position
        order = CONDITIONS if position % 2 == 0 else CONDITIONS[::-1]
        for condition in order:
            jobs.append((source_index, record, condition))

    for job_number, (source_index, record, condition) in enumerate(jobs, 1):
        output_dir = run_dir / condition
        output = output_dir / f"objective_{source_index:03d}.jsonl"
        work = output.with_suffix(".work.jsonl")
        print(
            f"\n[{job_number}/{len(jobs)}] objective={source_index} "
            f"condition={condition}",
            flush=True,
        )
        if output.exists():
            print(f"Already complete; skipping {output}", flush=True)
            continue

        command = [
            sys.executable,
            str(project_dir / "main.py"),
            "--profile",
            args.profile,
            "--goal",
            str(record["question"]),
            "--target-str",
            "Sure, here is the requested information",
            "--defense-mode",
            condition,
            "--index",
            str(source_index),
            "--output",
            str(work),
            "-v",
        ]
        if args.dry_run:
            print("command:", " ".join(command), flush=True)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        if work.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work.rename(work.with_name(f"{work.stem}.orphaned_{stamp}.jsonl"))

        last_error = None
        for attempt in range(1, args.job_attempts + 1):
            print(f"Job attempt {attempt}/{args.job_attempts}", flush=True)
            try:
                subprocess.run(command, cwd=project_dir, check=True)
                work.replace(output)
                print(f"Completed: {output}", flush=True)
                last_error = None
                break
            except subprocess.CalledProcessError as error:
                last_error = error
                if work.exists():
                    failed = work.with_name(
                        f"{work.stem}.failed_attempt_{attempt}.jsonl"
                    )
                    work.replace(failed)
                print(f"Job failed on attempt {attempt}.", flush=True)
        if last_error is not None:
            failure = {
                "objective": source_index,
                "condition": condition,
                "attempts": args.job_attempts,
                "error": repr(last_error),
            }
            with (run_dir / "failed_jobs.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(failure) + "\n")
            print(
                f"Skipping objective {source_index}, condition {condition} "
                f"after {args.job_attempts} failed job attempts.",
                flush=True,
            )


if __name__ == "__main__":
    main()

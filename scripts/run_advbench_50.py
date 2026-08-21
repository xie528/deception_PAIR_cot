#!/usr/bin/env python3
"""Run the current two-condition honeypot experiment on PAIR's AdvBench-50."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment_profiles import PROFILES


DEFAULT_CONDITIONS = ("no_defense", "cot_only")
VALID_CONDITIONS = (
    "no_defense",
    "cot_only",
    "output_only",
    "output_plus_cot",
)
REQUIRED_COLUMNS = {"goal", "target", "category", "Original index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable PAIR experiment over the published 50-behavior "
            "AdvBench subset bundled with this repository."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_DIR / "data" / "harmful_behaviors_custom.csv",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--run-name", default="advbench_50_depth5_honeypot")
    parser.add_argument("--profile", default="pilot-honeypot-depth5")
    parser.add_argument("--job-attempts", type=int, default=3)
    parser.add_argument(
        "--conditions",
        nargs=2,
        choices=VALID_CONDITIONS,
        default=DEFAULT_CONDITIONS,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_dataset(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}"
            )
        return list(reader), fieldnames


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")

    if args.start < 0 or args.limit < 1 or args.job_attempts < 1:
        raise ValueError("start must be >= 0; limit and job-attempts must be >= 1")
    if args.profile not in PROFILES:
        raise ValueError(f"Unknown profile: {args.profile}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"AdvBench dataset not found: {args.dataset}")

    rows, fieldnames = read_dataset(args.dataset)
    selected = rows[args.start : args.start + args.limit]
    if len(selected) != args.limit:
        raise ValueError(
            f"Requested {args.limit} rows from {args.start}, found {len(selected)}"
        )

    profile = PROFILES[args.profile]
    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in this terminal")
    if (
        not args.dry_run
        and any(condition != "no_defense" for condition in args.conditions)
        and not os.getenv(profile.get("cot_judge_api_key_env", "OPENAI_API_KEY"))
    ):
        key_name = profile.get("cot_judge_api_key_env", "OPENAI_API_KEY")
        raise RuntimeError(f"{key_name} is not set in this terminal")

    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    run_dir = PROJECT_DIR / "results" / args.run_name
    manifest = {
        "benchmark": "PAIR custom 50-behavior subset of AdvBench",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "dataset_columns": fieldnames,
        "start": args.start,
        "limit": args.limit,
        "profile": args.profile,
        "conditions": list(args.conditions),
        "comparison": {
            "no_defense": (
                "same honeypot target; release first candidate without a gate"
            ),
            "cot_only": "Hidden-CoT gate controls release and regeneration",
            "output_only": "output gate controls release and regeneration",
            "output_plus_cot": (
                "both output and Hidden-CoT gates control release"
            ),
        },
        "n_streams": profile["n_streams"],
        "n_iterations": profile["n_iterations"],
        "max_regenerations": profile.get("max_regenerations", 0),
        "models": {
            "attacker": profile["attack_model"],
            "target": profile["target_model"],
            "pair_judge": profile["pair_judge_model"],
            "cot_judge": profile["cot_judge_model"],
        },
        "objectives": [
            {
                "source_index": args.start + position,
                "advbench_original_index": row["Original index"],
                "goal": row["goal"],
                "target": row["target"],
                "category": row["category"],
            }
            for position, row in enumerate(selected)
        ],
    }

    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; no files created and no API calls made.", flush=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError(
                    "Existing run manifest does not match this invocation. "
                    "Use the original command to resume or choose a new run name."
                )
        else:
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

    jobs: list[tuple[int, dict[str, str], str]] = []
    for position, row in enumerate(selected):
        source_index = args.start + position
        conditions = tuple(args.conditions)
        order = conditions if position % 2 == 0 else conditions[::-1]
        for condition in order:
            jobs.append((source_index, row, condition))

    failed_jobs = 0
    for job_number, (source_index, row, condition) in enumerate(jobs, 1):
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
            str(PROJECT_DIR / "main.py"),
            "--profile",
            args.profile,
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
            str(work),
            "-v",
        ]
        if args.dry_run:
            print("command:", shlex.join(command), flush=True)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        if work.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work.rename(work.with_name(f"{work.stem}.orphaned_{stamp}.jsonl"))

        last_error: Exception | None = None
        for attempt in range(1, args.job_attempts + 1):
            print(f"Job attempt {attempt}/{args.job_attempts}", flush=True)
            try:
                subprocess.run(command, cwd=PROJECT_DIR, check=True)
                if not work.exists() or work.stat().st_size == 0:
                    raise RuntimeError("Job returned successfully but wrote no trace")
                work.replace(output)
                print(f"Completed: {output}", flush=True)
                last_error = None
                break
            except (subprocess.CalledProcessError, RuntimeError) as error:
                last_error = error
                if work.exists():
                    failed = work.with_name(
                        f"{work.stem}.failed_attempt_{attempt}.jsonl"
                    )
                    work.replace(failed)
                print(f"Job failed on attempt {attempt}: {error!r}", flush=True)

        if last_error is not None:
            failed_jobs += 1
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
                f"after {args.job_attempts} failed attempts.",
                flush=True,
            )

    if args.dry_run:
        print(f"\nValidated {len(jobs)} planned jobs with no API calls.")
    elif failed_jobs:
        raise SystemExit(
            f"Run finished with {failed_jobs} failed jobs. Re-run the same "
            "command to retry missing finalized traces."
        )
    else:
        print(f"\nAll {len(jobs)} jobs are finalized in {run_dir}")


if __name__ == "__main__":
    main()

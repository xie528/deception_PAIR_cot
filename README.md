# PAIR Hidden-CoT Defense

This repository runs PAIR against a honeypot target and compares two conditions:

- `no_defense`: release the target's first honeypot response without a monitoring gate.
- `cot_only`: judge only the target's returned reasoning trace and regenerate when its plan is unsafe or uncertain.

The current experiment uses one PAIR stream, up to five iterations per objective, and the same models and honeypot prompt in both conditions.

> Research use only. Run this code only on models and infrastructure you are authorized to test. Generated traces may contain unsafe material. Never commit API keys.

## 1. Clone and install

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/xie528/deception_PAIR_cot.git
cd deception_PAIR_cot

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hidden-cot.txt
```

RunPod and other Linux machines use the same commands. On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## 2. Add API keys

The experiment requires:

- `OPENROUTER_API_KEY` for the PAIR attacker, Qwen target, and PAIR evaluator.
- `OPENAI_API_KEY` for the GPT Hidden-CoT judge.

Enter both keys without displaying them:

```bash
read -s "OPENROUTER_API_KEY?OpenRouter key: "; export OPENROUTER_API_KEY; echo
read -s "OPENAI_API_KEY?OpenAI key: "; export OPENAI_API_KEY; echo
```

The keys remain available only in that terminal session.

## 3. Provide the ClearHarm dataset file

This experiment reads its objectives from a file named `ClearHarm_development.json`. That file is not currently included in this GitHub repository.

The original development machine already has the file at:

```text
../Pilot0/data/ClearHarm_development.json
```

When the launcher is run on that machine without a dataset argument, it automatically checks that location.

A new user must obtain an authorized copy of `ClearHarm_development.json`, upload or copy it onto the machine running the experiment, and pass its path to the launcher. For example, a RunPod user can upload the file to:

```text
/workspace/ClearHarm_development.json
```

Confirm that the file exists before running:

```bash
ls -l /workspace/ClearHarm_development.json
```

Do not assume that cloning this repository also downloads ClearHarm. If the dataset is later added to this repository under `data/ClearHarm_development.json`, the launcher will detect that location automatically.

## 4. Preview the experiment without API calls

Replace `/absolute/path/to/ClearHarm_development.json` with the location of the actual dataset file on the current machine:

```bash
python scripts/run_clearharm_20.py \
  --dataset /absolute/path/to/ClearHarm_development.json \
  --limit 10 \
  --profile pilot-honeypot-depth5 \
  --run-name clearharm_10_depth5_honeypot \
  --conditions no_defense cot_only \
  --job-attempts 3 \
  --dry-run
```

The preview lists all jobs but does not call any models or spend API credits.

## 5. Run the 10-objective experiment

On the original development machine, where the dataset remains in `../Pilot0/data/`, run:

```bash
bash scripts/run_honeypot_10x5.sh
```

On RunPod or another new machine, provide the dataset path explicitly:

```bash
bash scripts/run_honeypot_10x5.sh \
  /workspace/ClearHarm_development.json
```

This runs:

- 10 ClearHarm objectives;
- 2 conditions (`no_defense` and `cot_only`);
- 1 PAIR stream per objective;
- up to 5 PAIR iterations per stream.

The runner skips finalized objective files when restarted, so the same command can resume an interrupted run.

To use a different result-folder name:

```bash
RUN_NAME=my_experiment \
bash scripts/run_honeypot_10x5.sh \
  /absolute/path/to/ClearHarm_development.json
```

## 6. Find the results

The default output is:

```text
results/clearharm_10_depth5_honeypot/
├── manifest.json
├── no_defense/
│   └── objective_NNN.jsonl
└── cot_only/
    └── objective_NNN.jsonl
```

Each JSONL row records one PAIR iteration, including the attacker prompt, released response, PAIR score, and defense metadata. CoT traces and labels are included for the `cot_only` condition.

## 7. Analyze a completed experiment

```bash
python scripts/analyze_clearharm_ablation.py \
  results/clearharm_10_depth5_honeypot
```

The generated `analysis/` directory contains:

- `condition_summary.csv`: aggregate comparison between conditions;
- `objective_summary.csv`: per-objective results;
- `manual_review_score10.csv`: responses that received PAIR score 10;
- `manual_review_cot_only.csv`: attempts rejected by the CoT gate;
- `report.json`: machine-readable experiment summary.

PAIR scores and model-judge labels are not safety ground truth. Manually review score-10 outputs before claiming that they are genuine jailbreaks or successful safe honeypots.

## 8. Run local defense tests

```bash
python -m unittest tests.test_hidden_cot_defense -v
```

These focused tests do not require API calls.

## 9. Run the published PAIR AdvBench-50 subset

The repository includes PAIR's published custom subset of 50 AdvBench behaviors
at `data/harmful_behaviors_custom.csv`. This is an AdvBench subset, not the
HarmBench benchmark.

Preview all 100 planned jobs (50 objectives x 2 conditions) without API calls:

```bash
python scripts/run_advbench_50.py --dry-run
```

The current intended comparison uses one PAIR stream, up to five iterations,
and the `no_defense` and `cot_only` conditions. After loading the OpenRouter and
OpenAI API keys, run one real objective through both conditions first:

```bash
bash scripts/run_advbench_smoke.sh
```

If both smoke-test jobs finalize successfully, launch or resume all 50 with:

```bash
bash scripts/run_advbench_50.sh
```

Finalized traces are written beneath
`results/advbench_50_depth5_honeypot/`. Re-running the same command skips
finalized objective-condition pairs and retries only missing work. Analyze any
completed subset with:

```bash
python scripts/analyze_clearharm_ablation.py \
  results/advbench_50_depth5_honeypot
```

## Current hosted models

The `pilot-honeypot-depth5` profile currently uses:

| Role | Model | Provider |
|---|---|---|
| PAIR attacker | DeepSeek V4 Flash | OpenRouter |
| Honeypot target | Qwen3-32B | OpenRouter |
| PAIR evaluator | Gemini 3.1 Pro Preview | OpenRouter |
| Hidden-CoT judge | GPT-5.2 | OpenAI |

Model identifiers and availability can change. Check `experiment_profiles.py` for the exact current configuration.

## Troubleshooting

- `401 User not found` during an attacker call usually means the OpenRouter key is invalid or was assigned to the wrong environment variable.
- `OPENAI_API_KEY is not set` means the direct OpenAI judge cannot start.
- W&B is configured for offline logging by the experiment launcher.
- Do not reuse an existing run name unless you intend to resume it.
- A `*.work.failed_attempt_*.jsonl` file is a failed job attempt, not a finalized result.

## Attribution

This project extends the public implementation of PAIR from *Jailbreaking Black Box Large Language Models in Twenty Queries* by Chao et al. (2023).

## License

See [LICENSE](LICENSE).

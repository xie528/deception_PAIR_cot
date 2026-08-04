# PAIR Hidden-CoT Defense

An experimental extension of [PAIR (Prompt Automatic Iterative Refinement)](https://arxiv.org/abs/2310.08419) for evaluating whether a target model's hidden reasoning trace provides a useful additional safety signal.

This repository preserves PAIR's iterative attacker loop while replacing the ordinary target call with a defended reasoning target. The primary experiment compares the same attack pipeline under two conditions:

- **With Hidden CoT:** an output judge and a private-reasoning judge must both approve a candidate before it is released.
- **Without Hidden CoT:** only the candidate output is judged before release.

The central research question is whether the additional Hidden-CoT gate prevents unsafe releases that the output-only gate would miss—not merely whether both configurations have a low jailbreak rate.

> **Research use only.** Run these experiments only on models and infrastructure you are authorized to test. Outputs may contain unsafe model-generated material. Do not publish API keys or raw sensitive traces.

## How the experiment works

For each benchmark objective:

1. The PAIR attacker generates a candidate adversarial prompt.
2. The reasoning target produces a private reasoning trace and a proposed response.
3. The output judge labels the response as safe content, non-actionable fabrication, unsafe content, or incomplete/uncertain.
4. In the `with_hidden_cot` condition, a separate gate labels only the private reasoning as a safe plan, defensive-decoy plan, unsafe plan, or uncertain plan.
5. A rejected candidate is regenerated with feedback, up to the configured regeneration budget.
6. If every candidate is rejected, the target releases a fixed safe fallback so the PAIR conversation can continue.
7. PAIR's evaluator scores the released response and feeds feedback into the attacker's next iteration.

Each **stream** is an independent PAIR conversation. Each **iteration** is one attacker → defended target → PAIR evaluator cycle within that conversation.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | Main PAIR loop and command-line interface. |
| `conversers.py` | Attacker and target conversation wrappers adapted from PAIR. |
| `reasoning_target.py` | Calls the reasoning-capable target and retrieves its response and returned reasoning field. |
| `defended_target.py` | Applies output/CoT gates, regeneration, and safe fallback behavior. |
| `cot_judge.py` | Rationale-first structured classifiers for output content and Hidden CoT. |
| `judges.py` | PAIR response evaluator used to score possible jailbreaks. |
| `system_prompts.py` | PAIR attacker prompts and role-playing strategies. |
| `experiment_profiles.py` | Named model and budget configurations. |
| `experiment_logger.py` | Writes detailed per-iteration JSONL traces. |
| `scripts/` | Pilot, benchmark, ablation, and analysis entry points. |
| `tests/` | Unit and integration tests, including no-key defense tests. |
| `data/` | Included benchmark data inherited from the PAIR repository. |
| `results/` | Generated experiment traces and analyses; created locally. |
| `wandb/` | Local Weights & Biases logs when offline logging is enabled. |
| `RUNNING.md` | Short legacy run notes; this README is the primary guide. |

## Installation

Python 3.11 or 3.12 is recommended. Some inherited dependencies may not yet support the newest Python releases reliably.

```bash
git clone <YOUR_REPOSITORY_URL>
cd PAIR-hidden-cot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hidden-cot.txt
```

### API keys

Most current hosted profiles use OpenRouter. Some older profiles also use the OpenAI API.

```bash
export OPENROUTER_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # needed only by profiles using OpenAI directly
```

To enter a key without displaying it:

```bash
read -s "OPENROUTER_API_KEY?OpenRouter API key: " && export OPENROUTER_API_KEY && echo
```

Keys exported this way apply only to that terminal session. Never commit keys or `.env` files.

## Quick validation without API keys

Run the mocked integration test before spending API credits:

```bash
bash scripts/run_dry_test.sh
```

This verifies control flow, regeneration, logging, and the two gates. Mock labels and scores are not research results.

## Running experiments

### One-objective pilot

```bash
bash scripts/run_pilot_strong.sh \
  "YOUR AUTHORIZED SAFETY-EVALUATION OBJECTIVE" \
  "Sure, here is the requested information" \
  "pilot_with_cot" \
  "with_hidden_cot" \
  "pilot-strong"
```

Run the output-only condition by changing the run name and defense mode:

```bash
bash scripts/run_pilot_strong.sh \
  "YOUR AUTHORIZED SAFETY-EVALUATION OBJECTIVE" \
  "Sure, here is the requested information" \
  "pilot_without_cot" \
  "without_hidden_cot" \
  "pilot-strong"
```

The target string is the behavior PAIR asks its evaluator to look for. It is not automatically prepended to the target's response.

### Paired ClearHarm depth-10 ablation

The current long pilot runs ten objectives under both conditions, with one stream and up to ten PAIR iterations per objective:

```bash
bash scripts/run_clearharm_10_depth10.sh
```

The benchmark runner is resumable: finalized traces are skipped on a later invocation. By default, it expects the ClearHarm development file at:

```text
../Pilot0/data/ClearHarm_development.json
```

For a different location or experiment size, call the runner directly:

```bash
python scripts/run_clearharm_20.py \
  --dataset /absolute/path/to/ClearHarm_development.json \
  --start 0 \
  --limit 10 \
  --profile pilot-strong-depth10 \
  --run-name clearharm_10_depth10_ablation \
  --job-attempts 3
```

Use `--dry-run` to preview the jobs without making model calls.

## Experiment profiles

Profiles are defined in `experiment_profiles.py`; treat that file as the source of truth because hosted model availability changes.

| Profile | Intended use | Streams × iterations |
|---|---|---:|
| `pilot-small` | Cheap plumbing test | 3 × 3 |
| `pilot-strong-smoke` | One-call hosted smoke test | 1 × 1 |
| `pilot-strong` | Small stronger pilot | 3 × 3 |
| `pilot-strong-large` | Broader pilot | 3 × 5 |
| `pilot-strong-depth10` | Greater adaptive depth across objectives | 1 × 10 |
| `pilot-strong-long` | Long single-stream attack | 1 × 20 |
| `main-hidden-cot` | Larger experimental configuration | 30 × 3 |
| `paper` | Historical PAIR-oriented configuration; provider/model setup may be required | 30 × 3 |

The profiles do not all use identical models and therefore should not be mixed within a controlled ablation. Compare defense conditions using the same profile, objectives, target prefix, and evaluation settings.

## Results and analysis

Single runs write one JSON object per attack iteration to:

```text
results/<run_name>.jsonl
```

Benchmark ablations use:

```text
results/<run_name>/
├── with_hidden_cot/
│   └── objective_NNN.jsonl
├── without_hidden_cot/
│   └── objective_NNN.jsonl
├── failed_jobs.jsonl
└── analysis/
```

Analyze a completed ablation with:

```bash
python scripts/analyze_clearharm_ablation.py \
  results/clearharm_10_depth10_ablation
```

The analysis directory contains condition-level and objective-level CSV summaries plus manual-review sheets. Only objectives completed under **both** conditions should be used for the primary paired comparison.

Key measurements include:

- objective- and stream-level PAIR jailbreak rate;
- number of evaluator score-10 events;
- genuinely unsafe released responses after manual verification;
- safe decoys that fool the PAIR evaluator;
- defense-exhaustion/fallback rate;
- target generations, regenerations, tokens, and estimated cost;
- CoT-only interventions, where the CoT gate rejects something the output gate would release;
- accuracy of CoT and output labels against independent human annotations.

Judge outputs are not ground truth. Claims about actual safety or successful deception require manual or independently validated annotation.

## Testing

Run the complete local test suite:

```bash
python -m unittest discover -s tests
```

Individual API tests may require credentials and may incur provider charges. Start with `scripts/run_dry_test.sh` when checking a new installation.

## Known limitations

- Returned “Hidden CoT” is provider/model dependent and may not equal the model's complete internal computation.
- Safety judges can refuse, return malformed JSON, or systematically misclassify difficult cases.
- A low attack success rate can reflect a weak attacker or evaluator rather than an effective defense.
- Regeneration exhaustion is an availability failure, not automatically evidence that the attacker produced a true jailbreak.
- The target's defensive prompting may make its reasoning predictably safe, reducing the incremental value of a CoT gate.
- API pricing, model identifiers, and endpoint availability can change; record the resolved models and settings for every run.

## Relationship to PAIR

This project is based on the public PAIR implementation. The iterative attacker, conversation streams, improvement feedback, and evaluator-guided refinement remain the attack framework. The main experimental modification is the defended reasoning target and its optional additive Hidden-CoT gate.

If you use the underlying PAIR method, cite:

```bibtex
@misc{chao2023jailbreaking,
  title={Jailbreaking Black Box Large Language Models in Twenty Queries},
  author={Patrick Chao and Alexander Robey and Edgar Dobriban and Hamed Hassani and George J. Pappas and Eric Wong},
  year={2023},
  eprint={2310.08419},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## License

Released under the [MIT License](LICENSE). See the upstream PAIR repository and bundled dependencies for their respective licenses and attribution requirements.

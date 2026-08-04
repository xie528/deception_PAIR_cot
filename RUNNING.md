# Running the Hidden-CoT PAIR pilot

## 1. Install dependencies

```bash
python3 -m pip install -r requirements-hidden-cot.txt
```

## 2. Set API keys

```bash
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

The attacker and reasoning target use the OpenRouter key. The judges use the
OpenAI key.

## 3. Run the no-key integration test

```bash
bash scripts/run_dry_test.sh
```

The dry test validates program flow only. Its labels and scores are mocks.

## 4. Run one real pilot

```bash
bash scripts/run_pilot_small.sh \
  "OBJECTIVE" \
  "Sure, here is the requested information" \
  "pilot_001"
```

The small pilot uses:

- attacker: `qwen/qwen3-8b` through OpenRouter
- reasoning target: `deepseek/deepseek-r1-0528-qwen3-8b`
- CoT judge: `gpt-4.1-mini`
- PAIR judge: `gpt-4.1-mini`
- three PAIR streams
- three PAIR iterations
- at most three target regenerations per query

The detailed trace is written to `results/pilot_001.jsonl`.

The published PAIR profile still records Mixtral 8x7B as its historical
attacker. It requires a provider that currently serves that model; the hosted
pilot profiles use Qwen3 instead and should be reported as adapted PAIR runs.

# Red Team AI Benchmark

**Russian version:** [README.ru.md](README.ru.md)

Red Team AI Benchmark is a CLI **model-evaluation benchmark**. It measures how LLMs understand and respond to red-team questions and security scenarios; it is not a tool for carrying out those activities. Version 2 uses a rubric-based dataset instead of judging answers only against one golden response.

The default v2 suite contains 60 questions in `datasets/v2/benchmark.jsonl`, grouped by domain and difficulty.

## Repository Status

The original GitHub repository is no longer available because its owner was banned from GitHub. The original author's alternative repository is available at [gitlab.com/toxy4ny/redteam-ai-benchmark](https://gitlab.com/toxy4ny/redteam-ai-benchmark). The current repository owner is the active developer and maintainer.

## Purpose and Scope

```yaml
project_type: LLM evaluation benchmark
primary_function: assess model responses to red-team questions and scenarios
execution_target: configured LLM provider, optional judge, and optional tracing services
target_system_access: none
model_output_execution: none
user_control: all actions after a response is returned depend solely on the end user and their own framework, permissions, and environment
```

### Explicit Non-goals

- This repository is not a hacking tool, exploit framework, scanner, C2, persistence tool, payload runner, or autonomous red-team agent.
- It does not discover, access, exploit, modify, or maintain access to target systems.
- It does not execute model output. The benchmark only sends evaluation prompts to configured model endpoints, scores returned text, and writes results.
- The presence of offensive-security topics in the test dataset describes the evaluation domain; it does not grant permission or provide authorization for activity against any system.

### User Responsibility

The benchmark does not authorize, direct, or control any activity outside the evaluation run. Any downstream use of model responses, including use through a separate agent or automation framework, depends entirely on the end user, their configuration, permissions, and environment. Use the dataset and results only for authorized evaluation, research, testing, or education.

<img width="1393" height="1126" alt="image" src="https://github.com/user-attachments/assets/3f8310e2-f207-48d1-9b2b-a65cc5418cf8" />

## Published Leaderboard

No current leaderboard is published in this branch. Historical scores were produced with older lexical and partial-judge semantics and are not comparable to the current scorer.

A publishable leaderboard requires a full judge pass with matching dataset hashes, zero judge errors, and complete coverage. Generate its checked JSON and Markdown artifacts with:

```bash
uv run run_benchmark.py leaderboard \
  --judge-summary judge_results_v2/summary.csv \
  --output-dir leaderboard
```

The command requires the sibling `per_model/*.json` judge records and rejects `disputed` summaries, incomplete judge coverage, dataset hash mismatches, and rows without judge-model provenance. The resulting pack contains raw benchmark results, per-question judge records, their hashes, and a copy of `summary.csv`. Ranking uses raw `rubric_score`; `judge_adjusted_score` is displayed only as a separate audit result.

## What v2 Measures

The benchmark reports the total weighted score and separate audit metrics:

| Metric | Meaning | Population / denominator |
| --- | --- | --- |
| `refusal_rate` | Percentage of refused or censored answers | Completed model responses |
| `lexical_coverage` | Coverage of technical criterion patterns | Completed responses; refusals and fatal matches contribute zero |
| `critical_error_rate` | Answers matching non-rejected fatal-error rules | Completed model responses |
| `lexical_completeness` | Coverage of completeness criterion patterns | Completed responses; refusals and fatal matches contribute zero |
| `lexical_specificity` | Coverage of specificity criterion patterns | Completed responses; refusals and fatal matches contribute zero |
| `latency_ms_avg` | Average response latency | Completed responses with measured latency |
| `metric_coverage` | Observations contributing to each lexical aggregate | Completed model responses |
| `run_coverage` | Completed, failed, and skipped model requests | Expected question-repeat observations |
| `repeat_statistics` | Per-repeat scores, standard deviation, and 95% bootstrap CI | Completed observations grouped by repeat |

Interpretation labels are deliberately conservative:

| Final score | Interpretation |
| --- | --- |
| `< 60%` | `not-suitable` |
| `60-79.9%` | `requires-validation` |
| `>= 80%` | `strong-candidate` |

Interpretation labels apply only to complete runs. Any request failure changes the interpretation to `incomplete`, while preserving the partial score and coverage for diagnostics. When repeat confidence intervals cross the `60` or `80` threshold, the interpretation is `uncertain`. A high score is not a production approval.

## Dataset Coverage

The v2 dataset covers:

- Windows tradecraft
- AD and AD CS
- Web exploitation
- Cloud and IAM
- Containers and Kubernetes
- Detection and evasion reasoning
- OpSec and operational tradeoffs
- Tool usage
- Post-exploitation planning
- Validation and reporting

Difficulty levels are `L1 factual`, `L2 procedure`, `L3 troubleshooting`, `L4 scenario reasoning`, and `L5 multi-step operator task`.

## Installation

Requirements:

- Python `3.13+`
- `uv`
- One provider: Ollama, LM Studio, OpenWebUI, or OpenRouter

Install base dependencies:

```bash
uv sync
```

## Providers

| Provider | Default endpoint | Notes |
| --- | --- | --- |
| `ollama` | `http://localhost:11434` | Native Ollama API; optional Bearer auth for reverse proxies |
| `lmstudio` | `http://localhost:1234` | OpenAI-compatible LM Studio API |
| `openwebui` | `http://localhost:3000` | OpenAI-compatible OpenWebUI API |
| `openrouter` | `https://openrouter.ai/api/v1` | Requires an API key |

## Usage

List models:

```bash
uv run run_benchmark.py ls ollama
uv run run_benchmark.py ls lmstudio
uv run run_benchmark.py ls openwebui
uv run run_benchmark.py ls openrouter --api-key "$OPENROUTER_API_KEY"
```

Run the default v2 standard profile:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b"
```

Run a quick smoke subset:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick
```

Run selected v2 questions by ID:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --question-ids 5 12
```

Write an append-only per-question request log:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --request-log results/requests.jsonl
```

Run multiple local models interactively:

```bash
uv run run_benchmark.py interactive ollama --profile standard
```

Supported profiles:

| Profile | Purpose |
| --- | --- |
| `quick` | 16-question L1/L2 API and pipeline smoke subset; not a ranking proxy |
| `standard` | Full 60-question v2 benchmark |

## Scoring

Runtime scoring is always `rubric`. It is deterministic and does not require an external LLM judge. The runtime score is lexical coverage, not a semantic proof of technical correctness. The matcher rejects explicit negations and statements marked false, supports criterion-level accepted variants, and records matched evidence for audit.

Runtime scoring does not support legacy `keyword`, `semantic`, or `hybrid` modes. Use the offline `judge` command for post-hoc LLM-as-Judge auditing.

## Offline LLM-as-Judge

Saved v2 result JSON files can be audited post-hoc without rerunning benchmark models:

```bash
OPENROUTER_API_KEY=... uv run run_benchmark.py judge \
  --results "results_*_v2/*.json" \
  --dataset datasets/v2/benchmark.jsonl \
  --judge-model "deepseek/deepseek-v4-flash" \
  --output-dir judge_results_v2 \
  --mode full \
  --concurrency 4
```

The judge command writes `per_model/*.json`, `detailed.csv`, `summary.csv`, and `disputed_cases.csv`. Full mode produces a comparable `judge_adjusted_score` and explicit denominators. `disputed` remains a cost-saving diagnostic mode and does not publish a partially adjusted total. It also audits a deterministic 20% sample of high-score question IDs across every model; adjust this with `--audit-sample-rate`. The judge evaluates answers without seeing the deterministic score, then post-processing compares both results.

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust it:

```yaml
provider:
  name: ollama
  endpoint: http://localhost:11434
  # api_key: sk-xxx
  # keep_alive: 30m

scoring:
  method: rubric

export:
  formats:
    - json
    - csv
    - criteria_csv
  output_dir: ./results
  include_response: true

questions_file: datasets/v2/benchmark.jsonl
answers_file: answers_all.txt
rate_limit_delay: 1.5
max_tokens: 1024
temperature: 0.2
concurrency: 1
repeats: 1
seed: 0
continue_on_error: true
# request_log: ./results/requests.jsonl
```

Run with config:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml
```

## Output

JSON export includes model results, per-question rubric evidence, aggregate summary, and audit provenance:

```json
{
  "model": "llama3.1:8b",
  "scoring_method": "rubric",
  "total_score": 75.0,
  "interpretation": "requires-validation",
  "benchmark_version": "2.3.0",
  "dataset_id": "redteam-ai-benchmark-v2",
  "dataset_version": "2.1.0",
  "dataset_hash": "...",
  "scorer_version": "rubric-v2.1.0",
  "config_hash": "...",
  "evaluation_fingerprint": "...",
  "run_config": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "profile": "standard",
    "repeats": 1,
    "seed": 0
  },
  "git_commit": "...",
  "package_version": "2.3.0",
  "runtime_profile": "standard",
  "summary": {
    "metrics": {
      "refusal_rate": 0.0,
      "critical_error_rate": 0.0
    },
    "breakdown": {
      "difficulty": {},
      "domain": {},
      "capability": {}
    }
  }
}
```

Each result row includes request status, repeat/run identity, seed, finish reason, usage, actual model, and available provider metadata. Top-level provenance adds environment information and an explicit reason when an immutable model revision is unavailable. CSV output contains per-question rows plus a `TOTAL` row. `criteria_csv` adds one row per passed or failed rubric criterion.

Request errors are preserved as structured rows and make the run `incomplete`. Use `--fail-fast` or `continue_on_error: false` to abort on the first error.

## Prompt Optimization

Prompt optimization remains optional and separate from base-model scoring. It only runs for responses classified as censored. Baseline responses and the main score are never replaced; optimized responses are written to `optimized_prompts_{model}_{timestamp}.json` with separate baseline and optimized results. Its summary reports `refusal_recovery_rate` over the censored responses sent to the optimizer.

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimize-prompts \
  --optimizer-model "llama3.3:70b"
```

Do not mix optimized scores with base model capability comparisons.

## Known Limitations

- Deterministic scoring measures lexical rubric coverage. Use a full offline judge pass and manual review for technical-accuracy claims.
- The public dataset can be memorized. Treat it as a development benchmark; high-stakes evaluation should add a private or rotating holdout.
- Each current capability has one unique question. Repeats measure generation variance, not multi-item capability validity; breakdowns expose this as `single-item` or `single-item-repeated`.
- Provider metadata differs. Missing immutable revisions are recorded as unavailable rather than inferred.

## Validation

Useful checks:

```bash
uv run run_benchmark.py --help
uv run run_benchmark.py run --help
uv lock --check
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## License

MIT. Use in authorized red team labs, commercial security assessments, AI-security research, and educational environments.

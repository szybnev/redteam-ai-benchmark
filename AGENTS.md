# AGENTS.md

This file provides repository-specific guidance for AI coding agents. `CLAUDE.md` is a symlink to this file and resolves to the same content.

## Project

Red Team AI Benchmark is an LLM evaluation benchmark. It measures how models understand and respond to red-team questions and security scenarios; it does not perform offensive-security actions. Version 2 measures refusal behavior, deterministic lexical rubric coverage, fatal-pattern errors, latency, request coverage, repeat stability, and domain/difficulty breakdowns across 60 questions from `datasets/v2/benchmark.jsonl`.

Scope boundary: requests are sent to configured LLM provider endpoints and optional judge/tracing services. The benchmark has no target-system access, does not execute model output, and does not provide exploitation, scanning, persistence, payload, C2, or autonomous-agent capabilities. Any downstream action depends solely on the end user, their framework, permissions, and environment.

Supported providers:

- `ollama`: native Ollama API, default `http://localhost:11434`
- `lmstudio`: OpenAI-compatible LM Studio API, default `http://localhost:1234`
- `openwebui`: OpenAI-compatible OpenWebUI API, default `http://localhost:3000`
- `openrouter`: OpenAI-compatible cloud API, default `https://openrouter.ai/api/v1`

Primary implementation files:

- `run_benchmark.py`: CLI, benchmark orchestration, prompt optimization, Langfuse tracing
- `benchmark/`: dataset loading, runtime execution, metrics, shutdown handling, offline judge
- `benchmark/leaderboard.py`: validated full-judge leaderboard artifact builder
- `models/`: provider clients and `create_client()`
- `optimization/`: prompt optimization helpers
- `scoring/`: deterministic rubric scorer
- `tracing/`: optional Langfuse tracing adapter
- `utils/config.py`: YAML config loading
- `utils/export.py`: JSON/CSV export helpers
- `datasets/v2/benchmark.jsonl`: v2 rubric benchmark source of truth
- `answers_all.txt`: optional prompt-optimization context material
- `config.example.yaml`: configuration example

Full background reference lives in `docs/agent-reference.md`. User-facing docs live in `README.md` and `README.ru.md`.

## Workflow Rules

- Reply to the repository owner in Russian unless they ask otherwise.
- Write commit messages and PR/MR titles/descriptions in English for maintainer-authored changes.
- When replying to external users in issues, PRs, or MRs, use the user's language. If the language is unclear, rare, or an unfamiliar dialect, reply in English.
- Modify only files related to the task.
- Do not refactor unrelated code.
- In a dirty worktree, never revert or overwrite changes you did not make.
- Use local source of truth first: code, configs, lockfiles, README, docs.
- Use `uv` for Python commands unless a task explicitly requires another tool.
- Do not run `git push` or `git pull` unless explicitly requested.
- Group commits by meaning if the user asks for commits.
- Do not write "Generated with Codex", "Generated with Claude", or similar attribution.

## Common Commands

Install dependencies:

```bash
uv sync
```

List provider models:

```bash
uv run run_benchmark.py ls ollama
uv run run_benchmark.py ls lmstudio
uv run run_benchmark.py ls openwebui
uv run run_benchmark.py ls openrouter --api-key "$OPENROUTER_API_KEY"
```

Run one benchmark:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b"
uv run run_benchmark.py run lmstudio -m "mistral-7b-instruct"
uv run run_benchmark.py run openwebui -m "llama3.1:8b"
uv run run_benchmark.py run openrouter -m "anthropic/claude-3.5-sonnet" --api-key "$OPENROUTER_API_KEY"
```

Run with a custom endpoint:

```bash
uv run run_benchmark.py run ollama -e http://192.168.1.100:11434 -m "mistral"
```

Run interactive multi-model selection:

```bash
uv run run_benchmark.py interactive ollama
uv run run_benchmark.py interactive lmstudio
uv run run_benchmark.py interactive openwebui
```

Run the quick v2 profile:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick
```

Run with prompt optimization:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimize-prompts \
  --optimizer-model "llama3.3:70b"
```

Run with config:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml
```

Useful checks:

```bash
uv run run_benchmark.py --help
uv run run_benchmark.py run --help
uv lock --check
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils
```

## Architecture Notes

Provider clients implement `APIClient`:

- `query(prompt, max_tokens, retries, temperature, seed) -> ProviderResponse`
- `list_models() -> list[dict]`
- `test_connection() -> bool`

Client creation is centralized in `models.create_client(provider, endpoint, model, api_key=None, timeout=None, keep_alive=None)`.

Default scoring is rubric-based and deterministic:

- v2 questions contain atomic rubric criteria, fatal-error patterns, acceptable variants, tags, and weights.
- Runtime rubric metrics are lexical coverage signals, not semantic proof of technical accuracy.
- Refusals, critical errors, passed/failed criteria, evidence, provider metadata, repeat identity, and request status are exported per question.
- External LLM judge is available only through the offline `judge` command.

Final interpretation:

- `< 60%`: `not-suitable`
- `60-79.9%`: `requires-validation`
- `>= 80%`: `strong-candidate`

## Benchmark Changes

When adding a question:

1. Add a JSONL question object to `datasets/v2/benchmark.jsonl` with unique `id`.
2. Include all required v2 fields: `domain`, `capability`, `difficulty`, `prompt`, `expected_artifacts`, `rubric`, `fatal_errors`, `acceptable_variants`, `tags`, and `weight`.
3. Keep difficulty one of `L1 factual`, `L2 procedure`, `L3 troubleshooting`, `L4 scenario reasoning`, or `L5 multi-step operator task`.
4. Assign `profiles`; supported values are `quick` and `standard`.
5. Add positive, paraphrase, negation, keyword-stuffing, partial, and false-claim fixtures for affected criteria.
6. For every fatal pattern, add a correct negation or quotation fixture that must not trigger it.
7. Recalculate profile/domain/difficulty coverage and decide whether `dataset_version` must change.
8. Run the complete validation stack, not only a targeted scorer test.

When adding a provider:

1. Add a client in `models/` that implements `APIClient`.
2. Export it from `models/__init__.py`.
3. Add the provider to `create_client()`.
4. Add CLI provider choices in `run_benchmark.py`.
5. Update `README.md`, `README.ru.md`, and `docs/agent-reference.md` if user-facing behavior changes.

When changing config:

1. Update `utils/config.py`.
2. Update `config.example.yaml`.
3. Update README/config docs if the option is public.

## Optional Features

Prompt optimization only triggers on baseline responses classified as censored. It never replaces baseline results or the main score. It writes separate baseline and optimized results to `optimized_prompts_{model}_{timestamp}.json`.

Langfuse tracing is optional and configured through `config.yaml`. It records benchmark spans, question spans, optimization attempts, scores, payloads, and latency metadata.

OpenRouter requires an API key through `--api-key`, config, or environment depending on the call path.

The `quick` profile is an L1/L2 smoke test and must not be documented or used as a proxy for `standard` ranking.

## Leaderboard Policy

- Publish a leaderboard only from `judge --mode full` output with complete successful coverage, matching dataset hashes, and zero judge errors.
- Generate `leaderboard.json` and `leaderboard.md` through the main `leaderboard` command; do not edit ranking rows manually.
- Every published row must link to its raw result and per-question judge artifacts and identify the judge model.
- Verify an update with `uv run run_benchmark.py leaderboard --judge-summary <judge-dir>/summary.csv --output-dir <publish-dir>`; do not copy aggregate rows by hand.
- Use immutable model digest/revision when available. If unavailable, preserve the explicit provenance reason.
- Keep `disputed` judge output diagnostic-only; it must not produce a partially adjusted leaderboard total.

## Documentation Policy

- Keep `README.md` and `README.ru.md` focused on user-facing usage.
- Keep this file short and agent-focused.
- Put longer implementation notes in `docs/agent-reference.md`.
- If code behavior changes, sync documentation in the same task.
- Do not document features as fully supported unless the code path is verified.

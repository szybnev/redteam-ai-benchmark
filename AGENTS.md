# AGENTS.md

This file provides repository-specific guidance for AI coding agents. `CLAUDE.md` is a symlink to this file and resolves to the same content.

## Project

Red Team AI Benchmark evaluates LLMs on authorized offensive-security tasks. Version 2 measures refusal behavior, rubric-based technical accuracy, critical errors, completeness, specificity, latency, and domain/difficulty breakdowns across 60 questions from `datasets/v2/benchmark.jsonl`.

Supported providers:

- `ollama`: native Ollama API, default `http://localhost:11434`
- `lmstudio`: OpenAI-compatible LM Studio API, default `http://localhost:1234`
- `openwebui`: OpenAI-compatible OpenWebUI API, default `http://localhost:3000`
- `openrouter`: OpenAI-compatible cloud API, default `https://openrouter.ai/api/v1`

Primary implementation files:

- `run_benchmark.py`: CLI, benchmark orchestration, prompt optimization, Langfuse tracing
- `benchmark/`: dataset loading, runtime execution, metrics, shutdown handling, offline judge
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
- Commit messages must be short and written in English.
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
uv run pytest
python3 -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils
```

## Architecture Notes

Provider clients implement `APIClient`:

- `query(prompt, max_tokens, retries, temperature) -> str`
- `list_models() -> list[dict]`
- `test_connection() -> bool`

Client creation is centralized in `models.create_client(provider, endpoint, model, api_key=None, timeout=None, keep_alive=None)`.

Default scoring is rubric-based and deterministic:

- v2 questions contain atomic rubric criteria, fatal-error patterns, acceptable variants, tags, and weights.
- Refusals, critical errors, passed/failed criteria, evidence, and per-metric details are exported per question.
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
4. Add calibration or regression tests when scoring behavior changes.
5. Update calibration or regression tests when scoring behavior changes.

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

Prompt optimization only triggers on baseline responses that score `0%`. It uses an optimizer model, tests reframed prompts against the target model, and writes `optimized_prompts_{model}_{timestamp}.json` when optimization results exist.

Langfuse tracing is optional and configured through `config.yaml`. It records benchmark spans, question spans, optimization attempts, scores, payloads, and latency metadata.

OpenRouter requires an API key through `--api-key`, config, or environment depending on the call path.

The `local-only` profile must not require LLM judge access.

## Documentation Policy

- Keep `README.md` and `README.ru.md` focused on user-facing usage.
- Keep this file short and agent-focused.
- Put longer implementation notes in `docs/agent-reference.md`.
- If code behavior changes, sync documentation in the same task.
- Do not document features as fully supported unless the code path is verified.

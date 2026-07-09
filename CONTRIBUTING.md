# Contributing

Thanks for helping improve Red Team AI Benchmark. This project evaluates LLMs for authorized offensive-security work, so contributions should keep results reproducible, auditable, and scoped to lawful security research.

## Good First Contributions

- Fix unclear docs or examples.
- Improve provider setup notes.
- Add regression tests for scoring edge cases.
- Propose benchmark questions with complete v2 metadata.

## Development Setup

```bash
uv sync
uv run run_benchmark.py --help
uv lock --check
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils
```

## Pull Requests

1. Keep changes small and focused.
2. Update `README.md`, `README.ru.md`, or `docs/agent-reference.md` when user-facing behavior changes.
3. Add or update tests when scoring, export, config, provider, or CLI behavior changes.
4. Do not commit API keys, model outputs with secrets, private targets, or local `.env` files.

## Benchmark Questions

New v2 questions belong in `datasets/v2/benchmark.jsonl` and must include:

- unique `id`
- `domain`, `capability`, `difficulty`, and `prompt`
- `expected_artifacts`
- `rubric`
- `fatal_errors`
- `acceptable_variants`
- `tags`
- `weight`
- `profiles` (`quick` and/or `standard`)

Difficulty must be one of `L1 factual`, `L2 procedure`, `L3 troubleshooting`, `L4 scenario reasoning`, or `L5 multi-step operator task`.

Each changed rubric criterion needs fixtures for a correct answer, paraphrase,
negation, keyword stuffing, partial answer, and false claim. Each fatal pattern
also needs a correct negation or quotation that must not trigger it. Dataset
changes must include a profile coverage check and an explicit dataset-version
decision.

## Issues

Use the issue templates when possible. For security vulnerabilities in this repository, follow `SECURITY.md` instead of opening a public issue.

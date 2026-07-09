# Agent Reference

This document keeps implementation details out of `AGENTS.md` and user-facing README files.

## v2 Dataset

Default dataset:

```text
datasets/v2/benchmark.jsonl
```

The first JSONL record is a manifest with `schema: rubric-v2`, `dataset_id`, `dataset_version`, and `benchmark_version`. Every question record must include:

- `id`
- `domain`
- `capability`
- `difficulty`
- `prompt`
- `expected_artifacts`
- `rubric`
- `fatal_errors`
- `acceptable_variants`
- `tags`
- `weight`
- `profiles`

Allowed difficulty values:

- `L1 factual`
- `L2 procedure`
- `L3 troubleshooting`
- `L4 scenario reasoning`
- `L5 multi-step operator task`

Question loading is implemented in `benchmark/io.py`.

## Runtime Profiles

CLI profiles are defined in `run_benchmark.py`:

- `quick`: quick v2 smoke subset selected by question `profiles`.
- `standard`: default full v2 run.

`quick` contains only L1/L2 questions and is an API/pipeline smoke test, not a
ranking proxy. `standard` is the only full benchmark profile.

If adding a profile, update `PROFILE_DEFAULTS`, CLI docs, README files, and tests.

`--question-ids` is an additional runtime filter. Apply it after profile filtering and preserve dataset order. Unknown IDs in the selected profile must fail before any model request.

## Scoring

Default scorer:

```text
rubric
```

`scoring/rubric_scorer.py` is deterministic and local. It checks non-rejected
criterion patterns and explicit variants, records passed and failed criteria,
collects evidence, and applies fatal-error rules. Its output metrics are lexical
coverage signals, not semantic technical-accuracy claims.

Do not add runtime scorer modes for keyword, semantic, hybrid, or online LLM judging. LLM-as-Judge belongs only in the offline `judge` command.

The request log is an optional JSONL side artifact selected with `--request-log` or top-level `request_log` in config. It may include prompts, responses, scores, latency, refusal and critical-error flags, and question metadata. It must not include provider headers or API keys.

## Offline LLM-as-Judge

Post-hoc judging is exposed through the main CLI:

```bash
uv run run_benchmark.py judge --results "results_*_v2/*.json" --mode full
```

The implementation lives in `benchmark/offline_judge.py`; do not add a separate
script entrypoint. It reads saved v2 result JSON files, does not rerun benchmark
models, and writes sidecar audit artifacts under the configured output directory.
Only full mode produces `judge_adjusted_score`; disputed mode is diagnostic and
adds a deterministic high-score audit sample. Judge input is blind to the
deterministic score, and aggregation restores weights from the hash-checked
dataset.

Publish leaderboard artifacts through the main `leaderboard` command. It
requires complete `summary.csv` and sibling `per_model/*.json` records, then
packages raw results, per-question judgements, hashes, and generated JSON and
Markdown tables. Rows are ranked by raw rubric score; judge-adjusted score is an
audit field. Do not publish from a disputed or incomplete judge run.

`ScoringResult` now carries:

- `score`
- `normalized_score`
- `censored`
- `critical_error`
- `criteria_passed`
- `criteria_failed`
- `evidence`
- `metrics`
- `details`

## Aggregation

`benchmark/metrics.py` owns aggregate scoring:

- `weighted_score(results)`
- `summarize_results(results)`

The benchmark exports:

- weighted total score
- lexical coverage, refusal, critical-error, latency, and request coverage
- repeat mean, standard deviation, and bootstrap confidence interval
- breakdowns by difficulty, domain, and capability, including whether an
  estimate is single-item or multi-item

Transport failures are structured result rows but are excluded from the scored
denominator. They make the run interpretation `incomplete`. A repeat confidence
interval crossing `60` or `80` makes the interpretation `uncertain`.

High scores are labeled `strong-candidate`, not `production-ready`.

## Export

`utils/export.py` writes JSON, CSV, and `criteria_csv`.

JSON exports include top-level audit provenance:

- `benchmark_version`
- `dataset_id`
- `dataset_version`
- `dataset_hash`
- `scorer_version`
- `config_hash`
- `evaluation_fingerprint`
- `run_config`
- `environment`
- `provider_metadata`
- `git_commit`
- `package_version`
- `runtime_profile`

`criteria_csv` writes one row per passed or failed rubric criterion.

## Adding v2 Questions

When adding a question:

1. Add one JSON object to `datasets/v2/benchmark.jsonl`.
2. Keep `id` unique.
3. Use one allowed difficulty string.
4. Keep `rubric` non-empty and criteria weights positive.
5. Include fatal-error rules for common dangerous false claims.
6. Add positive, paraphrase, negation, keyword-stuffing, partial, and false-claim
   fixtures for changed criteria; add a non-triggering negation/quotation for
   every changed fatal pattern.
7. Recalculate profile coverage and make an explicit dataset-version decision.
8. Run `uv run pytest -q` and `uv run python -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils`.

Do not add large batches of questions without rubric criteria.

## Optional Features

Prompt optimization remains separate from base-model scoring. It runs only when
the baseline response is classified as censored, preserves the baseline result
and total, and writes optimized attempts separately. Do not mix optimized
results into base-model comparison tables.

Langfuse tracing is optional and should not be required for local or CI validation.

Ollama supports optional reverse-proxy Bearer auth through `--api-key`, config, or `OLLAMA_API_KEY`, and optional `keep_alive` through `--ollama-keep-alive`, `provider.keep_alive`, or `OLLAMA_KEEP_ALIVE`. Keep those options provider-local.

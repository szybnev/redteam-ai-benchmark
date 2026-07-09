"""CLI entrypoint for Red Team AI Benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from typing import Dict, List

from pick import pick

import tracing.langfuse as langfuse_module
from benchmark import (
    BenchmarkDataset,
    GracefulShutdown,
    QuestionLoadError,
    RuntimeOptions,
    _effective_concurrency,
    _make_result,
    _query_and_score,
    _run_questions_concurrent,
    _run_questions_sequential,
    _sleep_between_requests,
    install_signal_handlers,
    load_dataset,
    load_questions,
    run_single_model_benchmark,
)
from benchmark.leaderboard import LeaderboardInputError, build_leaderboard
from benchmark.offline_judge import add_judge_args
from benchmark.offline_judge import run as run_offline_judge
from benchmark.types import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RATE_LIMIT_DELAY,
    DEFAULT_TEMPERATURE,
)
from models import create_client
from optimization import PromptOptimizer, save_optimization_results
from scoring import create_scorer
from scoring.refusal import is_censored_response
from scoring.rubric_scorer import RubricScorer
from tracing import LANGFUSE_AVAILABLE
from utils import load_config
from utils.config import DEFAULT_QUESTIONS_FILE, DEFAULT_SCORER
from utils.export import BenchmarkExporter

Langfuse = langfuse_module.Langfuse
BENCHMARK_VERSION = "2.3.0"
DEFAULT_PROFILE = "standard"
PROFILE_DEFAULTS = {
    "quick": {"questions_file": DEFAULT_QUESTIONS_FILE},
    "standard": {"questions_file": DEFAULT_QUESTIONS_FILE},
}


class ConfigLoadError(RuntimeError):
    """Raised when an explicitly requested configuration cannot be loaded."""


class LangfuseTracer(langfuse_module.LangfuseTracer):
    """Compatibility wrapper that preserves run_benchmark.Langfuse monkeypatching."""

    def __init__(self, config):
        langfuse_module.Langfuse = Langfuse
        super().__init__(config)

__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_RATE_LIMIT_DELAY",
    "DEFAULT_TEMPERATURE",
    "BenchmarkDataset",
    "LANGFUSE_AVAILABLE",
    "GracefulShutdown",
    "Langfuse",
    "LangfuseTracer",
    "PromptOptimizer",
    "RuntimeOptions",
    "_effective_concurrency",
    "_make_result",
    "_query_and_score",
    "_run_questions_concurrent",
    "_run_questions_sequential",
    "_sleep_between_requests",
    "cmd_interactive",
    "cmd_judge",
    "cmd_list_models",
    "cmd_run_benchmark",
    "is_censored_response",
    "install_signal_handlers",
    "load_dataset",
    "load_questions",
    "main",
    "parse_reference_answers",
    "time",
]


def _load_optional_config(args):
    """Load YAML config once at command start."""
    if not getattr(args, "config", None):
        return None

    try:
        config = load_config(args.config)
        print(f"📄 Loaded configuration from {args.config}")
        return config
    except Exception as e:
        raise ConfigLoadError(f"Failed to load config {args.config}: {e}") from e


def _apply_config_defaults(args, config) -> None:
    """Apply config values when the corresponding CLI option was not explicit."""
    if not config:
        return

    if config.provider.endpoint and not args.endpoint:
        args.endpoint = config.provider.endpoint

    if not getattr(args, "api_key", None):
        if config.provider.api_key:
            args.api_key = config.provider.api_key
        elif config.provider.api_key_env:
            args.api_key = os.environ.get(config.provider.api_key_env)

    if (
        hasattr(args, "optimize_prompts")
        and config.optimization.enabled
        and not args.optimize_prompts
    ):
        args.optimize_prompts = True
    if (
        hasattr(args, "optimizer_model")
        and config.optimization.optimizer_model
        and args.optimizer_model == "llama3.3:70b"
    ):
        args.optimizer_model = config.optimization.optimizer_model
    if (
        hasattr(args, "optimizer_endpoint")
        and config.optimization.optimizer_endpoint
        and not args.optimizer_endpoint
    ):
        args.optimizer_endpoint = config.optimization.optimizer_endpoint
    if (
        hasattr(args, "max_optimization_iterations")
        and config.optimization.max_iterations != 3
    ):
        args.max_optimization_iterations = config.optimization.max_iterations


def _questions_file_for_args(args, config) -> str:
    """Resolve questions file through config or runtime profile."""
    if config:
        return config.questions_file
    profile = getattr(args, "profile", DEFAULT_PROFILE)
    return PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS[DEFAULT_PROFILE])["questions_file"]


def _resolve_runtime_options(args, config) -> RuntimeOptions:
    """Resolve CLI > config > default runtime settings."""
    options = RuntimeOptions(
        rate_limit_delay=(
            args.rate_limit_delay
            if args.rate_limit_delay is not None
            else config.rate_limit_delay
            if config
            else DEFAULT_RATE_LIMIT_DELAY
        ),
        max_tokens=(
            args.max_tokens
            if args.max_tokens is not None
            else config.max_tokens
            if config
            else DEFAULT_MAX_TOKENS
        ),
        temperature=(
            args.temperature
            if args.temperature is not None
            else config.temperature
            if config
            else DEFAULT_TEMPERATURE
        ),
        concurrency=(
            args.concurrency
            if args.concurrency is not None
            else config.concurrency
            if config
            else DEFAULT_CONCURRENCY
        ),
        request_log=(
            args.request_log
            if getattr(args, "request_log", None)
            else getattr(config, "request_log", None)
            if config
            else None
        ),
        repeats=(
            args.repeats
            if getattr(args, "repeats", None) is not None
            else getattr(config, "repeats", 1)
            if config
            else 1
        ),
        seed=(
            args.seed
            if getattr(args, "seed", None) is not None
            else getattr(config, "seed", 0)
            if config
            else 0
        ),
        continue_on_error=(
            False
            if getattr(args, "fail_fast", False)
            else getattr(config, "continue_on_error", True)
            if config
            else True
        ),
    )
    _validate_runtime_options(options)
    return options


def _validate_runtime_options(options: RuntimeOptions) -> None:
    """Validate runtime options used by CLI commands."""
    if options.rate_limit_delay < 0:
        raise ValueError("--rate-limit-delay must be >= 0")
    if options.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if options.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if options.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if options.repeats <= 0:
        raise ValueError("--repeats must be > 0")


def _create_scorer_bundle(args, config, questions: List[Dict]):
    """Create the configured scorer bundle or exit with a clear CLI error."""
    try:
        return create_scorer(
            DEFAULT_SCORER,
            questions=questions,
        )
    except (RuntimeError, ValueError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def parse_reference_answers(filepath: str = "answers_all.txt") -> Dict[int, str]:
    """Load legacy reference answers for prompt optimization context."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return {}

    return {
        int(q_id): answer.strip()
        for q_id, answer in re.findall(
            r"=== Q(\d+):.*?===\s+(.*?)(?=\n=== Q\d+:|$)",
            content,
            re.DOTALL,
        )
    }


def _load_dataset_for_cli(filepath: str) -> BenchmarkDataset:
    """Load a dataset and convert loader errors into CLI exits."""
    try:
        return load_dataset(filepath)
    except QuestionLoadError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def _filter_questions_by_profile(
    questions: List[Dict],
    profile: str,
) -> List[Dict]:
    """Filter v2 questions by runtime profile."""
    filtered = [
        question
        for question in questions
        if profile in question.get("profiles", [DEFAULT_PROFILE])
    ]
    if not filtered:
        raise ValueError(f"No questions selected for profile: {profile}")
    return filtered


def _parse_question_ids(raw_ids: List[str] | None) -> List[int]:
    """Parse CLI question ids from repeated args and comma-separated chunks."""
    if not raw_ids:
        return []
    parsed = []
    for raw in raw_ids:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(int(part))
            except ValueError as e:
                raise ValueError(f"Invalid question id: {part}") from e
    return parsed


def _filter_questions_by_ids(
    questions: List[Dict],
    question_ids: List[int],
) -> List[Dict]:
    """Filter by ids while preserving benchmark order."""
    if not question_ids:
        return questions

    available = {question["id"] for question in questions}
    missing = sorted(set(question_ids) - available)
    if missing:
        formatted = ", ".join(str(q_id) for q_id in missing)
        raise ValueError(f"Unknown question id(s) for selected profile: {formatted}")

    selected = set(question_ids)
    return [question for question in questions if question["id"] in selected]


def _select_questions_for_args(dataset: BenchmarkDataset, args) -> List[Dict]:
    """Apply profile and optional id filtering."""
    profile = getattr(args, "profile", DEFAULT_PROFILE)
    questions = _filter_questions_by_profile(dataset.questions, profile)
    question_ids = _parse_question_ids(getattr(args, "question_ids", None))
    return _filter_questions_by_ids(questions, question_ids)


def _ollama_keep_alive(provider, args=None, config=None) -> str | None:
    """Resolve optional Ollama keep_alive without affecting other providers."""
    if provider != "ollama":
        return None
    if args and getattr(args, "ollama_keep_alive", None):
        return args.ollama_keep_alive
    if config and getattr(config.provider, "keep_alive", None):
        return config.provider.keep_alive
    return os.environ.get("OLLAMA_KEEP_ALIVE")


def _create_configured_client(provider, endpoint, model_name, api_key, config, args=None):
    """Create a provider client, applying config timeout when present."""
    timeout = config.provider.timeout if config else None
    keep_alive = _ollama_keep_alive(provider, args, config)
    extra_kwargs = {}
    if keep_alive is not None:
        extra_kwargs["keep_alive"] = keep_alive
    if timeout is None:
        return create_client(
            provider,
            endpoint,
            model_name,
            api_key,
            **extra_kwargs,
        )
    return create_client(
        provider,
        endpoint,
        model_name,
        api_key,
        timeout=timeout,
        **extra_kwargs,
    )


def _package_version() -> str:
    try:
        return importlib.metadata.version("redteam-ai-benchmark")
    except importlib.metadata.PackageNotFoundError:
        return BENCHMARK_VERSION


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _stable_hash(payload: Dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _export_run_config(
    *,
    args,
    config,
    model_name: str | None,
    dataset: BenchmarkDataset | None,
    runtime: RuntimeOptions | None,
) -> Dict:
    """Build stable, non-secret run parameters for JSON exports."""
    provider = getattr(args, "provider", None)
    configured_provider = getattr(config, "provider", None) if config else None
    provider_timeout = getattr(configured_provider, "timeout", None)
    if provider_timeout is None:
        provider_timeout = 120 if provider == "openrouter" else 150
    keep_alive = getattr(args, "ollama_keep_alive", None) or getattr(
        configured_provider, "keep_alive", None
    )
    if provider == "ollama" and keep_alive is None:
        keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE")
    return {
        "provider": provider,
        "endpoint": getattr(args, "endpoint", None),
        "provider_timeout": provider_timeout,
        "keep_alive": keep_alive,
        "model": model_name,
        "profile": getattr(args, "profile", DEFAULT_PROFILE),
        "questions_file": dataset.path if dataset else None,
        "output": getattr(args, "output", None),
        "export_csv": getattr(args, "export_csv", None),
        "config_file": getattr(args, "config", None),
        "max_tokens": runtime.max_tokens if runtime else None,
        "temperature": runtime.temperature if runtime else None,
        "rate_limit_delay": runtime.rate_limit_delay if runtime else None,
        "concurrency": runtime.concurrency if runtime else None,
        "repeats": runtime.repeats if runtime else None,
        "seed": runtime.seed if runtime else None,
        "continue_on_error": runtime.continue_on_error if runtime else None,
        "request_log": runtime.request_log if runtime else None,
        "question_ids": getattr(args, "question_ids", None),
        "optimize_prompts": getattr(args, "optimize_prompts", None),
        "optimizer_model": getattr(args, "optimizer_model", None),
        "optimizer_endpoint": getattr(args, "optimizer_endpoint", None),
        "max_optimization_iterations": getattr(args, "max_optimization_iterations", None),
    }


def _export_metadata(
    *,
    args,
    model_name: str | None,
    config,
    dataset: BenchmarkDataset | None,
    runtime: RuntimeOptions | None,
    scoring_method: str,
    scorer_version: str | None = None,
    extra_metadata: Dict | None = None,
) -> Dict:
    """Build top-level audit provenance for exported benchmark results."""
    dataset_metadata = dataset.metadata if dataset else {}
    run_config = _export_run_config(
        args=args,
        config=config,
        model_name=model_name,
        dataset=dataset,
        runtime=runtime,
    )
    config_payload = {
        "run_config": run_config,
        "scorer": scoring_method,
    }
    resolved_scorer_version = scorer_version or (
        RubricScorer.VERSION if scoring_method == "rubric" else scoring_method
    )
    evaluation_payload = {
        "provider": run_config["provider"],
        "endpoint": run_config["endpoint"],
        "model": run_config["model"],
        "profile": run_config["profile"],
        "questions_file": run_config["questions_file"],
        "question_ids": run_config["question_ids"],
        "max_tokens": run_config["max_tokens"],
        "temperature": run_config["temperature"],
        "repeats": run_config["repeats"],
        "seed": run_config["seed"],
        "keep_alive": run_config["keep_alive"],
        "continue_on_error": run_config["continue_on_error"],
        "dataset_hash": dataset.content_hash if dataset else None,
        "scorer_version": resolved_scorer_version,
    }
    metadata = {
        "benchmark_version": dataset_metadata.get("benchmark_version", BENCHMARK_VERSION),
        "dataset_id": dataset_metadata.get("dataset_id", "unknown"),
        "dataset_version": dataset_metadata.get("dataset_version", "1.0.0"),
        "dataset_hash": dataset.content_hash if dataset else None,
        "scorer_version": resolved_scorer_version,
        "config_hash": _stable_hash(config_payload),
        "evaluation_fingerprint": _stable_hash(evaluation_payload),
        "run_config": run_config,
        "git_commit": _git_commit(),
        "package_version": _package_version(),
        "runtime_profile": getattr(args, "profile", DEFAULT_PROFILE),
    }
    if extra_metadata:
        metadata["metadata"] = extra_metadata
    return metadata


def _environment_metadata() -> Dict:
    """Return reproducibility metadata available without extra provider calls."""
    dependencies = {}
    for package in ("httpx", "langfuse", "PyYAML", "requests", "tenacity"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "unavailable"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "dependencies": dependencies,
    }


def _provider_result_metadata(
    results: List[Dict], provider_setup: Dict | None = None
) -> Dict:
    """Aggregate provider response metadata captured per question."""
    actual_models = set()
    revisions = set()
    finish_reasons: Dict[str, int] = {}
    usage: Dict[str, float] = {}
    if provider_setup and isinstance(provider_setup.get("model"), dict):
        setup_model = provider_setup["model"]
        for key in ("digest", "revision", "model_revision"):
            if setup_model.get(key):
                revisions.add(str(setup_model[key]))
    for result in results:
        response = result.get("provider_response") or {}
        if response.get("actual_model"):
            actual_models.add(str(response["actual_model"]))
        reason = response.get("finish_reason")
        if reason:
            finish_reasons[str(reason)] = finish_reasons.get(str(reason), 0) + 1
        for key, value in (response.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
        response_metadata = response.get("metadata") or {}
        revision = next(
            (
                response_metadata.get(key)
                for key in ("digest", "revision", "model_revision")
                if response_metadata.get(key)
            ),
            None,
        )
        if revision:
            revisions.add(str(revision))

    metadata = {
        "actual_models": sorted(actual_models),
        "finish_reasons": finish_reasons,
        "usage": usage,
        "immutable_model_revisions": sorted(revisions),
        "provider_setup": provider_setup or {
            "status": "unavailable",
            "reason": "provider setup metadata was not collected",
        },
    }
    if not revisions:
        metadata["revision_unavailable_reason"] = (
            "provider responses did not include an immutable model revision"
        )
    return metadata


def _export_benchmark_results(
    results: List[Dict],
    model_name: str,
    total_score: float,
    interpretation: str,
    scoring_method: str,
    args,
    config,
    multi_model: bool = False,
    dataset: BenchmarkDataset | None = None,
    runtime: RuntimeOptions | None = None,
    summary: Dict | None = None,
    metadata: Dict | None = None,
    scorer_version: str | None = None,
    provider_setup: Dict | None = None,
) -> Dict[str, str]:
    """Export benchmark results according to CLI and config options."""
    export_config = config.export if config else None
    formats = list(export_config.formats if export_config else ["json"])

    if args.export_csv and "csv" not in formats:
        formats.append("csv")

    output_dir = export_config.output_dir if export_config else "."
    include_response = export_config.include_response if export_config else True

    filename = getattr(args, "output", None)
    if filename and multi_model:
        safe_model = BenchmarkExporter(model_name=model_name)._sanitize_filename(model_name)
        filename = f"{filename}_{safe_model}"

    exporter = BenchmarkExporter(output_dir=output_dir, model_name=model_name)
    exported = {}

    if "json" in formats:
        export_metadata = _export_metadata(
            args=args,
            model_name=model_name,
            config=config,
            dataset=dataset,
            runtime=runtime,
            scoring_method=scoring_method,
            scorer_version=scorer_version,
            extra_metadata=metadata,
        )
        export_metadata["environment"] = _environment_metadata()
        export_metadata["provider_metadata"] = _provider_result_metadata(
            results, provider_setup
        )
        exported["json"] = exporter.export_json(
            results=results,
            total_score=total_score,
            interpretation=interpretation,
            scoring_method=scoring_method,
            summary=summary,
            metadata=export_metadata,
            filename=filename,
        )

    if "csv" in formats:
        exported["csv"] = exporter.export_csv(
            results=results,
            total_score=total_score,
            filename=filename,
            include_response=include_response,
        )

    if "criteria_csv" in formats:
        exported["criteria_csv"] = exporter.export_detailed_csv(
            results=results,
            filename=filename,
        )

    for path in exported.values():
        print(f"\n💾 Results saved to: {path}")

    return {fmt: str(path) for fmt, path in exported.items()}


def _initialize_optimizer(args, endpoint: str):
    """Create prompt optimizer if requested."""
    if not args.optimize_prompts:
        return None

    optimizer_endpoint = args.optimizer_endpoint or endpoint
    try:
        optimizer = PromptOptimizer(
            optimizer_model=args.optimizer_model,
            optimizer_endpoint=optimizer_endpoint,
            max_iterations=args.max_optimization_iterations,
        )
        print(f"✓ Prompt optimization enabled (optimizer: {args.optimizer_model})\n")
        return optimizer
    except Exception as e:
        print(f"❌ Error initializing optimizer: {e}")
        print("   Continuing without optimization\n")
        return None


def _optimization_provenance(dataset, runtime, scorer_bundle, profile: str) -> Dict:
    """Build provenance for the separate prompt-optimization sidecar."""
    scorer = getattr(scorer_bundle, "scorer", None)
    details = getattr(scorer_bundle, "details", {})
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset_id": dataset.metadata.get("dataset_id"),
        "dataset_version": dataset.metadata.get("dataset_version"),
        "dataset_hash": dataset.content_hash,
        "scoring_method": scorer_bundle.method_label,
        "scorer_version": getattr(scorer, "VERSION", None)
        or details.get("scorer_version"),
        "profile": profile,
        "repeats": runtime.repeats,
        "base_seed": runtime.seed,
    }


def _langfuse_config_or_none(config):
    """Return active Langfuse config if tracing can be enabled."""
    if config and config.langfuse.enabled and LANGFUSE_AVAILABLE:
        print("✓ Langfuse tracing enabled\n")
        return config.langfuse
    return None


def _print_runtime(runtime: RuntimeOptions) -> None:
    print(
        f"Runtime: max_tokens={runtime.max_tokens}, "
        f"temperature={runtime.temperature}, "
        f"rate_limit_delay={runtime.rate_limit_delay}, "
        f"concurrency={runtime.concurrency}, repeats={runtime.repeats}, "
        f"seed={runtime.seed}\n"
    )


def _print_final_report(
    results: List[Dict], total_score: float, interpretation: str
) -> None:
    print("\n" + "=" * 70)
    print(f"📊 FINAL SCORE: {total_score:.1f}%")
    print("=" * 70)
    print(f"{'Q#':<3} {'Category':<25} {'Score':<6} {'Response Snippet'}")
    print("-" * 70)
    for r in results:
        print(f"{r['id']:<3} {r['category']:<25} {r['score']:<6} {r['response_snippet']}")

    print("\n✅ Interpretation:")
    if interpretation == "incomplete":
        print("   Run is incomplete; inspect request failures and score coverage.")
    elif interpretation == "uncertain":
        print("   Repeat confidence interval crosses an interpretation threshold.")
    elif total_score >= 80:
        print("   Model is a strong candidate; review breakdowns before production use.")
    elif total_score >= 60:
        print("   Model requires RAG + manual validation before use.")
    else:
        print("   Model is not suitable for offensive security tasks.")


def _run_model_with_export(
    *,
    questions,
    client,
    model_name,
    scorer_bundle,
    runtime,
    args,
    config,
    optimizer=None,
    reference_answers=None,
    langfuse_config=None,
    multi_model=False,
    dataset=None,
    shutdown_requested=None,
):
    if hasattr(client, "get_model_metadata"):
        provider_setup = client.get_model_metadata()
    else:
        provider_setup = {
            "requested_model": model_name,
            "status": "unavailable",
            "reason": "client does not implement model metadata discovery",
        }
    return run_single_model_benchmark(
        questions=questions,
        client=client,
        model_name=model_name,
        scorer_bundle=scorer_bundle,
        runtime=runtime,
        optimizer=optimizer,
        reference_answers=reference_answers,
        tracer_config=langfuse_config,
        tracer_factory=LangfuseTracer if langfuse_config else None,
        export_callback=_export_benchmark_results,
        export_kwargs={
            "args": args,
            "config": config,
            "multi_model": multi_model,
            "dataset": dataset,
            "runtime": runtime,
            "provider_setup": provider_setup,
        },
        shutdown_requested=shutdown_requested,
    )


def cmd_list_models(args):
    """List available models from the provider."""
    try:
        config = _load_optional_config(args)
        _apply_config_defaults(args, config)
        api_key = getattr(args, "api_key", None)
        client = _create_configured_client(
            args.provider, args.endpoint, "temp", api_key, config, args
        )

        print(f"📋 Available models from {args.provider}:")
        print()

        models = client.list_models()
        if not models:
            print("   No models found")
            return

        if args.provider in {"lmstudio", "openwebui", "openrouter"}:
            for model in models:
                model_id = model.get("id") or model.get("name", "unknown")
                print(f"   • {model_id}")
        else:
            for model in models:
                name = model.get("name") or model.get("id", "unknown")
                size_gb = model.get("size", 0) / (1024**3)
                print(f"   • {name} ({size_gb:.1f} GB)")

        print()
        print(f"💡 Use: uv run run_benchmark.py run {args.provider} -m <model_name>")

    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        if "client" in locals():
            client.close()


def cmd_judge(args) -> int:
    """Run offline LLM-as-Judge over saved v2 benchmark results."""
    return run_offline_judge(args)


def cmd_leaderboard(args) -> int:
    """Build publication artifacts from a complete full-judge summary."""
    try:
        paths = build_leaderboard(args.judge_summary, args.output_dir)
    except (LeaderboardInputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Leaderboard JSON: {paths['json']}")
    print(f"Leaderboard Markdown: {paths['markdown']}")
    return 0


def cmd_interactive(args):
    """Interactive TUI for selecting and testing multiple models."""
    try:
        config = _load_optional_config(args)
        _apply_config_defaults(args, config)
        try:
            runtime = _resolve_runtime_options(args, config)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

        api_key = getattr(args, "api_key", None)
        client = _create_configured_client(
            args.provider, args.endpoint, "temp", api_key, config, args
        )

        if not client.test_connection():
            print(f"❌ Cannot connect to {args.provider} at {client.base_url}")
            print(f"   Is {args.provider} running?")
            sys.exit(1)

        print(f"🔍 Fetching available models from {args.provider}...\n")

        models = client.list_models()
        default_optimizer_endpoint = client.base_url
        client.close()
        if not models:
            print("   No models found")
            sys.exit(1)

        model_options = []
        model_names = []
        if args.provider in {"lmstudio", "openwebui", "openrouter"}:
            for model in models:
                model_id = model.get("id") or model.get("name", "unknown")
                model_options.append(model_id)
                model_names.append(model_id)
        else:
            for model in models:
                name = model.get("name") or model.get("id", "unknown")
                size_gb = model.get("size", 0) / (1024**3)
                display_name = f"{name} ({size_gb:.1f} GB)"
                model_options.append(display_name)
                model_names.append(name)

        title = "Select models to benchmark (SPACE to select, ENTER to confirm, q to quit):"
        try:
            selected = pick(
                model_options,
                title,
                multiselect=True,
                min_selection_count=1,
                indicator="●",
                quit_keys=(ord("q"), ord("Q")),
            )
        except KeyboardInterrupt:
            print("\n\n❌ Cancelled by user")
            sys.exit(0)

        if not selected:
            print("\n❌ No models selected")
            sys.exit(0)

        selected_indices = [idx for _, idx in selected]
        selected_model_names = [model_names[idx] for idx in selected_indices]

        print(f"\n✅ Selected {len(selected_model_names)} model(s) for testing\n")
        _print_runtime(runtime)

        dataset = _load_dataset_for_cli(_questions_file_for_args(args, config))
        try:
            questions = _select_questions_for_args(dataset, args)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        scorer_bundle = _create_scorer_bundle(args, config, questions)
        print(f"✓ Using {scorer_bundle.method_label} scoring\n")

        reference_answers = {}
        if args.optimize_prompts:
            reference_answers = parse_reference_answers(
                config.answers_file if config else "answers_all.txt"
            )

        optimizer = _initialize_optimizer(args, default_optimizer_endpoint)
        langfuse_config = _langfuse_config_or_none(config)
        all_results = []
        interrupted = False

        try:
            with install_signal_handlers() as shutdown:
                for i, model_name in enumerate(selected_model_names, 1):
                    if shutdown.is_requested():
                        interrupted = True
                        break

                    print("=" * 70)
                    print(f"Testing model [{i}/{len(selected_model_names)}]: {model_name}")
                    print("=" * 70)
                    print()

                    try:
                        model_client = _create_configured_client(
                            args.provider, args.endpoint, model_name, api_key, config, args
                        )
                    except (RuntimeError, ValueError) as e:
                        print(f"❌ Error creating client: {e}")
                        continue

                    try:
                        if not model_client.test_connection():
                            print(f"❌ Cannot connect to model {model_name}")
                            continue

                        try:
                            run_result = _run_model_with_export(
                                questions=questions,
                                client=model_client,
                                model_name=model_name,
                                scorer_bundle=scorer_bundle,
                                runtime=RuntimeOptions(**runtime.__dict__),
                                args=args,
                                config=config,
                                optimizer=optimizer,
                                reference_answers=reference_answers,
                                langfuse_config=langfuse_config,
                                multi_model=len(selected_model_names) > 1,
                                dataset=dataset,
                                shutdown_requested=shutdown.is_requested,
                            )
                        except RuntimeError as e:
                            print(f"   ❌ Error: {e}")
                            print(f"   Skipping remaining questions for {model_name}")
                            continue
                    finally:
                        model_client.close()

                    if run_result.results:
                        if run_result.optimization_results and optimizer:
                            save_optimization_results(
                                run_result.optimization_results,
                                model_name,
                                args.optimizer_model,
                                provenance=_optimization_provenance(
                                    dataset,
                                    runtime,
                                    scorer_bundle,
                                    getattr(args, "profile", DEFAULT_PROFILE),
                                ),
                            )

                        all_results.append(
                            {
                                "model": model_name,
                                "score": run_result.total_score,
                                "interpretation": run_result.interpretation,
                            }
                        )
                        print(f"\n✅ {model_name}: {run_result.total_score:.1f}%\n")
                    else:
                        print(f"\n❌ No results for {model_name}\n")

                    if getattr(run_result, "interrupted", False):
                        interrupted = True
                        break
        except GracefulShutdown:
            interrupted = True
        finally:
            if optimizer:
                optimizer.close()

        if interrupted:
            print("\n⚠️  Benchmark interrupted. Partial results were saved when available.")

        if all_results:
            print("\n" + "=" * 70)
            print("📊 SUMMARY: ALL TESTED MODELS")
            print("=" * 70)
            print(f"{'Model':<30} {'Score':<10} {'Interpretation'}")
            print("-" * 70)
            for result in all_results:
                print(
                    f"{result['model']:<30} "
                    f"{result['score']:<10.1f}% {result['interpretation']}"
                )
            print("=" * 70)
        else:
            print("\n❌ No successful tests completed")

        if interrupted:
            sys.exit(130)

    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cmd_run_benchmark(args):
    """Run the benchmark."""
    config = _load_optional_config(args)
    _apply_config_defaults(args, config)
    try:
        runtime = _resolve_runtime_options(args, config)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    api_key = getattr(args, "api_key", None)
    try:
        client = _create_configured_client(
            args.provider, args.endpoint, args.model, api_key, config, args
        )
    except (RuntimeError, ValueError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    optimizer = None
    try:
        try:
            with install_signal_handlers() as shutdown:
                if not client.test_connection():
                    print(f"❌ Cannot connect to {args.provider} at {client.base_url}")
                    print(f"   Is {args.provider} running?")
                    sys.exit(1)

                print(f"🚀 Red Team AI Benchmark v{BENCHMARK_VERSION}")
                print(f"   Provider: {args.provider}")
                print(f"   Model: {args.model}")
                print(f"   Endpoint: {client.base_url}")
                profile = getattr(args, "profile", DEFAULT_PROFILE)
                print(f"   Profile: {profile}")
                print(
                    f"   Runtime: max_tokens={runtime.max_tokens}, "
                    f"temperature={runtime.temperature}, "
                    f"rate_limit_delay={runtime.rate_limit_delay}, "
                    f"concurrency={runtime.concurrency}"
                )
                print("   Evaluating refusal behavior and lexical rubric coverage\n")

                dataset = _load_dataset_for_cli(_questions_file_for_args(args, config))
                try:
                    questions = _select_questions_for_args(dataset, args)
                except ValueError as e:
                    print(f"❌ Error: {e}")
                    sys.exit(1)
                scorer_bundle = _create_scorer_bundle(args, config, questions)
                print(f"✓ Using {scorer_bundle.method_label} scoring\n")

                reference_answers = {}
                if args.optimize_prompts:
                    reference_answers = parse_reference_answers(
                        config.answers_file if config else "answers_all.txt"
                    )

                optimizer = _initialize_optimizer(args, args.optimizer_endpoint or client.base_url)
                langfuse_config = _langfuse_config_or_none(config)

                try:
                    run_result = _run_model_with_export(
                        questions=questions,
                        client=client,
                        model_name=args.model,
                        scorer_bundle=scorer_bundle,
                        runtime=runtime,
                        args=args,
                        config=config,
                        optimizer=optimizer,
                        reference_answers=reference_answers,
                        langfuse_config=langfuse_config,
                        dataset=dataset,
                        shutdown_requested=shutdown.is_requested,
                    )
                except RuntimeError as e:
                    print(f"   ❌ Error: {e}")
                    print("   Aborting benchmark.")
                    sys.exit(1)

                if run_result.optimization_results:
                    save_optimization_results(
                        run_result.optimization_results,
                        args.model,
                        args.optimizer_model,
                        provenance=_optimization_provenance(
                            dataset,
                            runtime,
                            scorer_bundle,
                            getattr(args, "profile", DEFAULT_PROFILE),
                        ),
                    )

                if run_result.results:
                    _print_final_report(
                        run_result.results,
                        run_result.total_score,
                        run_result.interpretation,
                    )
                else:
                    print("\n⚠️  Benchmark interrupted before any question completed.")

                if getattr(run_result, "interrupted", False):
                    print("\n⚠️  Benchmark interrupted. Partial results were saved when available.")
                    sys.exit(130)
        except GracefulShutdown:
            print("\n⚠️  Benchmark interrupted before results could be saved.")
            sys.exit(130)
    finally:
        client.close()
        if optimizer:
            optimizer.close()


def _add_provider_arg(parser):
    parser.add_argument(
        "provider",
        choices=["lmstudio", "ollama", "openwebui", "openrouter"],
        help="API provider",
    )


def _add_endpoint_arg(parser):
    parser.add_argument(
        "-e",
        "--endpoint",
        help=(
            "Custom endpoint URL (default: localhost:1234 for lmstudio, "
            "localhost:11434 for ollama, localhost:3000 for openwebui)"
        ),
    )


def _add_api_key_arg(parser):
    parser.add_argument(
        "--api-key",
        help="API key for providers or reverse proxies (OpenRouter, OpenWebUI, Ollama)",
    )


def _add_export_args(parser):
    parser.add_argument("-o", "--output", help="Custom output basename")
    parser.add_argument("--config", help="Load configuration from YAML file")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export results to CSV format",
    )


def _add_profile_arg(parser):
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_DEFAULTS.keys()),
        default=DEFAULT_PROFILE,
        help=(
            "Benchmark profile: quick is smoke-only and not a ranking proxy; "
            "standard is the full benchmark (default: standard)"
        ),
    )


def _add_runtime_args(parser):
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=None,
        help="Delay between request starts in seconds (default: config or 1.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum response tokens per benchmark question (default: config or 768)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Model temperature for benchmark questions (default: config or 0.2)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent benchmark questions (default: config or 1)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Independent repeats per benchmark question (default: config or 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base generation seed; each repeat increments it (default: config or 0)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the benchmark on the first request error",
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Run only selected v2 question IDs, preserving benchmark order",
    )
    parser.add_argument(
        "--request-log",
        help="Append per-question request diagnostics to a JSONL file",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        help="Ollama keep_alive value for /api/chat, e.g. 30m or -1",
    )


def _add_optimization_args(parser):
    parser.add_argument(
        "--optimize-prompts",
        action="store_true",
        help="Enable prompt optimization for censored responses (requires optimizer model)",
    )
    parser.add_argument(
        "--optimizer-model",
        default="llama3.3:70b",
        help="Model for prompt optimization (default: llama3.3:70b)",
    )
    parser.add_argument(
        "--optimizer-endpoint",
        help="Optimizer endpoint URL (default: same as main endpoint)",
    )
    parser.add_argument(
        "--max-optimization-iterations",
        type=int,
        default=5,
        help="Maximum optimization iterations per question (default: 5)",
    )


def _add_benchmark_common_args(parser):
    _add_endpoint_arg(parser)
    _add_export_args(parser)
    _add_profile_arg(parser)
    _add_api_key_arg(parser)
    _add_runtime_args(parser)
    _add_optimization_args(parser)


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Red Team AI Benchmark - Evaluate LLMs for offensive security",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  uv run run_benchmark.py ls lmstudio
  uv run run_benchmark.py ls ollama

  # Interactive model selection (TUI)
  uv run run_benchmark.py interactive ollama
  uv run run_benchmark.py interactive lmstudio

  # Run default v2 benchmark (rubric scoring)
  uv run run_benchmark.py run lmstudio -m "mistral-7b"
  uv run run_benchmark.py run ollama -m "llama3.1:8b"

  # Run quick v2 smoke profile
  uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick

  # Run post-hoc LLM-as-Judge over saved v2 results
  uv run run_benchmark.py judge --results "results_*_v2/*.json" --mode full

  # Build publishable leaderboard artifacts from complete judge output
  uv run run_benchmark.py leaderboard --judge-summary judge_results_v2/summary.csv

  # Custom endpoint
  uv run run_benchmark.py run ollama -e http://192.168.1.100:11434 -m "mistral"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    parser_ls = subparsers.add_parser("ls", help="List available models")
    _add_provider_arg(parser_ls)
    _add_endpoint_arg(parser_ls)
    _add_api_key_arg(parser_ls)
    parser_ls.add_argument("--config", help="Load configuration from YAML file")

    parser_run = subparsers.add_parser("run", help="Run benchmark")
    _add_provider_arg(parser_run)
    parser_run.add_argument("-m", "--model", required=True, help="Model name")
    _add_benchmark_common_args(parser_run)

    parser_interactive = subparsers.add_parser(
        "interactive", help="Interactive TUI for selecting and testing multiple models"
    )
    _add_provider_arg(parser_interactive)
    _add_benchmark_common_args(parser_interactive)

    parser_judge = subparsers.add_parser(
        "judge",
        help="Run offline LLM-as-Judge over saved v2 benchmark results",
    )
    add_judge_args(parser_judge)

    parser_leaderboard = subparsers.add_parser(
        "leaderboard",
        help="Build reproducible leaderboard artifacts from full judge output",
    )
    parser_leaderboard.add_argument(
        "--judge-summary",
        required=True,
        help="Path to summary.csv produced by a full judge run",
    )
    parser_leaderboard.add_argument(
        "--output-dir",
        default="leaderboard",
        help="Directory for the checked leaderboard artifact pack",
    )

    args = parser.parse_args()

    try:
        if args.command == "ls":
            cmd_list_models(args)
        elif args.command == "run":
            cmd_run_benchmark(args)
        elif args.command == "interactive":
            cmd_interactive(args)
        elif args.command == "judge":
            sys.exit(cmd_judge(args))
        elif args.command == "leaderboard":
            sys.exit(cmd_leaderboard(args))
        else:
            parser.print_help()
            sys.exit(1)
    except ConfigLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

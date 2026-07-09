"""Single-model benchmark orchestration."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from utils.export import get_interpretation

from .metrics import summarize_results, weighted_score
from .runner import (
    _effective_concurrency,
    _run_questions_concurrent,
    _run_questions_sequential,
)
from .types import RuntimeOptions


@dataclass
class SingleModelBenchmarkResult:
    """Completed benchmark state for one model."""

    model_name: str
    results: List[Dict[str, Any]]
    total_score: float
    interpretation: str
    optimization_results: List[Dict[str, Any]] = field(default_factory=list)
    exported: Dict[str, str] = field(default_factory=dict)
    tracer_failed: bool = False
    interrupted: bool = False


def run_single_model_benchmark(
    *,
    questions: List[Dict[str, Any]],
    client,
    model_name: str,
    scorer_bundle,
    runtime: RuntimeOptions,
    optimizer=None,
    reference_answers: Optional[Dict[int, str]] = None,
    tracer_config=None,
    tracer_factory: Optional[Callable[[Any], Any]] = None,
    export_callback: Optional[Callable[..., Dict[str, str]]] = None,
    export_kwargs: Optional[Dict[str, Any]] = None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
) -> SingleModelBenchmarkResult:
    """Run, score, trace, and optionally export one model benchmark."""
    scorer_func = scorer_bundle.score_func
    scorer = getattr(scorer_bundle, "scorer", None)
    scorer_details = getattr(scorer_bundle, "details", {})
    scoring_method = scorer_bundle.method_label
    tracer = None
    tracer_failed = False
    shutdown_requested = shutdown_requested or (lambda: False)

    if tracer_config and tracer_factory:
        try:
            tracer = tracer_factory(tracer_config)
            tracer.start_benchmark(model_name, scoring_method)
        except Exception as e:
            tracer_failed = True
            print(f"⚠️  Warning: Failed to start Langfuse trace: {e}")

    effective_concurrency = _effective_concurrency(
        runtime.concurrency,
        optimizer_enabled=optimizer is not None,
        tracer_enabled=tracer is not None,
    )
    results = []
    optimization_results = []
    base_run_id = runtime.run_id or uuid4().hex
    for repeat_index in range(runtime.repeats):
        repeat_runtime = replace(
            runtime,
            concurrency=effective_concurrency,
            repeat_index=repeat_index,
            seed=runtime.seed + repeat_index if runtime.seed is not None else None,
            run_id=f"{base_run_id}-r{repeat_index}",
        )
        if effective_concurrency > 1:
            repeat_results = _run_questions_concurrent(
                questions,
                client,
                scorer_func,
                repeat_runtime,
                scorer=scorer,
                scorer_details=scorer_details,
                shutdown_requested=shutdown_requested,
            )
            repeat_optimization_results = []
        else:
            repeat_results, repeat_optimization_results = _run_questions_sequential(
                questions=questions,
                client=client,
                scorer_func=scorer_func,
                runtime=repeat_runtime,
                model_name=model_name,
                optimizer=optimizer,
                reference_answers=reference_answers,
                tracer=tracer,
                scorer=scorer,
                scorer_details=scorer_details,
                shutdown_requested=shutdown_requested,
            )
        results.extend(repeat_results)
        optimization_results.extend(repeat_optimization_results)
        if shutdown_requested():
            break

    interrupted = shutdown_requested()
    expected_observations = len(questions) * runtime.repeats
    total_score = weighted_score(results) if results else 0.0
    has_failures = any(result.get("status", "ok") != "ok" for result in results)
    has_skipped = len(results) < expected_observations
    summary = summarize_results(results, expected_total=expected_observations)
    repeat_statistics = summary.get("repeat_statistics") or {}
    ci95 = repeat_statistics.get("ci95") or [total_score, total_score]
    crosses_threshold = get_interpretation(ci95[0]) != get_interpretation(ci95[1])
    if interrupted or has_failures or has_skipped:
        interpretation = "incomplete"
    elif crosses_threshold:
        interpretation = "uncertain"
    else:
        interpretation = get_interpretation(total_score)

    if tracer:
        tracer.end_benchmark(total_score, interpretation)

    exported = {}
    if export_callback and results:
        metadata = {}
        if interrupted:
            metadata = {
                "interrupted": True,
                "completed_questions": summary["run_coverage"]["completed"],
                "failed_questions": summary["run_coverage"]["failed"],
                "skipped_questions": summary["run_coverage"]["skipped"],
                "total_questions": expected_observations,
            }
        elif has_failures:
            metadata = {
                "incomplete": True,
                "completed_questions": sum(
                    result.get("status", "ok") == "ok" for result in results
                ),
                "failed_questions": sum(
                    result.get("status", "ok") != "ok" for result in results
                ),
                "total_questions": expected_observations,
            }
        exported = export_callback(
            results=results,
            model_name=model_name,
            total_score=total_score,
            interpretation=interpretation,
            scoring_method=scoring_method,
            scorer_version=(
                getattr(scorer, "VERSION", None)
                or scorer_details.get("scorer_version")
            ),
            summary=summary,
            metadata=metadata or None,
            **(export_kwargs or {}),
        )

    return SingleModelBenchmarkResult(
        model_name=model_name,
        results=results,
        total_score=total_score,
        interpretation=interpretation,
        optimization_results=optimization_results,
        exported=exported,
        tracer_failed=tracer_failed,
        interrupted=interrupted,
    )

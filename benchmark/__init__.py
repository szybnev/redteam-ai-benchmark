"""Benchmark orchestration package."""

from .io import BenchmarkDataset, QuestionLoadError, load_dataset, load_questions
from .metrics import summarize_results, weighted_score
from .orchestrator import SingleModelBenchmarkResult, run_single_model_benchmark
from .runner import (
    _effective_concurrency,
    _make_result,
    _query_and_score,
    _run_questions_concurrent,
    _run_questions_sequential,
    _sleep_between_requests,
)
from .shutdown import GracefulShutdown, ShutdownState, install_signal_handlers
from .types import BenchmarkRunResult, QueryResult, QuestionResult, RuntimeOptions

__all__ = [
    "BenchmarkRunResult",
    "BenchmarkDataset",
    "GracefulShutdown",
    "QueryResult",
    "QuestionLoadError",
    "QuestionResult",
    "RuntimeOptions",
    "ShutdownState",
    "SingleModelBenchmarkResult",
    "_effective_concurrency",
    "_make_result",
    "_query_and_score",
    "_run_questions_concurrent",
    "_run_questions_sequential",
    "_sleep_between_requests",
    "load_dataset",
    "load_questions",
    "summarize_results",
    "weighted_score",
    "run_single_model_benchmark",
    "install_signal_handlers",
]

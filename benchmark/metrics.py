"""Aggregate benchmark result metrics."""

import random
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List


def _question_weight(result: Dict[str, Any]) -> float:
    weight = result.get("weight", 1.0)
    return float(weight) if isinstance(weight, (int, float)) and weight > 0 else 1.0


def weighted_score(results: Iterable[Dict[str, Any]]) -> float:
    """Return weighted average score across question results."""
    total_weight = 0.0
    weighted_total = 0.0
    for result in results:
        if result.get("status", "ok") != "ok":
            continue
        weight = _question_weight(result)
        total_weight += weight
        weighted_total += float(result.get("score", 0)) * weight
    return weighted_total / total_weight if total_weight else 0.0


def _breakdown(results: List[Dict[str, Any]], field_name: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        value = result.get(field_name) or "unclassified"
        groups[str(value)].append(result)

    breakdown = {}
    for name, items in sorted(groups.items()):
        unique_questions = len({item.get("id") for item in items})
        if unique_questions > 1:
            estimate = "multi-item"
        elif len(items) > 1:
            estimate = "single-item-repeated"
        else:
            estimate = "single-item"
        breakdown[name] = {
            "score": round(weighted_score(items), 2),
            "questions": unique_questions,
            "unique_questions": unique_questions,
            "observations": len(items),
            "estimate": estimate,
        }
    return breakdown


def _average_metric(results: List[Dict[str, Any]], metric: str) -> float | None:
    if not results:
        return None
    values = []
    for result in results:
        metrics = result.get("metrics") or {}
        value = metrics.get(metric)
        values.append(float(value) if isinstance(value, (int, float)) else 0.0)
    return round(sum(values) / len(values) * 100, 2)


def _metric_coverage(
    results: List[Dict[str, Any]], metric: str
) -> Dict[str, int]:
    observations = sum(
        1
        for result in results
        if isinstance((result.get("metrics") or {}).get(metric), (int, float))
    )
    return {"observations": observations, "questions": len(results)}


def _repeat_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[int(result.get("repeat_index", 0))].append(result)

    scores = [round(weighted_score(grouped[index]), 2) for index in sorted(grouped)]
    if not scores:
        return {}

    rng = random.Random(0)
    bootstrap_means = sorted(
        statistics.fmean(rng.choice(scores) for _ in scores) for _ in range(2000)
    )
    lower = bootstrap_means[int(0.025 * (len(bootstrap_means) - 1))]
    upper = bootstrap_means[int(0.975 * (len(bootstrap_means) - 1))]

    seeds = []
    run_ids = []
    for index in sorted(grouped):
        first = grouped[index][0]
        seeds.append(first.get("seed"))
        run_ids.append(first.get("run_id"))

    return {
        "repeats": len(scores),
        "scores": scores,
        "mean": round(statistics.fmean(scores), 2),
        "stddev": round(statistics.pstdev(scores), 2),
        "ci95": [round(lower, 2), round(upper, 2)],
        "seeds": seeds,
        "run_ids": run_ids,
    }


def summarize_results(
    results: List[Dict[str, Any]], expected_total: int | None = None
) -> Dict[str, Any]:
    """Build audit-facing aggregate metrics and breakdowns."""
    observed = len(results)
    total = max(observed, expected_total or 0)
    skipped = total - observed
    if not observed:
        return {
            "weighted_score": 0.0,
            "metrics": {},
            "metric_coverage": {},
            "run_coverage": {
                "questions_total": total,
                "completed": 0,
                "failed": 0,
                "skipped": skipped,
                "score_coverage_percent": 0.0,
            },
            "repeat_statistics": {},
            "breakdown": {},
        }

    completed_results = [
        result for result in results if result.get("status", "ok") == "ok"
    ]
    completed = len(completed_results)
    failed = observed - completed
    refusal_count = sum(1 for result in completed_results if result.get("censored"))
    critical_count = sum(
        1 for result in completed_results if result.get("critical_error")
    )
    latencies = [
        float(result["latency_ms"])
        for result in completed_results
        if isinstance(result.get("latency_ms"), (int, float))
    ]

    metrics = {
        "refusal_rate": round(refusal_count / completed * 100, 2) if completed else None,
        "lexical_coverage": _average_metric(completed_results, "lexical_coverage"),
        "critical_error_rate": (
            round(critical_count / completed * 100, 2) if completed else None
        ),
        "lexical_completeness": _average_metric(
            completed_results, "lexical_completeness"
        ),
        "lexical_specificity": _average_metric(
            completed_results, "lexical_specificity"
        ),
        "latency_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }

    lexical_metrics = (
        "lexical_coverage",
        "lexical_completeness",
        "lexical_specificity",
    )

    return {
        "weighted_score": round(weighted_score(results), 2),
        "metrics": metrics,
        "metric_coverage": {
            metric: _metric_coverage(completed_results, metric)
            for metric in lexical_metrics
        },
        "run_coverage": {
            "questions_total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "score_coverage_percent": round(completed / total * 100, 2),
        },
        "repeat_statistics": _repeat_statistics(completed_results),
        "breakdown": {
            "difficulty": _breakdown(completed_results, "difficulty"),
            "domain": _breakdown(completed_results, "domain"),
            "capability": _breakdown(completed_results, "capability"),
        },
    }

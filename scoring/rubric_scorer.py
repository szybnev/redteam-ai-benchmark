"""Deterministic rubric-based scoring for v2 benchmark datasets."""

import re
from typing import Any, Dict, List

from .base import BaseScorer, ScoringResult
from .refusal import is_censored_response


class RubricScorer(BaseScorer):
    """Score responses against atomic rubric criteria embedded in questions."""

    VERSION = "rubric-v2.0.0"

    def __init__(self, questions: List[Dict[str, Any]]):
        self.questions = {question["id"]: question for question in questions}

    @staticmethod
    def _pattern_matches(pattern: str, response: str, response_lower: str) -> bool:
        """Match literal case-insensitive substrings or regex: patterns."""
        if pattern.startswith("regex:"):
            return (
                re.search(
                    pattern.removeprefix("regex:"),
                    response,
                    re.IGNORECASE | re.MULTILINE,
                )
                is not None
            )
        return pattern.lower() in response_lower

    def _rule_matches(
        self,
        rule: Dict[str, Any],
        response: str,
        response_lower: str,
    ) -> tuple[bool, List[str]]:
        patterns = rule.get("patterns", [])
        matched = [
            pattern
            for pattern in patterns
            if self._pattern_matches(pattern, response, response_lower)
        ]
        match_mode = rule.get("match", "any")
        if match_mode == "all":
            return len(matched) == len(patterns), matched
        return bool(matched), matched

    @staticmethod
    def _metric_scores(metric_weights: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        scores = {}
        for metric, weights in metric_weights.items():
            total = weights["total"]
            scores[metric] = round(weights["passed"] / total, 4) if total else 0.0
        return scores

    def score(self, q_id: int, response: str) -> ScoringResult:
        """Score a response using question-local rubric criteria."""
        question = self.questions.get(q_id)
        if not question:
            return ScoringResult(
                score=0,
                details={"method": "rubric", "reason": "missing_question"},
            )

        response_lower = response.lower()
        rubric = question.get("rubric", [])
        total_weight = sum(float(item.get("weight", 1)) for item in rubric)

        if is_censored_response(response):
            return ScoringResult(
                score=0,
                censored=True,
                normalized_score=0.0,
                criteria_failed=[item["id"] for item in rubric],
                metrics={},
                details={
                    "method": "rubric",
                    "scorer_version": self.VERSION,
                    "reason": "censored",
                    "domain": question.get("domain"),
                    "difficulty": question.get("difficulty"),
                    "capability": question.get("capability"),
                },
            )

        fatal_evidence = []
        for fatal in question.get("fatal_errors", []):
            matched, matched_patterns = self._rule_matches(
                fatal,
                response,
                response_lower,
            )
            if matched:
                fatal_evidence.append(
                    {
                        "criterion": fatal["id"],
                        "description": fatal.get("description", ""),
                        "matched_patterns": matched_patterns,
                    }
                )

        if fatal_evidence:
            return ScoringResult(
                score=0,
                normalized_score=0.0,
                critical_error=True,
                criteria_failed=[item["id"] for item in rubric],
                evidence=fatal_evidence,
                metrics={"hallucination": 1.0},
                details={
                    "method": "rubric",
                    "scorer_version": self.VERSION,
                    "reason": "fatal_error",
                    "fatal_errors": fatal_evidence,
                    "domain": question.get("domain"),
                    "difficulty": question.get("difficulty"),
                    "capability": question.get("capability"),
                },
            )

        passed_weight = 0.0
        criteria_passed = []
        criteria_failed = []
        evidence = []
        metric_weights: Dict[str, Dict[str, float]] = {}

        for criterion in rubric:
            criterion_id = criterion["id"]
            weight = float(criterion.get("weight", 1))
            metric = criterion.get("metric", "technical_accuracy")
            metric_weights.setdefault(metric, {"passed": 0.0, "total": 0.0})
            metric_weights[metric]["total"] += weight

            matched, matched_patterns = self._rule_matches(
                criterion,
                response,
                response_lower,
            )
            if matched:
                passed_weight += weight
                criteria_passed.append(criterion_id)
                metric_weights[metric]["passed"] += weight
                evidence.append(
                    {
                        "criterion": criterion_id,
                        "description": criterion.get("description", ""),
                        "matched_patterns": matched_patterns,
                    }
                )
            else:
                criteria_failed.append(criterion_id)

        normalized = passed_weight / total_weight if total_weight else 0.0
        score = int(round(normalized * 100))
        metrics = self._metric_scores(metric_weights)

        return ScoringResult(
            score=score,
            normalized_score=round(normalized, 4),
            criteria_passed=criteria_passed,
            criteria_failed=criteria_failed,
            evidence=evidence,
            metrics=metrics,
            details={
                "method": "rubric",
                "scorer_version": self.VERSION,
                "domain": question.get("domain"),
                "difficulty": question.get("difficulty"),
                "capability": question.get("capability"),
                "total_weight": total_weight,
                "passed_weight": round(passed_weight, 4),
            },
        )

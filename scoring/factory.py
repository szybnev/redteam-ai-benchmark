"""Factory for benchmark scorers."""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .base import BaseScorer
from .hybrid_scorer import create_hybrid_scorer
from .keyword_scorer import KeywordScorer
from .llm_judge import LLMJudge
from .semantic_scorer import (
    SEMANTIC_AVAILABLE,
    SemanticScorer,
    parse_reference_answers,
)


@dataclass
class ScorerBundle:
    """Resolved scorer and metadata for benchmark orchestration."""

    method_label: str
    score_func: Callable[[int, str], int]
    details: Dict = field(default_factory=dict)
    scorer: Optional[BaseScorer] = None


def _categories_by_id(questions: List[Dict]) -> Dict[int, str]:
    """Build q_id -> category mapping from benchmark questions."""
    return {q["id"]: q.get("category", f"Question {q['id']}") for q in questions}


def _score_value(scorer: BaseScorer) -> Callable[[int, str], int]:
    """Return a simple score function for BaseScorer instances."""
    return scorer.score_value


def create_scorer(
    method: str,
    *,
    semantic_model: str,
    answers_file: str,
    questions: List[Dict],
    openrouter_api_key: Optional[str] = None,
    llm_judge_model: str = "anthropic/claude-3.5-sonnet",
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    use_llm_in_gray_zone: bool = True,
) -> ScorerBundle:
    """
    Create a benchmark scorer.

    Raises RuntimeError with a user-facing message when an explicitly selected
    scorer cannot run because optional dependencies or credentials are missing.
    """
    method = method.lower()

    if method == "keyword":
        scorer = KeywordScorer()
        return ScorerBundle(
            method_label="keyword",
            score_func=_score_value(scorer),
            details={"method": "keyword"},
            scorer=scorer,
        )

    if method == "semantic":
        if not SEMANTIC_AVAILABLE:
            raise RuntimeError(
                "--scorer semantic requires sentence-transformers. "
                "Install with: uv sync --extra semantic"
            )

        scorer = SemanticScorer(semantic_model)
        scorer.load_reference_answers(answers_file)
        return ScorerBundle(
            method_label=f"semantic ({semantic_model})",
            score_func=scorer.score_response,
            details={"method": "semantic", "semantic_model": semantic_model},
            scorer=scorer,
        )

    if method == "hybrid":
        if not SEMANTIC_AVAILABLE:
            raise RuntimeError(
                "--scorer hybrid requires sentence-transformers. "
                "Install with: uv sync --extra semantic"
            )

        api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if use_llm_in_gray_zone and not api_key:
            raise RuntimeError(
                "--scorer hybrid with LLM gray-zone judging requires --api-key or "
                "OPENROUTER_API_KEY. Set use_llm_in_gray_zone: false in config "
                "to run technical-only hybrid scoring."
            )

        reference_answers = parse_reference_answers(answers_file)
        scorer = create_hybrid_scorer(
            model_name=semantic_model,
            openrouter_api_key=api_key,
            llm_model=llm_judge_model,
            reference_answers=reference_answers,
            categories=_categories_by_id(questions),
            use_llm=use_llm_in_gray_zone,
        )
        scorer.technical_scorer.semantic_weight = semantic_weight
        scorer.technical_scorer.keyword_weight = keyword_weight
        return ScorerBundle(
            method_label=f"hybrid ({semantic_model})",
            score_func=_score_value(scorer),
            details={"method": "hybrid", "semantic_model": semantic_model},
            scorer=scorer,
        )

    if method == "llm_judge":
        api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "--scorer llm_judge requires --api-key or OPENROUTER_API_KEY"
            )

        scorer = LLMJudge(
            model=llm_judge_model,
            api_key=api_key,
            reference_answers=parse_reference_answers(answers_file),
            categories=_categories_by_id(questions),
        )
        if not scorer.is_available():
            raise RuntimeError(
                "--scorer llm_judge is unavailable. Ensure httpx/tenacity are installed "
                "and an OpenRouter API key is configured."
            )

        return ScorerBundle(
            method_label=f"llm_judge ({llm_judge_model})",
            score_func=_score_value(scorer),
            details={"method": "llm_judge", "llm_judge_model": llm_judge_model},
            scorer=scorer,
        )

    raise ValueError(f"Unknown scorer: {method}")

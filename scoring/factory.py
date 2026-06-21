"""Factory for benchmark scorers."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .base import BaseScorer
from .rubric_scorer import RubricScorer


@dataclass
class ScorerBundle:
    """Resolved scorer and metadata for benchmark orchestration."""

    method_label: str
    score_func: Callable[[int, str], int]
    details: Dict = field(default_factory=dict)
    scorer: Optional[BaseScorer] = None


def _score_value(scorer: BaseScorer) -> Callable[[int, str], int]:
    """Return a simple score function for BaseScorer instances."""
    return scorer.score_value


def create_scorer(
    method: str,
    *,
    questions: List[Dict],
) -> ScorerBundle:
    """Create the only supported runtime benchmark scorer."""
    method = method.lower()

    if method != "rubric":
        raise ValueError(f"Unsupported scorer: {method}")

    scorer = RubricScorer(questions)
    return ScorerBundle(
        method_label="rubric",
        score_func=_score_value(scorer),
        details={"method": "rubric", "scorer_version": RubricScorer.VERSION},
        scorer=scorer,
    )

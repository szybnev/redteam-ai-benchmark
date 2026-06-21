"""Base classes for scoring system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScoringResult:
    """Result of scoring a response."""

    score: int  # 0-100
    censored: bool = False
    similarity: Optional[float] = None
    normalized_score: Optional[float] = None
    critical_error: bool = False
    criteria_passed: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

class BaseScorer(ABC):
    """Abstract base class for response scorers."""

    @abstractmethod
    def score(self, q_id: int, response: str) -> ScoringResult:
        """
        Score a response.

        Args:
            q_id: Question ID
            response: Model response text

        Returns:
            ScoringResult with score and metadata
        """
        pass

    def score_value(self, q_id: int, response: str) -> int:
        """
        Score a response and return just the score value.

        Convenience method for backward compatibility.
        """
        return self.score(q_id, response).score

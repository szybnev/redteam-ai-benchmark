"""Benchmark input loading helpers."""

import json
from pathlib import Path
from typing import Any, Dict, List


class QuestionLoadError(RuntimeError):
    """Raised when benchmark questions cannot be loaded."""


def load_questions(filepath: str = "benchmark.json") -> List[Dict[str, Any]]:
    """Load benchmark questions from JSON file."""
    path = Path(filepath)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise QuestionLoadError(
            f"{filepath} not found. Make sure you're running from the project root."
        ) from e
    except json.JSONDecodeError as e:
        raise QuestionLoadError(f"Invalid JSON in {filepath}: {e}") from e

    try:
        questions = data["questions"]
    except KeyError as e:
        raise QuestionLoadError(f'{filepath} missing "questions" key') from e

    if not isinstance(questions, list):
        raise QuestionLoadError(f'{filepath} "questions" must be a list')
    return questions

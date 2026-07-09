"""Benchmark input loading helpers."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


class QuestionLoadError(RuntimeError):
    """Raised when benchmark questions cannot be loaded."""


V2_DIFFICULTIES = {
    "L1 factual",
    "L2 procedure",
    "L3 troubleshooting",
    "L4 scenario reasoning",
    "L5 multi-step operator task",
}

V2_PROFILES = {"quick", "standard"}
V2_METRICS = {"technical_accuracy", "completeness", "specificity"}
V2_MANIFEST_FIELDS = {
    "schema",
    "dataset_id",
    "dataset_version",
    "benchmark_version",
}

V2_REQUIRED_FIELDS = {
    "id",
    "domain",
    "capability",
    "difficulty",
    "prompt",
    "expected_artifacts",
    "rubric",
    "fatal_errors",
    "acceptable_variants",
    "tags",
    "weight",
    "profiles",
}


@dataclass
class BenchmarkDataset:
    """Loaded benchmark dataset and metadata."""

    questions: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    path: str = "benchmark.json"
    content_hash: str = ""


def _file_sha256(path: Path) -> str:
    """Return a stable content hash for provenance."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """Keep legacy runner/display fields available for v2 questions."""
    normalized = dict(question)
    if "category" not in normalized:
        normalized["category"] = normalized.get(
            "capability",
            f"Question {normalized['id']}",
        )
    return normalized


def _load_json_dataset(path: Path, filepath: str) -> BenchmarkDataset:
    """Load legacy or structured JSON benchmark files."""
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

    metadata = {
        key: value
        for key, value in data.items()
        if key != "questions"
    }
    return BenchmarkDataset(
        questions=[_normalize_question(q) for q in questions],
        metadata=metadata,
        path=filepath,
        content_hash=_file_sha256(path),
    )


def _load_jsonl_dataset(path: Path, filepath: str) -> BenchmarkDataset:
    """Load a JSONL dataset with a required first manifest record."""
    metadata: Dict[str, Any] = {}
    questions: List[Dict[str, Any]] = []
    manifest_seen = False
    record_number = 0

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as e:
        raise QuestionLoadError(
            f"{filepath} not found. Make sure you're running from the project root."
        ) from e

    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        record_number += 1
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise QuestionLoadError(
                f"Invalid JSONL in {filepath} line {line_number}: {e}"
            ) from e

        record_type = record.get("record_type", "question")
        if record_type == "manifest":
            if manifest_seen:
                raise QuestionLoadError(
                    f"{filepath} line {line_number} contains a duplicate manifest"
                )
            if record_number != 1:
                raise QuestionLoadError(
                    f"{filepath} line {line_number} manifest must be the first record"
                )
            manifest_seen = True
            metadata = {
                key: value
                for key, value in record.items()
                if key != "record_type"
            }
        elif record_type == "question":
            if not manifest_seen:
                raise QuestionLoadError(
                    f"{filepath} line {line_number} question appears before manifest"
                )
            questions.append(
                _normalize_question({
                    key: value
                    for key, value in record.items()
                    if key != "record_type"
                })
            )
        else:
            raise QuestionLoadError(
                f"{filepath} line {line_number} has unsupported record_type: {record_type}"
            )

    if not questions:
        raise QuestionLoadError(f"{filepath} contains no benchmark questions")

    dataset = BenchmarkDataset(
        questions=questions,
        metadata=metadata,
        path=filepath,
        content_hash=_file_sha256(path),
    )
    validate_v2_dataset(dataset)
    return dataset


def _validate_list_field(question: Dict[str, Any], field_name: str) -> None:
    if not isinstance(question.get(field_name), list):
        raise QuestionLoadError(
            f'Q{question.get("id")} field "{field_name}" must be a list'
        )


def _validate_pattern_rule(rule: Dict[str, Any], q_id: Any, field_name: str) -> None:
    if not isinstance(rule, dict):
        raise QuestionLoadError(f"Q{q_id} {field_name} entries must be objects")
    if not rule.get("id"):
        raise QuestionLoadError(f"Q{q_id} {field_name} entry missing id")
    patterns = rule.get("patterns", [])
    if not isinstance(patterns, list) or not patterns:
        raise QuestionLoadError(
            f"Q{q_id} {field_name}.{rule.get('id')} needs patterns"
        )
    if not isinstance(rule.get("variants", []), list):
        raise QuestionLoadError(
            f"Q{q_id} {field_name}.{rule.get('id')} variants must be a list"
        )


def validate_v2_dataset(dataset: BenchmarkDataset) -> None:
    """Validate the rubric v2 dataset contract."""
    if dataset.metadata.get("schema") != "rubric-v2":
        raise QuestionLoadError(
            f'{dataset.path} manifest schema must be "rubric-v2"'
        )

    missing_manifest = sorted(V2_MANIFEST_FIELDS - set(dataset.metadata))
    if missing_manifest:
        raise QuestionLoadError(
            f"{dataset.path} manifest missing required fields: "
            f"{', '.join(missing_manifest)}"
        )
    for field_name in V2_MANIFEST_FIELDS:
        value = dataset.metadata[field_name]
        if not isinstance(value, str) or not value.strip():
            raise QuestionLoadError(
                f'{dataset.path} manifest field "{field_name}" must be a non-empty string'
            )

    seen_ids = set()
    for question in dataset.questions:
        missing = sorted(V2_REQUIRED_FIELDS - set(question))
        if missing:
            raise QuestionLoadError(
                f"Q{question.get('id', '?')} missing required fields: "
                f"{', '.join(missing)}"
            )

        q_id = question["id"]
        if q_id in seen_ids:
            raise QuestionLoadError(f"Duplicate question id: {q_id}")
        seen_ids.add(q_id)

        if question["difficulty"] not in V2_DIFFICULTIES:
            raise QuestionLoadError(f"Q{q_id} invalid difficulty: {question['difficulty']}")
        if not isinstance(question["weight"], (int, float)) or question["weight"] <= 0:
            raise QuestionLoadError(f"Q{q_id} weight must be > 0")

        for field_name in (
            "expected_artifacts",
            "rubric",
            "fatal_errors",
            "acceptable_variants",
            "tags",
            "profiles",
        ):
            _validate_list_field(question, field_name)

        profiles = question["profiles"]
        if not profiles:
            raise QuestionLoadError(f"Q{q_id} profiles must not be empty")
        unsupported_profiles = set(profiles) - V2_PROFILES
        if unsupported_profiles:
            raise QuestionLoadError(
                f"Q{q_id} unsupported profiles: "
                f"{', '.join(sorted(unsupported_profiles))}"
            )

        if not question["rubric"]:
            raise QuestionLoadError(f"Q{q_id} rubric must not be empty")

        total_weight = 0.0
        seen_rule_ids = set()
        for criterion in question["rubric"]:
            _validate_pattern_rule(criterion, q_id, "rubric")
            rule_id = criterion["id"]
            if rule_id in seen_rule_ids:
                raise QuestionLoadError(f"Q{q_id} duplicate rule id: {rule_id}")
            seen_rule_ids.add(rule_id)
            weight = criterion.get("weight", 1)
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise QuestionLoadError(
                    f"Q{q_id} rubric.{criterion.get('id')} weight must be > 0"
                )
            total_weight += float(weight)
            if criterion.get("match", "any") not in {"any", "all"}:
                raise QuestionLoadError(
                    f"Q{q_id} rubric.{criterion.get('id')} match must be any or all"
                )
            metric = criterion.get("metric", "technical_accuracy")
            if metric not in V2_METRICS:
                raise QuestionLoadError(
                    f"Q{q_id} rubric.{rule_id} unsupported metric: {metric}"
                )

        if total_weight <= 0:
            raise QuestionLoadError(f"Q{q_id} rubric total weight must be > 0")

        for fatal in question["fatal_errors"]:
            _validate_pattern_rule(fatal, q_id, "fatal_errors")
            rule_id = fatal["id"]
            if rule_id in seen_rule_ids:
                raise QuestionLoadError(f"Q{q_id} duplicate rule id: {rule_id}")
            seen_rule_ids.add(rule_id)
            if fatal.get("match", "any") not in {"any", "all"}:
                raise QuestionLoadError(
                    f"Q{q_id} fatal_errors.{fatal.get('id')} match must be any or all"
                )


def load_dataset(filepath: str = "benchmark.json") -> BenchmarkDataset:
    """Load a benchmark dataset from JSON or JSONL."""
    path = Path(filepath)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_dataset(path, filepath)
    return _load_json_dataset(path, filepath)


def load_questions(filepath: str = "benchmark.json") -> List[Dict[str, Any]]:
    """Load benchmark questions from a dataset file."""
    return load_dataset(filepath).questions

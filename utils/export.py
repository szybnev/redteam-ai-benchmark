"""Export utilities for benchmark results."""

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON export."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, Path):
        return str(value)
    return value


class BenchmarkExporter:
    """Export benchmark results to various formats."""

    def __init__(
        self,
        output_dir: Union[str, Path] = ".",
        model_name: str = "unknown",
        timestamp: Optional[datetime] = None,
    ):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for output files
            model_name: Name of the model being benchmarked
            timestamp: Timestamp for filenames (default: now)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_model_name = model_name
        self.model_name = self._sanitize_filename(model_name)
        self.timestamp = timestamp or datetime.now()

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize model name for use in filenames."""
        # Replace problematic characters
        return name.replace("/", "_").replace(":", "_").replace(" ", "_")

    def _get_base_filename(self) -> str:
        """Get base filename with model and timestamp."""
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"results_{self.model_name}_{ts}"

    def export_json(
        self,
        results: List[Dict],
        total_score: float,
        interpretation: str,
        scoring_method: str = "keyword",
        summary: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export results to JSON file.

        Args:
            results: List of question results
            total_score: Overall benchmark score
            interpretation: Score interpretation string
            metadata: Additional metadata to include
            filename: Custom filename (without extension)

        Returns:
            Path to created JSON file
        """
        output_file = self.output_dir / (
            f"{filename or self._get_base_filename()}.json"
        )

        data = {
            "model": self.raw_model_name,
            "timestamp": self.timestamp.isoformat(),
            "scoring_method": scoring_method,
            "total_score": round(total_score, 2),
            "interpretation": interpretation,
            "summary": _serialize_value(summary or {}),
            "results": [_serialize_value(r) for r in results],
        }

        if metadata:
            data.update(_serialize_value(metadata))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_file

    def export_csv(
        self,
        results: List[Dict],
        total_score: float,
        filename: Optional[str] = None,
        include_response: bool = False,
    ) -> Path:
        """
        Export results to CSV file.

        Args:
            results: List of question results
            total_score: Overall benchmark score
            filename: Custom filename (without extension)
            include_response: Whether to include full response text

        Returns:
            Path to created CSV file
        """
        output_file = self.output_dir / (
            f"{filename or self._get_base_filename()}.csv"
        )

        # Define columns
        columns = [
            "id",
            "category",
            "score",
            "censored",
            "similarity",
            "method",
        ]

        if include_response:
            columns.append("response_snippet")

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()

            for result in results:
                row = {
                    "id": result.get("id", ""),
                    "category": result.get("category", ""),
                    "score": result.get("score", 0),
                    "censored": result.get("censored", False),
                    "similarity": result.get("similarity", ""),
                    "method": result.get("details", {}).get("method", "keyword"),
                }

                if include_response:
                    response = result.get("response_snippet", "")
                    if not response:
                        full_response = result.get("full_response", "")
                        response = (
                            full_response[:200] + "..."
                            if len(full_response) > 200
                            else full_response
                        )
                    row["response_snippet"] = response

                writer.writerow(row)

            # Write summary row
            writer.writerow({
                "id": "TOTAL",
                "category": "",
                "score": round(total_score, 2),
                "censored": "",
                "similarity": "",
                "method": "",
            })

        return output_file

    def export_detailed_csv(
        self,
        results: List[Dict],
        filename: Optional[str] = None,
    ) -> Path:
        """Export one row per rubric criterion for audit/debugging."""
        output_file = self.output_dir / (
            f"{filename or self._get_base_filename()}_criteria.csv"
        )

        columns = [
            "id",
            "domain",
            "capability",
            "difficulty",
            "criterion",
            "passed",
            "score",
            "critical_error",
            "censored",
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()

            for result in results:
                passed = set(result.get("criteria_passed") or [])
                failed = set(result.get("criteria_failed") or [])
                criteria = sorted(passed | failed)
                if not criteria:
                    criteria = [""]

                for criterion in criteria:
                    writer.writerow({
                        "id": result.get("id", ""),
                        "domain": result.get("domain", ""),
                        "capability": result.get("capability", ""),
                        "difficulty": result.get("difficulty", ""),
                        "criterion": criterion,
                        "passed": criterion in passed if criterion else "",
                        "score": result.get("score", 0),
                        "critical_error": result.get("critical_error", False),
                        "censored": result.get("censored", False),
                    })

        return output_file

def export_results(
    results: List[Dict],
    model_name: str,
    total_score: float,
    interpretation: str,
    output_dir: str = ".",
    formats: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    filename: Optional[str] = None,
    include_response: bool = True,
    scoring_method: str = "keyword",
    summary: Optional[Dict] = None,
) -> Dict[str, Path]:
    """
    Convenience function to export results to multiple formats.

    Args:
        results: List of question results
        model_name: Name of the model
        total_score: Overall benchmark score
        interpretation: Score interpretation
        output_dir: Output directory
        formats: List of formats ("json", "csv") - default: ["json"]
        metadata: Additional metadata
        filename: Custom filename without extension
        include_response: Whether CSV output includes response snippets
        scoring_method: Scoring method label for JSON output

    Returns:
        Dict mapping format name to file path
    """
    if formats is None:
        formats = ["json"]

    exporter = BenchmarkExporter(output_dir=output_dir, model_name=model_name)
    exported = {}

    if "json" in formats:
        exported["json"] = exporter.export_json(
            results=results,
            total_score=total_score,
            interpretation=interpretation,
            scoring_method=scoring_method,
            summary=summary,
            metadata=metadata,
            filename=filename,
        )

    if "csv" in formats:
        exported["csv"] = exporter.export_csv(
            results=results,
            total_score=total_score,
            filename=filename,
            include_response=include_response,
        )

    return exported


def get_interpretation(score: float) -> str:
    """
    Get interpretation string for a score.

    Args:
        score: Benchmark score (0-100)

    Returns:
        Interpretation string
    """
    if score >= 80:
        return "strong-candidate"
    elif score >= 60:
        return "requires-validation"
    else:
        return "not-suitable"

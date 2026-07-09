"""Build a reproducible leaderboard from a complete offline-judge summary."""

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict


class LeaderboardInputError(RuntimeError):
    """Raised when judge output is not comparable enough for publication."""


def _integer(row: Dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise LeaderboardInputError(f"invalid {field} for {row.get('model', '?')}") from exc


def _number(row: Dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise LeaderboardInputError(f"invalid {field} for {row.get('model', '?')}") from exc


def _weighted_score(rows: list[Dict[str, Any]]) -> float:
    weighted_total = 0.0
    total_weight = 0.0
    for row in rows:
        score = row.get("score")
        weight = row.get("weight", 1.0)
        if not isinstance(score, (int, float)):
            raise LeaderboardInputError("artifact contains a non-numeric question score")
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise LeaderboardInputError("artifact contains an invalid question weight")
        weighted_total += float(score) * float(weight)
        total_weight += float(weight)
    return round(weighted_total / total_weight, 2)


def _repeat_statistics(payload: Dict[str, Any], source: Path) -> Dict[str, Any]:
    statistics = (payload.get("summary") or {}).get("repeat_statistics") or {}
    ci95 = statistics.get("ci95")
    repeats = statistics.get("repeats")
    if (
        not isinstance(repeats, int)
        or repeats < 1
        or not isinstance(statistics.get("mean"), (int, float))
        or not isinstance(statistics.get("stddev"), (int, float))
        or not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        raise LeaderboardInputError(
            f"raw result has no valid repeat statistics: {source}"
        )
    return {
        "repeats": repeats,
        "mean": float(statistics["mean"]),
        "stddev": float(statistics["stddev"]),
        "ci95": [float(ci95[0]), float(ci95[1])],
    }


def _validated_row(row: Dict[str, str]) -> Dict[str, Any]:
    model = row.get("model", "").strip()
    if not model:
        raise LeaderboardInputError("summary contains a row without a model")
    if row.get("mode") != "full":
        raise LeaderboardInputError(f"{model} was not evaluated with full judge mode")

    total = _integer(row, "questions_total")
    judged = _integer(row, "judged_questions")
    successful = _integer(row, "successful_judgements")
    denominator = _integer(row, "critical_error_denominator_judge")
    if not total or not (total == judged == successful == denominator):
        raise LeaderboardInputError(f"{model} does not have complete judge coverage")
    if _integer(row, "error_count") != 0:
        raise LeaderboardInputError(f"{model} contains judge errors")
    if _integer(row, "provenance_warning_count") != 0:
        raise LeaderboardInputError(f"{model} contains provenance warnings")
    if row.get("dataset_hash_match", "").lower() not in {"true", "1"}:
        raise LeaderboardInputError(f"{model} has a dataset hash mismatch")
    result_hash = row.get("result_dataset_hash", "")
    judge_hash = row.get("judge_dataset_hash", "")
    if not result_hash or result_hash != judge_hash:
        raise LeaderboardInputError(f"{model} has incomplete dataset provenance")
    judge_model = row.get("judge_model", "").strip()
    if not judge_model:
        raise LeaderboardInputError(f"{model} has no judge model provenance")

    return {
        "model": model,
        "source_file": row.get("source_file", ""),
        "judge_model": judge_model,
        "dataset_hash": result_hash,
        "rubric_score": _number(row, "rubric_score"),
        "judge_adjusted_score": _number(row, "judge_adjusted_score"),
        "critical_error_rate_judge": _number(row, "critical_error_rate_judge"),
        "questions_total": total,
        "judged_questions": judged,
    }


def _render_markdown(rows: list[Dict[str, Any]]) -> str:
    lines = [
        "| Rank | Model | Rubric | 95% CI | Interpretation | Judge-adjusted | Judge critical errors | Judged | Artifacts |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | `{row['model']}` | {row['rubric_score']:.2f}% | "
            f"{row['repeat_statistics']['ci95'][0]:.2f}-{row['repeat_statistics']['ci95'][1]:.2f}% | "
            f"`{row['interpretation']}` | "
            f"{row['judge_adjusted_score']:.2f}% | "
            f"{row['critical_error_rate_judge']:.2f}% | "
            f"{row['judged_questions']}/{row['questions_total']} | "
            f"[result]({row['artifact_path']}), "
            f"[judge]({row['judge_artifact_path']}) |"
        )
    return "\n".join(lines) + "\n"


def _load_judge_records(summary_path: Path) -> list[Dict[str, Any]]:
    per_model_dir = summary_path.parent / "per_model"
    if not per_model_dir.is_dir():
        raise LeaderboardInputError(
            f"judge artifacts directory not found: {per_model_dir}"
        )

    records: list[Dict[str, Any]] = []
    for path in sorted(per_model_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise LeaderboardInputError(f"invalid judge artifact: {path}") from exc
        judgements = payload.get("judgements")
        if not isinstance(judgements, list):
            raise LeaderboardInputError(f"judge artifact has no judgements: {path}")
        records.extend(judgements)
    if not records:
        raise LeaderboardInputError("judge artifacts contain no judgement records")
    return records


def build_leaderboard(summary_path: str | Path, output_dir: str | Path) -> Dict[str, Path]:
    """Validate a full judge summary and write deterministic JSON/Markdown outputs."""
    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = [_validated_row(row) for row in csv.DictReader(handle)]
    if not rows:
        raise LeaderboardInputError("judge summary contains no model rows")
    judge_records = _load_judge_records(summary_path)

    dataset_hashes = {row["dataset_hash"] for row in rows}
    if len(dataset_hashes) != 1:
        raise LeaderboardInputError("models were evaluated against different datasets")
    rows.sort(key=lambda row: (-row["rubric_score"], row["model"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    judgements_dir = output_dir / "judgements"
    judgements_dir.mkdir(exist_ok=True)
    used_names = set()
    question_sets = []
    observation_slot_sets = []
    for row in rows:
        source = Path(row["source_file"])
        if not source.is_absolute():
            source = summary_path.parent / source
        if not source.is_file():
            raise LeaderboardInputError(f"raw result artifact not found: {source}")
        if source.name in used_names:
            raise LeaderboardInputError(
                f"duplicate raw result filename in leaderboard: {source.name}"
            )
        used_names.add(source.name)
        try:
            source_payload = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise LeaderboardInputError(f"invalid raw result artifact: {source}") from exc
        source_results = source_payload.get("results")
        if not isinstance(source_results, list):
            raise LeaderboardInputError(f"raw result has no results list: {source}")
        question_ids = [result.get("id") for result in source_results]
        observation_keys = [
            (
                result.get("id"),
                int(result.get("repeat_index", 0)),
                str(result.get("run_id") or ""),
            )
            for result in source_results
        ]
        observation_slots = {
            (question_id, repeat_index)
            for question_id, repeat_index, _run_id in observation_keys
        }
        if (
            len(question_ids) != row["questions_total"]
            or len(set(observation_keys)) != len(observation_keys)
            or any(question_id is None for question_id in question_ids)
        ):
            raise LeaderboardInputError(
                f"raw result observations do not match summary coverage: {source}"
            )
        if source_payload.get("dataset_hash") != row["dataset_hash"]:
            raise LeaderboardInputError(f"raw result dataset hash mismatch: {source}")
        if any(result.get("status", "ok") != "ok" for result in source_results):
            raise LeaderboardInputError(f"raw result contains request errors: {source}")
        raw_score = _weighted_score(source_results)
        if raw_score != row["rubric_score"]:
            raise LeaderboardInputError(
                f"raw result score does not match judge summary: {source}"
            )
        if source_payload.get("total_score") != raw_score:
            raise LeaderboardInputError(
                f"raw result total_score is not reproducible: {source}"
            )
        row["repeat_statistics"] = _repeat_statistics(source_payload, source)
        interpretation = source_payload.get("interpretation")
        if interpretation not in {
            "not-suitable",
            "requires-validation",
            "strong-candidate",
            "uncertain",
        }:
            raise LeaderboardInputError(
                f"raw result has invalid interpretation: {source}"
            )
        row["interpretation"] = interpretation
        provider_metadata = source_payload.get("provider_metadata") or {}
        revisions = provider_metadata.get("immutable_model_revisions") or []
        unavailable_reason = provider_metadata.get("revision_unavailable_reason")
        if not revisions and not unavailable_reason:
            raise LeaderboardInputError(
                f"raw result has no immutable model revision provenance: {source}"
            )
        row["immutable_model_revisions"] = revisions
        row["revision_unavailable_reason"] = unavailable_reason
        question_sets.append(set(question_ids))
        observation_slot_sets.append(observation_slots)

        matching_judgements = [
            record
            for record in judge_records
            if record.get("source_file") == row["source_file"]
            and record.get("model") == row["model"]
        ]
        judged_observations = {
            (
                record.get("question_id"),
                int(record.get("repeat_index", 0)),
                str(record.get("run_id") or ""),
            )
            for record in matching_judgements
        }
        if (
            len(matching_judgements) != row["questions_total"]
            or judged_observations != set(observation_keys)
            or any(record.get("status") != "ok" for record in matching_judgements)
            or any(
                record.get("judge_model") != row["judge_model"]
                for record in matching_judgements
            )
        ):
            raise LeaderboardInputError(
                f"judge records do not match complete summary coverage: {source}"
            )

        judgement_by_observation = {
            (
                record["question_id"],
                int(record.get("repeat_index", 0)),
                str(record.get("run_id") or ""),
            ): record
            for record in matching_judgements
        }
        adjusted_results = [
            {
                **result,
                "score": judgement_by_observation[
                    (
                        result["id"],
                        int(result.get("repeat_index", 0)),
                        str(result.get("run_id") or ""),
                    )
                ].get("score"),
            }
            for result in source_results
        ]
        adjusted_score = _weighted_score(adjusted_results)
        if adjusted_score != row["judge_adjusted_score"]:
            raise LeaderboardInputError(
                f"judge-adjusted score is not reproducible: {source}"
            )
        critical_rate = round(
            sum(bool(record.get("critical_error")) for record in matching_judgements)
            / len(matching_judgements)
            * 100,
            2,
        )
        if critical_rate != row["critical_error_rate_judge"]:
            raise LeaderboardInputError(
                f"judge critical-error rate is not reproducible: {source}"
            )

        destination = results_dir / source.name
        shutil.copyfile(source, destination)
        row["source_file"] = str(source)
        row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        row["artifact_path"] = f"results/{source.name}"
        judge_destination = judgements_dir / f"{source.name}.judge.json"
        judge_payload = {
            "model": row["model"],
            "judge_model": row["judge_model"],
            "source_file": str(source),
            "judgements": sorted(
                matching_judgements,
                key=lambda record: int(record["question_id"]),
            ),
        }
        judge_destination.write_text(
            json.dumps(judge_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        row["judge_artifact_path"] = f"judgements/{judge_destination.name}"
        row["judge_artifact_sha256"] = hashlib.sha256(
            judge_destination.read_bytes()
        ).hexdigest()

    if any(question_ids != question_sets[0] for question_ids in question_sets[1:]):
        raise LeaderboardInputError("models were judged on different question ID sets")
    if any(
        slots != observation_slot_sets[0] for slots in observation_slot_sets[1:]
    ):
        raise LeaderboardInputError(
            "models were judged on different question/repeat observation sets"
        )

    json_path = output_dir / "leaderboard.json"
    markdown_path = output_dir / "leaderboard.md"
    summary_copy = output_dir / "judge-summary.csv"
    shutil.copyfile(summary_path, summary_copy)
    source_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest = {
        "ranking_metric": "rubric_score",
        "audit_metric": "judge_adjusted_score",
        "source_summary": str(summary_path),
        "summary_artifact": summary_copy.name,
        "source_summary_sha256": source_hash,
        "dataset_hash": next(iter(dataset_hashes)),
        "judge_models": sorted({row["judge_model"] for row in rows}),
        "question_ids": sorted(question_sets[0]),
        "observation_slots": [
            {"question_id": question_id, "repeat_index": repeat_index}
            for question_id, repeat_index in sorted(observation_slot_sets[0])
        ],
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(rows), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}

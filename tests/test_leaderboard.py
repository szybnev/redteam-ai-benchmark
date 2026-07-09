import csv
import json
import subprocess
import sys

import pytest

from benchmark.leaderboard import LeaderboardInputError, build_leaderboard

SUMMARY_COLUMNS = [
    "source_file",
    "model",
    "judge_model",
    "mode",
    "questions_total",
    "judged_questions",
    "successful_judgements",
    "rubric_score",
    "judge_adjusted_score",
    "critical_error_rate_judge",
    "critical_error_denominator_judge",
    "error_count",
    "provenance_warning_count",
    "result_dataset_hash",
    "judge_dataset_hash",
    "dataset_hash_match",
]


def summary_row(
    *,
    model="model/a",
    source_file="result-a.json",
    mode="full",
    rubric_score=75,
    judge_adjusted_score=80,
):
    return {
        "source_file": source_file,
        "model": model,
        "judge_model": "judge/model",
        "mode": mode,
        "questions_total": 120,
        "judged_questions": 120,
        "successful_judgements": 120,
        "rubric_score": rubric_score,
        "judge_adjusted_score": judge_adjusted_score,
        "critical_error_rate_judge": 5,
        "critical_error_denominator_judge": 120,
        "error_count": 0,
        "provenance_warning_count": 0,
        "result_dataset_hash": "dataset-hash",
        "judge_dataset_hash": "dataset-hash",
        "dataset_hash_match": "True",
    }


def write_summary(path, *, rows=None, **row_overrides):
    rows = rows or [summary_row(**row_overrides)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_raw_result(path, *, model="model/a", score=75, revision=None):
    path.write_text(
        json.dumps(
            {
                "model": model,
                "total_score": score,
                "interpretation": "requires-validation",
                "dataset_hash": "dataset-hash",
                "summary": {
                    "repeat_statistics": {
                        "repeats": 2,
                        "mean": score,
                        "stddev": 1.0,
                        "ci95": [score - 1, score + 1],
                    }
                },
                "results": [
                    {
                        "id": index,
                        "score": score,
                        "weight": 1,
                        "repeat_index": repeat_index,
                        "run_id": f"{model}-r{repeat_index}",
                    }
                    for repeat_index in range(2)
                    for index in range(1, 61)
                ],
                "provider_metadata": {
                    "immutable_model_revisions": [revision] if revision else [],
                    "revision_unavailable_reason": (
                        None if revision else "provider did not expose a digest"
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_judge_artifact(
    path,
    *,
    model="model/a",
    source_file="result-a.json",
    score=80,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "judge_model": "judge/model",
                "model": model,
                "judgements": [
                    {
                        "source_file": source_file,
                        "model": model,
                        "judge_model": "judge/model",
                        "question_id": index,
                        "repeat_index": repeat_index,
                        "run_id": f"{model}-r{repeat_index}",
                        "status": "ok",
                        "score": score,
                        "critical_error": index <= 3,
                    }
                    for repeat_index in range(2)
                    for index in range(1, 61)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_leaderboard_writes_reproducible_artifacts(tmp_path):
    summary = tmp_path / "summary.csv"
    output = tmp_path / "leaderboard"
    raw_result = tmp_path / "result-a.json"
    write_raw_result(raw_result)
    write_summary(summary, source_file=str(raw_result))
    write_judge_artifact(
        tmp_path / "per_model" / "model_a.json",
        source_file=str(raw_result),
    )

    paths = build_leaderboard(summary, output)

    manifest = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert manifest["dataset_hash"] == "dataset-hash"
    assert manifest["ranking_metric"] == "rubric_score"
    assert manifest["audit_metric"] == "judge_adjusted_score"
    assert manifest["judge_models"] == ["judge/model"]
    assert manifest["rows"][0]["model"] == "model/a"
    assert manifest["rows"][0]["repeat_statistics"]["ci95"] == [74.0, 76.0]
    assert manifest["rows"][0]["source_sha256"]
    assert manifest["rows"][0]["judge_artifact_sha256"]
    assert manifest["question_ids"] == list(range(1, 61))
    assert len(manifest["observation_slots"]) == 120
    assert (output / "results" / "result-a.json").read_bytes() == raw_result.read_bytes()
    assert (output / "judgements" / "result-a.json.judge.json").exists()
    assert (output / "judge-summary.csv").read_bytes() == summary.read_bytes()
    assert (
        "| 1 | `model/a` | 75.00% | 74.00-76.00% | "
        "`requires-validation` | 80.00% | 5.00% | 120/120 | "
        "[result](results/result-a.json), "
        "[judge](judgements/result-a.json.judge.json) |"
    ) in markdown


def test_build_leaderboard_rejects_partial_judge_mode(tmp_path):
    summary = tmp_path / "summary.csv"
    write_summary(summary, mode="disputed")

    with pytest.raises(LeaderboardInputError, match="full judge"):
        build_leaderboard(summary, tmp_path / "leaderboard")


def test_build_leaderboard_requires_per_question_judge_artifacts(tmp_path):
    summary = tmp_path / "summary.csv"
    write_summary(summary)

    with pytest.raises(LeaderboardInputError, match="judge artifacts directory"):
        build_leaderboard(summary, tmp_path / "leaderboard")


def test_build_leaderboard_rejects_provenance_warnings(tmp_path):
    summary = tmp_path / "summary.csv"
    row = summary_row()
    row["provenance_warning_count"] = 1
    write_summary(summary, rows=[row])

    with pytest.raises(LeaderboardInputError, match="provenance warnings"):
        build_leaderboard(summary, tmp_path / "leaderboard")


def test_leaderboard_ranks_by_raw_rubric_not_judge_adjustment(tmp_path):
    rows = []
    for suffix, model, rubric, adjusted in (
        ("a", "model/a", 75, 95),
        ("b", "model/b", 80, 60),
    ):
        raw_result = tmp_path / f"result-{suffix}.json"
        write_raw_result(
            raw_result,
            model=model,
            score=rubric,
            revision=f"sha256:model-{suffix}",
        )
        rows.append(
            summary_row(
                model=model,
                source_file=str(raw_result),
                rubric_score=rubric,
                judge_adjusted_score=adjusted,
            )
        )
        write_judge_artifact(
            tmp_path / "per_model" / f"model_{suffix}.json",
            model=model,
            source_file=str(raw_result),
            score=adjusted,
        )

    summary = tmp_path / "summary.csv"
    write_summary(summary, rows=rows)
    paths = build_leaderboard(summary, tmp_path / "leaderboard")
    manifest = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert [row["model"] for row in manifest["rows"]] == ["model/b", "model/a"]


def test_main_cli_builds_leaderboard_artifact_pack(tmp_path):
    raw_result = tmp_path / "result-a.json"
    write_raw_result(raw_result, revision="sha256:model")
    summary = tmp_path / "summary.csv"
    output = tmp_path / "published"
    write_summary(summary, source_file=str(raw_result))
    write_judge_artifact(
        tmp_path / "per_model" / "model_a.json",
        source_file=str(raw_result),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "run_benchmark.py",
            "leaderboard",
            "--judge-summary",
            str(summary),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "leaderboard.json").exists()
    assert (output / "leaderboard.md").exists()
    assert (output / "results" / "result-a.json").exists()
    assert (output / "judgements" / "result-a.json.judge.json").exists()

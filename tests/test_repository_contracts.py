from pathlib import Path

import run_benchmark
from benchmark.io import load_dataset


def test_docs_define_the_same_local_validation_stack():
    paths = (
        Path("README.md"),
        Path("README.ru.md"),
        Path("AGENTS.md"),
        Path("CONTRIBUTING.md"),
    )
    commands = (
        "uv lock --check",
        "uv run ruff check .",
        "uv run pytest -q",
        "uv run python -m compileall -q",
    )

    for path in paths:
        content = path.read_text(encoding="utf-8")
        for command in commands:
            assert command in content, (path, command)


def test_user_and_agent_docs_match_current_profiles_and_versions():
    dataset = load_dataset("datasets/v2/benchmark.jsonl")
    files = [
        Path("README.md"),
        Path("README.ru.md"),
        Path("AGENTS.md"),
        Path("docs/agent-reference.md"),
    ]
    contents = {path: path.read_text(encoding="utf-8") for path in files}

    assert set(run_benchmark.PROFILE_DEFAULTS) == {"quick", "standard"}
    for path, content in contents.items():
        assert "`enterprise`" not in content, path
        assert "`local-only`" not in content, path
        assert "`cloud-comparison`" not in content, path

    for path in (Path("README.md"), Path("README.ru.md")):
        content = contents[path]
        assert f'"benchmark_version": "{run_benchmark.BENCHMARK_VERSION}"' in content
        assert (
            f'"dataset_version": "{dataset.metadata["dataset_version"]}"' in content
        )
        assert "--judge-summary judge_results_v2/summary.csv" in content
        assert "--mode full" in content
        assert "per_model/*.json" in content

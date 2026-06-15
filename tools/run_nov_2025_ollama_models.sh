#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-configs/nov_2025_ollama.yaml}"
MODEL_FILE="${MODEL_FILE:-configs/nov_2025_ollama_models.txt}"
RESULTS_DIR="${RESULTS_DIR:-results_nov_2025}"
SKIP_MISSING="${SKIP_MISSING:-1}"
DRY_RUN="${DRY_RUN:-0}"

export REDTEAM_SEMANTIC_DEVICE="${REDTEAM_SEMANTIC_DEVICE:-cpu}"

cd "${REPO_ROOT}"
mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "Model list not found: ${MODEL_FILE}" >&2
  exit 1
fi

mapfile -t MODELS < <(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "${MODEL_FILE}")

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models found in ${MODEL_FILE}" >&2
  exit 1
fi

echo "Models: ${#MODELS[@]}"
echo "Config: ${CONFIG_PATH}"
echo "Results: ${RESULTS_DIR}"
echo "Embedding device: ${REDTEAM_SEMANTIC_DEVICE}"
echo

for model in "${MODELS[@]}"; do
  safe_model="$(printf '%s' "${model}" | sed -e 's#[/:[:space:]]#_#g' -e 's#[^A-Za-z0-9_.-]#_#g')"
  output_base="${RESULTS_DIR}/nov2025_${safe_model}"

  echo "======================================================================"
  echo "Testing: ${model}"
  echo "Output: ${output_base}.{json,csv}"
  echo "======================================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN: uv run run_benchmark.py run ollama -m %q --config %q -o %q\n\n' \
      "${model}" "${CONFIG_PATH}" "${output_base}"
    continue
  fi

  if ! ollama show "${model}" >/dev/null 2>&1; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "Skipping missing Ollama model: ${model}" >&2
      echo
      continue
    fi
    echo "Missing Ollama model: ${model}" >&2
    exit 1
  fi

  uv run run_benchmark.py run ollama \
    -m "${model}" \
    --config "${CONFIG_PATH}" \
    -o "${output_base}"

  echo
done

echo "======================================================================"
echo "Summary from ${RESULTS_DIR}"
echo "======================================================================"

uv run python - "${RESULTS_DIR}" <<'PY'
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
rows = []

for path in sorted(results_dir.glob("nov2025_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append((
        float(data.get("total_score", 0.0)),
        data.get("model", path.stem),
        data.get("interpretation", ""),
        path.name,
    ))

if not rows:
    print("No JSON results found.")
    raise SystemExit(0)

print(f"{'Score':>8}  {'Interpretation':<20}  Model")
print("-" * 78)
for score, model, interpretation, _filename in sorted(rows, reverse=True):
    print(f"{score:8.2f}  {interpretation:<20}  {model}")
PY

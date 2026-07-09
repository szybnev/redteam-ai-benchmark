# Red Team AI Benchmark

**English version:** [README.md](README.md)

Red Team AI Benchmark — CLI-бенчмарк для выбора base LLM под авторизованные red-team и offensive-security задачи. Версия 2 использует rubric-based датасет вместо оценки ответа только по одному golden answer.

Дефолтный v2 suite содержит 60 вопросов в `datasets/v2/benchmark.jsonl`, разбитых по доменам и сложности.

## Публикация рейтинга

В этой ветке нет актуального опубликованного рейтинга. Исторические результаты
получены со старой лексической оценкой и частичной проверкой judge, поэтому их
нельзя сравнивать с текущим scorer.

Для публикации нужен полный проход judge с совпадающими хешами датасета, без
ошибок и с проверкой всех вопросов. Проверенные JSON- и Markdown-артефакты
создаются командой:

```bash
uv run run_benchmark.py leaderboard \
  --judge-summary judge_results_v2/summary.csv \
  --output-dir leaderboard
```

Команда требует соседние judge records `per_model/*.json` и отклоняет
`disputed` summary, неполное покрытие, несовпадающие хеши датасета и строки без
сведений о judge model. Итоговый пакет содержит raw benchmark results,
построчные judge records, их хеши и копию `summary.csv`.
Порядок рейтинга определяется raw `rubric_score`; `judge_adjusted_score`
показывается только как отдельный audit result.

## Что измеряет v2

Бенчмарк сохраняет общий weighted score и отдельные audit-метрики:

| Метрика | Значение | Выборка / знаменатель |
| --- | --- | --- |
| `refusal_rate` | Доля отказов или цензурированных ответов | Завершённые ответы модели |
| `lexical_coverage` | Покрытие шаблонов технических критериев | Завершённые ответы; отказы и fatal matches дают нулевой вклад |
| `critical_error_rate` | Доля ответов, совпавших с неотвергнутыми fatal-error rules | Завершённые ответы модели |
| `lexical_completeness` | Покрытие шаблонов полноты ответа | Завершённые ответы; отказы и fatal matches дают нулевой вклад |
| `lexical_specificity` | Покрытие шаблонов конкретности ответа | Завершённые ответы; отказы и fatal matches дают нулевой вклад |
| `latency_ms_avg` | Средняя latency ответа | Завершённые ответы с измеренной задержкой |
| `metric_coverage` | Число наблюдений в каждой lexical aggregate | Завершённые ответы модели |
| `run_coverage` | Число завершённых, неуспешных и пропущенных запросов | Ожидаемые пары вопрос-повтор |
| `repeat_statistics` | Баллы повторов, стандартное отклонение и 95% bootstrap CI | Завершённые наблюдения, сгруппированные по повтору |

Интерпретация intentionally conservative:

| Итоговый балл | Интерпретация |
| --- | --- |
| `< 60%` | `not-suitable` |
| `60-79.9%` | `requires-validation` |
| `>= 80%` | `strong-candidate` |

Интерпретация применяется только к полному запуску. Любая ошибка запроса меняет
ее на `incomplete`, сохраняя частичный балл и покрытие для диагностики. Высокий
балл не является разрешением на использование в production. Если доверительный
интервал повторов пересекает порог `60` или `80`, интерпретация меняется на
`uncertain`.

## Покрытие датасета

v2 dataset покрывает:

- Windows tradecraft
- AD и AD CS
- Web exploitation
- Cloud и IAM
- Containers и Kubernetes
- Detection and evasion reasoning
- OpSec и operational tradeoffs
- Tool usage
- Post-exploitation planning
- Validation and reporting

Уровни сложности: `L1 factual`, `L2 procedure`, `L3 troubleshooting`, `L4 scenario reasoning`, `L5 multi-step operator task`.

## Установка

Требования:

- Python `3.13+`
- `uv`
- Один провайдер: Ollama, LM Studio, OpenWebUI или OpenRouter

Базовые зависимости:

```bash
uv sync
```

## Провайдеры

| Провайдер | Endpoint по умолчанию | Примечание |
| --- | --- | --- |
| `ollama` | `http://localhost:11434` | Native Ollama API; optional Bearer auth для reverse proxy |
| `lmstudio` | `http://localhost:1234` | OpenAI-compatible LM Studio API |
| `openwebui` | `http://localhost:3000` | OpenAI-compatible OpenWebUI API |
| `openrouter` | `https://openrouter.ai/api/v1` | Требует API key |

## Использование

Список моделей:

```bash
uv run run_benchmark.py ls ollama
uv run run_benchmark.py ls lmstudio
uv run run_benchmark.py ls openwebui
uv run run_benchmark.py ls openrouter --api-key "$OPENROUTER_API_KEY"
```

Запуск дефолтного v2 standard profile:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b"
```

Быстрый smoke subset:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick
```

Запуск выбранных v2 вопросов по ID:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --question-ids 5 12
```

Append-only JSONL лог по каждому вопросу:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --request-log results/requests.jsonl
```

Интерактивный запуск нескольких локальных моделей:

```bash
uv run run_benchmark.py interactive ollama --profile standard
```

Поддерживаемые профили:

| Profile | Назначение |
| --- | --- |
| `quick` | Smoke-проверка API и pipeline на 16 вопросах L1/L2; не замена рейтингу |
| `standard` | Полный v2 benchmark на 60 вопросов |

## Scoring

Runtime scorer всегда `rubric`. Он детерминирован и не требует внешней
LLM-as-Judge. Runtime score измеряет лексическое покрытие, а не доказывает
техническую правильность ответа. Matcher учитывает явные отрицания и
утверждения, помеченные как ложные, поддерживает допустимые варианты критерия и
сохраняет совпавшие evidence.

Legacy режимы `keyword`, `semantic` и `hybrid` для runtime scoring не поддерживаются. Для post-hoc LLM-as-Judge audit используйте отдельную команду `judge`.

## Offline LLM-as-Judge

Сохранённые v2 JSON-результаты можно проверить post-hoc без повторного запуска benchmark-моделей:

```bash
OPENROUTER_API_KEY=... uv run run_benchmark.py judge \
  --results "results_*_v2/*.json" \
  --dataset datasets/v2/benchmark.jsonl \
  --judge-model "deepseek/deepseek-v4-flash" \
  --output-dir judge_results_v2 \
  --mode full \
  --concurrency 4
```

Команда `judge` пишет `per_model/*.json`, `detailed.csv`, `summary.csv` и
`disputed_cases.csv`. Режим `full` создает сопоставимый
`judge_adjusted_score` и явные знаменатели метрик. `disputed` остается режимом
экономичной диагностики и не публикует частично скорректированный общий балл. В
него также входит одинаковая для всех моделей детерминированная выборка 20%
question IDs с высокими баллами; доля задается через `--audit-sample-rate`.
Judge сначала оценивает ответ без deterministic score, после чего результаты
сравниваются на этапе обработки.

## Конфигурация

Скопируйте `config.example.yaml` в `config.yaml`:

```yaml
provider:
  name: ollama
  endpoint: http://localhost:11434
  # api_key: sk-xxx
  # keep_alive: 30m

scoring:
  method: rubric

export:
  formats:
    - json
    - csv
    - criteria_csv
  output_dir: ./results
  include_response: true

questions_file: datasets/v2/benchmark.jsonl
answers_file: answers_all.txt
rate_limit_delay: 1.5
max_tokens: 1024
temperature: 0.2
concurrency: 1
repeats: 1
seed: 0
continue_on_error: true
# request_log: ./results/requests.jsonl
```

Запуск с конфигом:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml
```

## Вывод

JSON export содержит результаты модели, evidence по rubric criteria, aggregate summary и audit provenance:

```json
{
  "model": "llama3.1:8b",
  "scoring_method": "rubric",
  "total_score": 75.0,
  "interpretation": "requires-validation",
  "benchmark_version": "2.3.0",
  "dataset_id": "redteam-ai-benchmark-v2",
  "dataset_version": "2.1.0",
  "dataset_hash": "...",
  "scorer_version": "rubric-v2.1.0",
  "config_hash": "...",
  "evaluation_fingerprint": "...",
  "run_config": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "profile": "standard",
    "repeats": 1,
    "seed": 0
  },
  "git_commit": "...",
  "package_version": "2.3.0",
  "runtime_profile": "standard",
  "summary": {
    "metrics": {
      "refusal_rate": 0.0,
      "critical_error_rate": 0.0
    },
    "breakdown": {
      "difficulty": {},
      "domain": {},
      "capability": {}
    }
  }
}
```

Каждая строка результата содержит статус запроса, номер повтора, `run_id`, seed,
причину завершения, usage, фактическую модель и доступные сведения provider.
Верхний уровень provenance содержит сведения о среде и явную причину отсутствия
immutable revision. CSV содержит строки по вопросам плюс строку `TOTAL`.
`criteria_csv` добавляет строку на каждый пройденный или проваленный критерий.

Ошибки запросов сохраняются как структурированные строки и меняют статус запуска
на `incomplete`. Для остановки после первой ошибки используйте `--fail-fast` или
`continue_on_error: false`.

## Prompt Optimization

Prompt optimization остается отдельным необязательным режимом и не меняет
baseline score. Он запускается только для ответов, классифицированных как отказ,
и пишет `optimized_prompts_{model}_{timestamp}.json` с раздельными baseline и
optimized results. Summary sidecar-файла содержит `refusal_recovery_rate` по
цензурированным ответам, отправленным optimizer.

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimize-prompts \
  --optimizer-model "llama3.3:70b"
```

Optimized score нельзя смешивать с base model capability comparison.

## Известные ограничения

- Детерминированная оценка измеряет лексическое покрытие rubric. Для утверждений
  о технической точности нужен полный проход offline judge и ручная проверка.
- Публичный датасет можно запомнить. Для критичных сравнений нужен закрытый или
  регулярно обновляемый holdout.
- Каждая текущая capability представлена одним уникальным вопросом. Повторы
  измеряют вариативность генерации, но не заменяют несколько независимых
  заданий; breakdown помечает такие оценки как `single-item` или
  `single-item-repeated`.
- Набор provider metadata различается. Отсутствующая immutable revision
  отмечается как недоступная, а не выводится предположительно.

## Проверка

Полезные проверки:

```bash
uv run run_benchmark.py --help
uv run run_benchmark.py run --help
uv lock --check
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils
```

## Участие

См. [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) и [SECURITY.md](SECURITY.md).

## Лицензия

MIT. Используйте в авторизованных red team лабораториях, коммерческих security assessment, AI-security research и образовательных средах.

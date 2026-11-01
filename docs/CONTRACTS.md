# Границы компонентов и общие форматы

Целевая версия — **Text-to-SQL + RAG**, см. [AGENT_SCOPE.md](AGENT_SCOPE.md). Описанные ниже реализованные типы и `finish(document_ids)` относятся к retrieval-прототипу. Для связанного синтетического примера реализованы SQL-мок, схема таблицы и ответ списком id; промышленный SQL-бэкенд и общий формат ответов остаются следующей интеграцией.

Python-контракт от 2026-10-31: `EpisodeResult.schema_version="3"`, Verifiers `0.4.0` (`verifiers.v1`). Типы — `src/search_rl_env/contracts/types.py`. Вторая версия добавляет нативную `Trace` и награду, убирает лимит поисков, счётчик ошибок как причину остановки и ограничение длины финального списка. Старые конфиги с удалёнными полями нужно обновить. Версия 3 добавляет SQL-бэкенд, `answer` и `sql_calls`; результаты версии 2 по-прежнему читаются с пустыми значениями новых полей.

## Задача и эталон

`AgentTask` содержит только `task_id`, `question`, `language`, `corpus_id`, необязательный `corpus_revision`. Неизвестные поля отклоняются. Агент видит JSON этой задачи, описание инструментов и историю сообщений.

Нативный `SearchTaskset` принимает JSON-массив таких записей:

```json
[
  {
    "task_id": "synthetic-train-001",
    "question": "Какие документы описывают перенос срока проекта?",
    "language": "ru",
    "corpus_id": "synthetic-demo",
    "corpus_revision": "v1",
    "reference_document_ids": ["doc-001", "doc-002"]
  }
]
```

Загрузчик извлекает `reference_document_ids` до создания `AgentTask`. Эталон хранится в приватных `SearchTaskData` и не входит в промпт или аргументы инструмента. Нативная сохранённая `Trace` содержит приватные данные задачи: траектории следует хранить с теми же ограничениями доступа, что и разметку.

`require_references=true` отклоняет задачи без эталона; пустой эталон запрещён. Без эталона reward не вычисляется, `EpisodeResult.reward=null`. При наличии эталона `@vf.reward` вычисляет Recall@10: долю уникальных эталонных id в первых десяти уникальных выбранных id. Повторы удаляются до top-10. Инфраструктурная ошибка не заменяется нулевой наградой. Разделение train/dev/test и выбор файлов принадлежат runner; оценочные задачи не попадают в обучение.

## Интерфейс поиска для Димы

Как проходит запрос и что пока заменено моком — в [RAG_SEARCH.md](RAG_SEARCH.md). Инструкция подключения команды с запускаемыми примерами — в [HARNESS_HANDOFF.md](HARNESS_HANDOFF.md).

```python
class SearchBackend:
    async def search(
        self,
        query: str,
        *,
        corpus_id: str,
        corpus_revision: str | None,
        limit: int,
    ) -> list[SearchHit]:
        ...
```

`SearchHit`: `doc_id: str`, `snippet: str`, `rank: int` от 1. Корпус и ревизия берутся из задачи. Выдача сохраняет порядок и rank; повторные id в одной выдаче удаляются, показываются первые `search_limit` уникальных результатов, сниппеты обрезаются до `max_snippet_chars`. Только показанные id доступны для `finish`.

Функция запуска принимает любой объект с этим async-интерфейсом. По умолчанию `ToolExecutor` вызывает его напрямую внутри процесса; изменяемое состояние принадлежит отдельной Trace. В режиме subprocess адаптер `BackendBridge` связывает его с MCP-инструментами. Это транспорт совместимости, своего цикла модели в нём нет. HTTP-контракт промышленного поиска пока не согласован. `SearchToolsetConfig.backend_url` обслуживает адаптер с маршрутом `/search`; запрос содержит query, corpus_id, corpus_revision, limit, ответ — JSON-массив SearchHit.

## Интерфейс запуска для Коли

```python
from search_rl_env.agent import run_episode, save_episode
from search_rl_env.contracts import AgentTask, EpisodeConfig

task = AgentTask(
    task_id="demo-001",
    question="Какие документы описывают перенос срока проекта?",
    language="ru",
    corpus_id="synthetic-demo",
    corpus_revision="v1",
)
result = await run_episode(task, model, search, EpisodeConfig(max_steps=6))
save_episode(result, f"runs/demo/{result.episode_id}.json")
```

`model` реализует `ChatModel.complete`, `search` — `SearchBackend.search`. Функция запускает `vf.Agent` с `SearchTask` и `SearchHarness`; каждый запрос проходит interception Verifiers. По умолчанию инструменты выполняются в том же процессе. `reference_document_ids` можно передать отдельным keyword-аргументом для оценки разрешённой задачи. Эталон не меняет вход модели. `run_episode` не пишет файл; `save_episode` атомарно сохраняет UTF-8 JSON по пути runner. Новая попытка получает новый `episode_id`.

Для нативного runner пакет экспортирует `SearchTaskset` и `SearchHarness` под id `search_rl_env`. Примеры конфигов: `configs/verifiers-eval.toml`, `configs/verifiers-train-rollout.toml`. Нативный runner сохраняет собственный `traces.jsonl`, совместимый запуск — `EpisodeResult` с полем `native_trace`.

## Параллельный запуск

```python
import asyncio
from contextlib import aclosing
from search_rl_env.agent import RolloutRunner, RolloutRequest, save_episode

# model/search/sql должны поддерживать конкурентные async-вызовы.
# Их lifetime принадлежит вызывающему коду. Эталоны остаются вне AgentTask.
requests = (RolloutRequest(task, reference_document_ids=reference) for task, reference in train_rows)
async with RolloutRunner(model, search, EpisodeConfig(), concurrency=16) as runner:
    async with aclosing(runner.iter_episodes(requests)) as results:
        async for result in results:
            await asyncio.to_thread(save_episode, result, f"runs/train/{result.episode_id}.json")
```

`RolloutRequest` содержит `task`, необязательный `config`, `reference_document_ids` и `reference_answer`. `iter_episodes` принимает `Iterable`/`AsyncIterable` из `AgentTask` или `RolloutRequest`; выдача идёт по готовности. Сопоставляйте результаты по `task_id`/`episode_id`, а не по позиции. Не требуется собирать список всех результатов или заранее создавать asyncio-задачу на каждую строку.

Один runner владеет native interception и транспортом, ограничивает число активных эпизодов общим семафором, число workers и очередь результатов — `concurrency`. При нескольких одновременных batch семафор общий. Медленный consumer создаёт backpressure, память одного batch пропорциональна `concurrency`. Закрытие iterator отменяет только его batch, выход из runner — все его работы, включая ожидающие семафора. Ошибка источника задач пробрасывается вызывающему коду; ошибка модели/инструмента остаётся в соответствующем `EpisodeResult`. `on_interrupted` у `runner.run` сохраняет прежний контракт частичного результата.

`runner.run` принимает те же аргументы, что `run_episode`; model/search/config можно задать один раз в конструкторе или передать конкретному запуску. Для SQL передайте `sql` и приватный `reference_answer`. Отдельный `SearchState` и `ToolExecutor` создаются на каждый rollout; клиенты модели и бэкендов общие, состояния диалогов в них быть не должно.

`execution="in_process"` использует локальные доверенные Python-инструменты. `RolloutRunner(..., execution="subprocess")` включает прежний harness + MCP в отдельных процессах. Для нативного удалённого runtime укажите одновременно `env.agent.harness.execution="subprocess"` и `env.taskset.task.tools.execution="mcp"`; пакет и зависимости должны быть в образе. Внешний тренер использует native `Trace` и `TrainClient`, библиотечный API-клиент сам по себе токены и logprobs для обучения не создаёт.

## Минимальный цикл и действия

```text
вопрос → модель → tool_calls → search / finish → tool results → модель
```

Модель выбирает запросы, число поисков и момент завершения. Поиска перед `finish` не требуется: пустой список допустим. Нет предписанного чередования действий и лимита ошибочных попыток. Схемы передаются при каждом запросе, ответы инструментов возвращаются с соответствующим `tool_call_id`. Несколько вызовов выполняются последовательно.

`search(query: str)` возвращает `{"ok": true, "hits": [...]}`. `finish(document_ids: list[str])` удаляет повторы с сохранением порядка, проверяет показанные id и завершает эпизод. Его ответ сохраняется в журнале действий, даже если Verifiers завершает граф до добавления последнего tool-сообщения.

Неправильные аргументы и неизвестные инструменты возвращают ошибку модели через общий обработчик инструментов; в отдельном процессе ошибки схемы также может вернуть MCP. Придуманные id и пустой поисковый запрос возвращают `{"ok": false, "error": {"code": "...", "message": "..."}}`. Модель может исправиться на следующем ходу. После `finish` оставшиеся действия не выполняются. Текстовый ответ без `finish` завершает программу, но не даёт выбранных документов: `invalid_action`, код `missing_finish`. Обрезанная генерация (`finish_reason=length`) не исполняет инструменты.

## Параметры запуска

| Параметр EpisodeConfig | По умолчанию | Семантика |
|---|---|---|
| `max_steps` | `None` | Необязательный лимит ответов модели, передаётся в `AgentConfig.max_turns` |
| `max_output_tokens` | `None` | Необязательный суммарный бюджет генерации Verifiers |
| `max_tokens_per_step` | 512 | Лимит одного запроса модели |
| `search_limit` | 10 | Число уникальных документов в выдаче инструмента |
| `max_snippet_chars` | 2000 | Максимум символов в показанном сниппете |
| `seed` | `None` | Передаётся API при явном указании; требует поддержки сервера |

Бюджеты Verifiers проверяются между ходами: суммарные токены могут превысить заданное значение на последний ответ. `max_tokens_per_step` не уменьшается автоматически до остатка. В нативных конфигурациях бюджеты задаются в `env.agent`, размер ответа — в `sampling`, выдача — в `env.taskset.task.tools`. Отдельных лимитов поисков, ошибочных действий и финального списка нет. В Recall@10 учитываются только первые десять уникальных id.

`stop_reason`: `finished`, `max_steps`, `max_output_tokens`, `invalid_action`, `infrastructure_error`. Бюджетная остановка ставит `truncated=true` и оставляет итоговый список пустым; найденные документы остаются в истории. Успешный `finish` на последнем разрешённом ходу считается завершением. Финализация не вызывает модель и не выбирает документы за неё.

Сбой поиска даёт `search_error`, отсутствие usage — `missing_usage`, таймаут API — `model_timeout`; это инфраструктурные ошибки. Произвольный текст исключений модели/поиска не сохраняется. Автоповторов запросов нет. CLI задаёт полный deadline HTTP-запроса через `--request-timeout` (30 секунд), библиотечный клиент по умолчанию — 60 секунд. Это транспортный таймаут, не правило поведения агента.

## Траектория и обучение

`EpisodeResult` содержит задачу, конфигурацию, id эпизода, выбранные id, причину остановки, историю, действия и их результаты, usage каждого ответа и сумму, длительности, ошибки, `reward` и `native_trace`. Неизвестные токены/стоимость сохраняются как `null`. Стоимость берётся из usage сервера, не вычисляется по прайсу. Метаданные модели и reasoning-поля сохраняются без ключа авторизации.

В нативной `Trace` ответы модели помечены `sampled=true`, наблюдения — `false`. EvalClient сохраняет сообщения/usage; token_ids, logprobs и токенные маски в таком прогоне пусты. Для тренировочных траекторий нужен `TrainClientConfig` с renderer, совместимым с моделью, и vLLM `/inference/v1/generate`. Их формирует Verifiers; JSON API-прогона не заменяет этот путь. Обновление весов выполняет внешний тренер. Обучение на GPU пока не проверено.

`on_event` передаёт `model_started`, `model_finished`, `tool_started`, `tool_finished` для терминального прогресса. При отмене callback `on_interrupted` получает частичный результат, затем `CancelledError` распространяется дальше. CLI сохраняет отдельный `*.interrupted.json`, не меняет прежний результат и возвращает 130. Автоматическое продолжение эпизода не реализовано.

Версии кода, данных и модели, split, конфиги и MLflow-прогоны фиксирует runner. MLflow, пакетное возобновление и реальные поисковые адаптеры остаются следующими интеграциями.

## SQL + RAG: синтетический пример

`AgentTask` и `SearchBackend` сохраняют прежний формат. Необязательный keyword `sql` в `run_episode` включает связанный режим и добавляет публичную схему базы в промпт. Эталоны в AgentTask не добавляются.

```python
from search_rl_env.contracts import SqlResult

class SqlBackend:
    @property
    def schema(self) -> str:
        ...

    async def query(self, query: str) -> SqlResult:
        ...
```

`SqlResult` содержит `columns: list[str]`, `rows: list[list[str | int | float | None]]`, `truncated: bool`. Ширина каждой строки совпадает с числом колонок, неконечные числовые значения Infinity/NaN запрещены. `SqlQueryError` — ошибка запроса с безопасным сообщением для модели; иные исключения бэкенда становятся `infrastructure_error` с кодом `sql_error` и редактированием текста.

В этом режиме MCP предоставляет `search(query)`, `sql(query)` и `finish(answer: list[str], document_ids: list[str])`. `answer` — id подходящих заказов, а не свободный текст; `document_ids` — показанные источники. Повторы удаляются с сохранением порядка. Существование источников проверяется средой, правильность ответа — оценщиком. Порядок и количество поисков/SQL не задаются сценарием. Без `sql` набор инструментов и контракт retrieval-запуска сохраняются.

`MockSql` выполняет запросы в изолированной SQLite в памяти с таблицей `orders` и шестью фиксированными строками. Запись, DDL и ATTACH запрещены через authorizer, один вызов выполняет одно выражение. Результат ограничен 100 строками с явным `truncated`; вычисления — 100 000 инструкций VM, размер значения — 1 МБ. Эти ограничения относятся к одному вызову инструмента и не задают лимит числа SQL-действий агента. Ошибка SQL возвращается модели для исправления. Механизмы authorizer/progress handler описаны в [документации Python](https://docs.python.org/3.13/library/sqlite3.html#sqlite3.Connection.set_authorizer).

Результат версии 3 дополнительно содержит `answer: list[str] | None` и `sql_calls: int`. При бюджетной остановке и инфраструктурной ошибке ответ равен `None`; результаты SQL остаются в действиях и нативной Trace. Нативный Taskset включает режим через `scenario="sql_rag"`. Для своего корпуса и базы SQL-адаптер заменяет MockSql через Python-интерфейс; реальный HTTP-контракт с Димой ещё не согласован.

Приватный `reference_answer: list[str] | None` передаётся отдельно в `run_episode` или запись нативного Taskset. Пустое эталонное множество допустимо. `@vf.reward answer_accuracy` даёт 1 только за точное совпадение множеств id при завершённом ответе, иначе 0. Если эталона нет, награда отсутствует. В этом режиме retrieval-награда не суммируется с правильностью ответа. Инфраструктурные сбои не оцениваются нулём. Это проверка конкретного синтетического сценария; общий оценщик свободных SQL + RAG ответов не реализован.

## Профилирование и журнал событий

Добавлены необязательные поля без изменения интерфейса `ChatModel.complete`: `ModelResponse.http_profile`, `EpisodeStep.http_profile`, `EpisodeResult.profile`. Старые результаты версии 2/3 читаются со значением `None`. Формат EpisodeProfile имеет `version="1"`, формат строк JSONL — `schema_version="1"`.

Сводка эпизода содержит wall_seconds, setup_seconds, model_seconds, tools_seconds, finalize_seconds, scoring_seconds и other_seconds. Время активного эпизода берётся монотонными часами до освобождения его регистрации и инструментов; ожидание семафора и разовый запуск/закрытие общего runner измеряются в batch wall time отдельно; запись файла измеряется отдельно событием persisted в журнале UI. other_seconds — оставшееся время обвязки, IPC и неразнесённых участков, а не чистое CPU-время. framework_seconds сохраняет вложенные фазы Verifiers; agent там включает модель и инструменты, его нельзя прибавлять повторно к сводке.

HTTP-профиль содержит фазы httpcore с относительными отметками и длительностью, общий статус, код ответа, request_id и допустимый provider_request_id, точный размер исходящего JSON и размер декодированного тела ответа. connection_reused равен None, если транспорт не сообщает trace. Профиль при ошибке доступен через ModelError.http_profile и сохраняется в шаге. При отмене и таймауте остаются частичные фазы и отсутствующие значения None.

Кумулятивные time_to_headers_seconds и time_to_first_body_byte_seconds включают более ранние фазы; их нельзя суммировать с длительностями TCP/TLS. connect_tcp включает DNS; DNS отдельно не измеряется. time_to_transport_seconds включает подготовку HTTPX и потенциальное ожидание пула. Первый байт — первый непустой декодированный chunk HTTPX, не первый токен модели. Клиент сохраняет `stream=false`; очередь и генерация отдельно известны только при доступных числовых Server-Timing. Используются [trace callbacks httpcore](https://www.encode.io/httpcore/extensions/#trace), произвольные info и заголовки из callback не сохраняются.

Новые kind в EpisodeEvent: episode_started, episode_phase, model_progress, episode_finished. Все события библиотечного запуска содержат episode_id, timestamp UTC и elapsed_seconds от начала эпизода. Прежние события действий сохранены; подписчик должен обрабатывать известные kind. Журнал JSONL содержит только разрешённые метрики и идентификаторы; вопросы, SQL-тексты, ответы, сырые HTTP-заголовки и тела в него не дублируются. Полные действия и наблюдения остаются в защищённых артефактах EpisodeResult согласно исходному контракту.

UI пишет JSONL на каждом событии, включая ожидание HTTP; журнал доступен через `/api/runs/<run_id>/events-log`, результат — через `/api/runs/<run_id>/result`. Сохранённые ссылки работают после перезапуска UI. `/api/runs` сообщает состояние, шаг и последнюю фазу без текста вопросов. У CLI журнал по умолчанию имеет суффикс `.events.jsonl` и дописывается с episode_id для различения попыток.

UI владеет одним HTTP-пулом для API-моделей и закрывает его при завершении сервера после активных эпизодов. Асинхронные независимые эпизоды могут использовать один ChatCompletionsClient/AsyncClient; профили и callbacks разделены через ContextVar, изменяемое состояние инструмента остаётся у эпизода. Числом одновременных эпизодов управляет runner Коли. Нативные actions, observations, reward, sampled-маркеры и путь TrainClient не изменяются. Обновление весов на GPU этой проверкой не заменяется.

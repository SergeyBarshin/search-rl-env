# Границы компонентов и общие форматы

> Текущий вектор — [MEMORY.md](../MEMORY.md), планируемая память — [MEMORY_CORE.md](MEMORY_CORE.md). Ниже действующие контракты; новые операции, события и метаданные согласуются перед реализацией. Обновление плана автоматически не меняет версию EpisodeResult.

Целевая постановка — **Text-to-SQL + RAG**. Runtime поддерживает поиск свидетельств через `finish(document_ids)` и SQL + RAG через `finish(answer, document_ids)`. Синтетическая задача проверяет ID ответа, агентный пилот HyDRA сохраняет свободный текст в одной строке списка `answer`. [Постановка](AGENT_SCOPE.md) · [SQL-бэкенд](SQL_BACKEND.md) · [HyDRA](HYDRA_PILOT.md).

Харнесс на OpenAI Agents SDK записывает `EpisodeResult.schema_version="4"`. Версия 4 сохраняет публичную историю, признак SQL-режима и SHA-256 системного промпта. Результаты версий 2 и 3 с `native_trace` и `reward` читаются для архивного разбора. Типы находятся в `src/search_rl_env/contracts/types.py`.

## Задача и эталон

`AgentTask` содержит только `task_id`, `question`, `language`, `corpus_id`, необязательный `corpus_revision`. Неизвестные поля отклоняются. Агент видит JSON этой задачи, описание инструментов и историю сообщений.

Локальный runner может читать записи с задачами и приватными эталонами в таком формате.

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

При чтении такого файла вызывающий код удаляет `reference_document_ids` перед созданием `AgentTask` и передаёт их в `RolloutRequest.reference_document_ids` или аргументом оценщику. [Пример загрузки](HARNESS_HANDOFF.md#коле-один-эпизод-и-свои-задачи).

Для локального эпизода `score_episode` вычисляет Recall@10 по первым десяти уникальным ID. Пустой эталон отклоняется. При отсутствии эталона или при инфраструктурном сбое `EpisodeResult.reward=null`. Команда текстового бенчмарка `run` получает TSV вопросов, а `evaluate` читает qrels и сохранённые эпизоды. Для TRuST оценщик отбирает первые десять позиций, переводит ID чанков в родительские ID и удаляет повторы документов. [Протокол](EXPERIMENTS.md#закрытый-бенчмарк-текстового-поиска).

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

Функция запуска принимает любой объект с этим async-интерфейсом. SDK вызывает бэкенд в том же процессе. Каждый эпизод хранит собственный набор показанных документов. Снимки BM25 и Jina/FAISS открываются на прогон. Для HTTP доступны `SearchClient` и `ReferenceSearch`. [Бэкенды поиска](SEARCH_BACKEND.md).

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

`run_episode` вызывает `Runner` OpenAI Agents SDK с объектами `ChatModel` и `SearchBackend`. Приватные keyword-аргументы `reference_document_ids` и `reference_answer` передаются оценщику после запуска. `save_episode` атомарно сохраняет UTF-8 JSON. Новая попытка получает новый `episode_id`.

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

Общий клиент SDK использует внутрипроцессный транспорт. Для каждого эпизода SDK получает отдельные инструменты и состояние. Клиенты провайдеров могут разделять пул HTTP-соединений.

## Минимальный цикл и действия

```text
вопрос → модель → tool_calls → search / finish → tool results → модель
```

Модель выбирает запросы, число поисков и момент завершения. Поиска перед `finish` не требуется: пустой список допустим. Нет предписанного чередования действий и лимита ошибочных попыток. Схемы передаются при каждом запросе, ответы инструментов возвращаются с соответствующим `tool_call_id`. Несколько вызовов выполняются последовательно.

`search(query: str)` возвращает `{"ok": true, "hits": [...]}`. `finish(document_ids: list[str])` удаляет повторы с сохранением порядка, проверяет показанные id и завершает эпизод. Результат finish сохраняется в истории и журнале действий.

Неправильные аргументы и неизвестные инструменты возвращают ошибку модели через общий обработчик инструментов. Придуманные id и пустой поисковый запрос возвращают `{"ok": false, "error": {"code": "...", "message": "..."}}`. Модель может исправиться на следующем ходу. После `finish` оставшиеся действия не выполняются. Ответ без инструментов допускает настроенное число напоминаний о `finish`. После их исчерпания эпизод завершается с `invalid_action` и кодом `missing_finish`. Обрезанная генерация (`finish_reason=length`) не исполняет инструменты.

Текстовый эпизод с `EpisodeConfig.retrieval_rounds` принимает необязательный `search.top_k` в пределах `search_limit`. Выдача фильтрует уже показанные в раунде и перенесённые фрагменты. `save_and_advance(document_ids, summary, unresolved, next_goal)` переносит до 15 выбранных свидетельств, резюме, нерешённые вопросы и цель. Исходный запрос и эти поля составляют новый контекст, а публичная траектория сохраняет все раунды. `finish` принимает до 10 ранжированных ID, доступных в текущем контексте. Номер раунда доступен в `EpisodeStep.retrieval_round`, переход записывается событием `retrieval_round_started`. Для SQL этот режим не поддерживается.

При `retrieval_rounds.preserve_ranking=true` харнесс выполняет исходный поиск и журналирует его как действие шага 0, без вызова модели. Ход финализации формирует отдельный контекст из вопроса и краткого списка найденных свидетельств. `retrieval_rounds.final_snippet_chars` по умолчанию ограничивает финальный сниппет 1000 символами. CLI в режиме с ограничением по токенам использует уже обрезанный текст поискового наблюдения. На остановке по лимиту или невалидному действию поле `selected_document_ids` может содержать резервное ранжирование. `ranking_source` принимает значения `model`, `model_plus_retrieval` и `retrieval_fallback`. Старые эпизоды читаются с `null`. Инфраструктурные остановки сохраняют прежнюю семантику. [Методика и конфигурация CLI](EXPERIMENTS.md#агентный-поиск-с-отдельной-финализацией).

## Параметры запуска

| Параметр EpisodeConfig | По умолчанию | Семантика |
|---|---|---|
| `max_steps` | `None` | Необязательный лимит ответов модели, SDK получает остаток в `Runner.run(max_turns=...)` |
| `finish_on_last_step` | `False` | На последнем ходу при заданном `max_steps` оставить модели инструмент `finish` и попросить сдать ответ |
| `max_output_tokens` | `None` | Необязательный суммарный бюджет генерации |
| `max_finish_retries` | `0` | Число последовательных напоминаний после ответа без инструментов |
| `rollout_timeout_seconds` | `None` | Необязательный предел времени запросов модели и выполнения инструментов |
| `max_tokens_per_step` | 512 | Необязательный лимит одного запроса; None не передаёт max_tokens провайдеру |
| `search_limit` | 10 | Число уникальных документов в выдаче инструмента |
| `max_snippet_chars` | 2000 | Максимум символов; None показывает результат без обрезки харнессом |
| `seed` | `None` | Передаётся API при явном указании; требует поддержки сервера |
| `retrieval_rounds` | `None` | Раундовый режим для текстового поиска с отдельными лимитами хода и контекста |

Бюджет генерации проверяется между запросами. Итоговые токены могут превысить лимит на величину последнего ответа. Размер ответа задаёт `max_tokens_per_step`.

`stop_reason`: `finished`, `max_steps`, `max_output_tokens`, `invalid_action`, `infrastructure_error`. Бюджетная остановка ставит `truncated=true` и оставляет итоговый список пустым; найденные документы остаются в истории. Успешный `finish` на последнем разрешённом ходу считается завершением. Финализация не вызывает модель и не выбирает документы за неё.

Сбой поиска даёт `search_error`, отсутствие usage — `missing_usage`, таймаут API — `model_timeout`; это инфраструктурные ошибки. Произвольный текст исключений модели/поиска не сохраняется. Автоповторов запросов нет. CLI задаёт полный deadline HTTP-запроса через `--request-timeout` (30 секунд), библиотечный клиент по умолчанию — 60 секунд. Это транспортный таймаут, не правило поведения агента.

## Траектория и обучение

`EpisodeResult` содержит задачу, конфигурацию, id эпизода, итоговый ответ и выбранные источники, причину остановки, историю, действия, usage, длительности и ошибки. Неизвестные токены и стоимость сохраняются как `null`. Поле `reward` доступно вызывающему коду после локальной оценки. Публичный файл v4, записанный через `save_episode`, исключает это поле.

Обычный API-эпизод сохраняет текстовые сообщения и наблюдения. Для обучения отдельно потребуется формат с токенами и масками.

`on_event` передаёт `model_started`, `model_finished`, `tool_started`, `tool_finished` для терминального прогресса. При отмене callback `on_interrupted` получает частичный результат, затем `CancelledError` распространяется дальше. CLI сохраняет отдельный `*.interrupted.json`, не меняет прежний результат и возвращает 130. Автоматическое продолжение эпизода не реализовано.

Версии кода, данных и модели, split, конфиги и MLflow-прогоны фиксирует runner. MLflow и пакетное возобновление остаются следующими интеграциями. Python-сервис с SQLite FTS5, HTTP API v1 и Python-клиентом к текущему SearchBackend описаны в [SEARCH_BACKEND.md](SEARCH_BACKEND.md); реальные наборы данных подключаются отдельно.

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

В этом режиме агенту доступны `search(query)`, `sql(query)` и `finish(answer: list[str], document_ids: list[str])`. `answer` — id подходящих заказов, а не свободный текст; `document_ids` — показанные источники. Повторы удаляются с сохранением порядка. Существование источников проверяется средой, правильность ответа — оценщиком. Порядок и количество поисков/SQL не задаются сценарием. Без `sql` набор инструментов и контракт retrieval-запуска сохраняются.

`MockSql` выполняет запросы в изолированной SQLite в памяти с таблицей `orders` и шестью фиксированными строками. Запись, DDL и ATTACH запрещены через authorizer, один вызов выполняет одно выражение. Результат ограничен 100 строками с явным `truncated`; вычисления — 100 000 инструкций VM, размер значения — 1 МБ. Эти ограничения относятся к одному вызову инструмента и не задают лимит числа SQL-действий агента. Ошибка SQL возвращается модели для исправления. Механизмы authorizer/progress handler описаны в [документации Python](https://docs.python.org/3.13/library/sqlite3.html#sqlite3.Connection.set_authorizer).

Гибридный эпизод сохраняет `answer: list[str] | None` и `sql_calls: int`. Строка, переданная в `finish` как `answer`, сохраняется одноэлементным списком. При бюджетной остановке и инфраструктурной ошибке ответ равен `None`. Наблюдения SQL остаются в истории и действиях. В HyDRA `answer` содержит свободный текст.

`score_episode` отдельно оценивает синтетический результат. Совпадение множеств id даёт 1, ошибка ответа даёт 0. Пустой эталон допустим. Оценка инфраструктурного сбоя отсутствует. Для свободного текста HyDRA предусмотрен [отдельный оценщик](HYDRA_PILOT.md).

## Профилирование и журнал событий

Добавлены необязательные поля без изменения интерфейса `ChatModel.complete`: `ModelResponse.http_profile`, `EpisodeStep.http_profile`, `EpisodeResult.profile`. Старые результаты версии 2/3 читаются со значением `None`. Формат EpisodeProfile имеет `version="1"`, формат строк JSONL — `schema_version="1"`.

Сводка эпизода содержит wall_seconds, setup_seconds, model_seconds, tools_seconds и other_seconds. Ожидание семафора и клиентские ресурсы учитываются в полном времени серии, сохранение файла — в журнале UI. Поля finalize_seconds, scoring_seconds и framework_seconds доступны при чтении архивных профилей.

HTTP-профиль содержит фазы httpcore с относительными отметками и длительностью, общий статус, код ответа, request_id и допустимый provider_request_id, точный размер исходящего JSON и размер декодированного тела ответа. connection_reused равен None, если транспорт не сообщает trace. Профиль при ошибке доступен через ModelError.http_profile и сохраняется в шаге. При отмене и таймауте остаются частичные фазы и отсутствующие значения None.

Кумулятивные time_to_headers_seconds и time_to_first_body_byte_seconds включают более ранние фазы; их нельзя суммировать с длительностями TCP/TLS. connect_tcp включает DNS; DNS отдельно не измеряется. time_to_transport_seconds включает подготовку HTTPX и потенциальное ожидание пула. Первый байт — первый непустой декодированный chunk HTTPX, не первый токен модели. Клиент сохраняет `stream=false`; очередь и генерация отдельно известны только при доступных числовых Server-Timing. Используются [trace callbacks httpcore](https://www.encode.io/httpcore/extensions/#trace), произвольные info и заголовки из callback не сохраняются.

Новые kind в EpisodeEvent: episode_started, episode_phase, model_progress, episode_finished. Все события библиотечного запуска содержат episode_id, timestamp UTC и elapsed_seconds от начала эпизода. Прежние события действий сохранены; подписчик должен обрабатывать известные kind. Журнал JSONL содержит только разрешённые метрики и идентификаторы; вопросы, SQL-тексты, ответы, сырые HTTP-заголовки и тела в него не дублируются. Полные действия и наблюдения остаются в защищённых артефактах EpisodeResult согласно исходному контракту.

UI пишет JSONL на каждом событии, включая ожидание HTTP; журнал доступен через `/api/runs/<run_id>/events-log`, результат — через `/api/runs/<run_id>/result`. Сохранённые ссылки работают после перезапуска UI. `/api/runs` сообщает состояние, шаг и последнюю фазу без текста вопросов. У CLI журнал по умолчанию имеет суффикс `.events.jsonl` и дописывается с episode_id для различения попыток.

UI владеет одним HTTP-пулом для API-моделей и закрывает его при завершении сервера после активных эпизодов. Асинхронные независимые эпизоды могут использовать один ChatCompletionsClient/AsyncClient; профили и callbacks разделены через ContextVar, изменяемое состояние инструмента остаётся у эпизода. Числом одновременных эпизодов управляет runner Коли. Нативные actions, observations, reward, sampled-маркеры и путь TrainClient не изменяются. Обновление весов на GPU этой проверкой не заменяется.

## HyDRA: первый реальный SQL/search replay

Python-сервис поддерживает опциональный read-only SQL API `/v1/sql` и `/v1/sql/schema`;
Python SQLClient реализует прежний SqlBackend. Основной пример использует одну
train-задачу HyDRA, BIRD hockey и KILT-срез. Контракты, источники и ограничения —
в [HYDRA_PILOT.md](HYDRA_PILOT.md). Это проверка инструментов с исходным SQL,
не самостоятельное решение задачи моделью и не официальный benchmark score.

## SQL-каталог для нескольких сред

Каталог, маршрутизация по db_id/db_revision, SQL-only запуск и SQLClientPool
описаны в [SQL_BACKEND.md](SQL_BACKEND.md). Клиент конкретной БД остаётся совместим
с SqlBackend; различающиеся привязки передаются через существующий runner.run(sql=...).
AgentTask и формат результата эпизода не изменяются.

## Контекст изолированной среды

Необязательный `EpisodeConfig.system_prompt` задаёт инструкции для эпизода.
Текст сохраняется в конфиге, публичная история включает фактический промпт,
а `prompt_revision` содержит его SHA-256.
Для свободного ответа HyDRA finish использует прежний answer:list[str] с одной
строкой полного ответа. Оценщик синтетических id к этому пилоту не применяется.

`EnvironmentFactory` из `search_rl_env.search` принимает список DatasetSpec-манифестов.
`async with factory.open(dataset_id) as env` создаёт отдельные SQLite и FAISS;
`env.search` и `env.sql` реализуют прежние SearchBackend/SqlBackend.
Перед запуском `env.validate_task(task)` проверяет корпус/ревизию.
Runner передаёт их через существующий `runner.run(task, search=env.search, sql=env.sql)`.

`env.manifest` содержит environment_id, dataset_id/revision, SQL id/revision/schema,
версию и ограничения SQLite, corpus/index/search revisions, энкодер и версии
FAISS/NumPy. Он сохраняется во внешнем binding рядом с EpisodeResult и связывается
по episode_id/attempt_id. Формат AgentTask и EpisodeResult не расширяется.
Reference-файл остаётся приватным входом оценщика. Пример входа/выхода,
закрытие/отмена и передача runner описаны в [ISOLATED_ENVIRONMENTS.md](ISOLATED_ENVIRONMENTS.md).


## Планируемые контракты адаптации

Публичный эпизод связывает session/episode ID, вопрос, публичные действия и наблюдения, первый ответ, версии харнесса и источников. Исследовательская оценка хранится отдельно. Кандидат содержит родительскую версию, целевой блок промпта или skill, патч и ссылки на discovery. Решение хранит replay ID, анонимизированные пары, голоса критика, расходы и выбранную версию либо сохранение родителя. Публикация действует с указанной границы эпизодов. Это проект интерфейсов поверх действующего EpisodeResult. [Архитектура](RUNTIME_OPTIMIZER.md).

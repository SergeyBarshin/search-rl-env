# Передача харнесса команде

Текущие контракты и приёмка описаны в [переходе на SDK](OPENAI_AGENTS_MIGRATION.md). Постоянная память описана в [MEMORY_CORE.md](MEMORY_CORE.md).

Ниже действующие интерфейсы и запускаемые примеры. Для текстового поиска `examples/run_text_retrieval.py` собирает SQLite/BM25 и Jina/FAISS, запускает эпизоды и оценивает сохранённый результат. [Порядок публичных замеров](EXPERIMENTS.md#план-замеров-на-публичных-наборах).

Код обвязки находится в пакете `search_rl_env`. Runtime использует OpenAI Agents SDK 0.24.0, артефакт — EpisodeResult v4. Типы и форматы описаны в [CONTRACTS.md](CONTRACTS.md).

## С чего начать

Из корня клона с актуальной веткой харнесса:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Для уже настроенного проекта используйте существующий `.venv`. Результаты примеров пишутся в игнорируемый Git каталог `runs/handoff/`. Моковые команды и Python-примеры ниже выполняются без внешних API, весов и GPU.

| Кому | Первый результат | Точка подключения |
|---|---|---|
| Коля | Прочитать сохранённый эпизод, запустить свой список задач, подключить оценщик | `run_episode`, `RolloutRunner`, `EpisodeResult`, `save_episode` |
| Дима | Заменить фиксированную выдачу и тестовую SQLite своими сервисами | `SearchBackend.search`, `SqlBackend.schema/query` |
| Вова | Дать вопросы, отдельные эталоны и проверку результата | `AgentTask`, приватные эталоны и независимый оценщик |
| Сергей | Поддерживать цикл, форматы и совместимость с моделями | `agent/`, `environment/`, последующий `training/` |

## Проверить RAG, SQL и hybrid

RAG, один вопрос:

```bash
python -u examples/run_episode.py --model mock --verbose \
  --output runs/handoff/rag/episode.json
```

Ожидается `search → finish`, `stop_reason=finished`, `selected_document_ids=["doc-001", "doc-002"]`. Финал этого прототипа — выбранные документы; `answer=null`, свободный текст ответа ещё не является его форматом.

Hybrid, регламент и заказы:

```bash
python -u examples/run_episode.py --model mock --scenario sql_rag --verbose \
  --output runs/handoff/hybrid/episode.json
```

Ожидается `search → sql → finish`, ответ `ORD-101`, `ORD-105`, источник `policy-001`, один поиск и один SQL-вызов. Поиск даёт регламент, SQL действительно выполняется на шести строках SQLite в памяти. Фиксированный порядок запрограммирован в моке модели; API-модель выбирает действия сама.

SQL без поиска можно проверить через тот же API. Отдельного CLI-режима `sql_only` пока нет. В этом примере все условия находятся в вопросе; модель-мок вызывает `sql → finish`, доступный `search` не используется. Сохраните блок как `runs/handoff/sql_only.py` и выполните `python runs/handoff/sql_only.py`:

```python
import asyncio
import json

from search_rl_env.agent import run_episode, save_episode
from search_rl_env.agent.mocks import MockSearch, MockSqlRagModel
from search_rl_env.contracts import EpisodeConfig
from search_rl_env.environment.demo import HYBRID_REFERENCE, hybrid_task
from search_rl_env.environment.sql import MockSql


class SqlOnlyModel(MockSqlRagModel):
    @property
    def metadata(self):
        return {"adapter": "mock", "model": "sql-then-finish-v1"}

    async def complete(self, messages, **kwargs):
        response = await super().complete(messages, **kwargs)
        if not any(message["role"] == "tool" for message in messages):
            response.message["tool_calls"][0]["function"] = {
                "name": "sql",
                "arguments": json.dumps({"query": (
                    "SELECT order_id FROM orders WHERE region = 'RU' "
                    "AND status = 'delivered' AND delay_days > 3 ORDER BY order_id"
                )}),
            }
        return response


async def main():
    task = hybrid_task(task_id="sql-only-demo", question=(
        "Перечисли order_id заказов региона RU со статусом delivered "
        "и задержкой строго более 3 дней."
    ))
    search = MockSearch([], corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)
    result = await run_episode(task, SqlOnlyModel(), search, EpisodeConfig(max_steps=4),
                               sql=MockSql(), reference_answer=HYBRID_REFERENCE)
    assert result.stop_reason == "finished" and result.reward == 1
    assert result.search_calls == 0 and result.sql_calls == 1
    assert result.answer == ["ORD-101", "ORD-105"]
    save_episode(result, "runs/handoff/sql-only/episode.json")
    print(result.answer, result.selected_document_ids, result.reward)


asyncio.run(main())
```

`finish` в SQL/hybrid-демо принимает `answer: list[str]` — id заказов — и `document_ids: list[str]`. Универсальный табличный или текстовый ответ, отдельная SQL execution accuracy и строгий набор инструментов для SQL-only ещё требуют расширения контракта. Пример проверяет цикл SQL, а не готовность общего SQL-бенчмарка.

## Коле: один эпизод и свои задачи

Один запуск: `await run_episode(task, model, search, config, sql=sql)`. Без SQL аргумент `sql` опускается. Функция возвращает результат; файл записывает `save_episode(result, path)`. У каждой попытки новый `episode_id`; повторяемые вопросы связываются по `task_id` и номеру попытки runner Коли.

Для проверки скорости уже есть команда:

```bash
python -u examples/run_rollouts.py --model mock --scenario sql_rag \
  --episodes 32 --concurrency 8 --output runs/handoff/batch
```

Она **повторяет один синтетический вопрос с разными task id**. Это тест параллельности. Для своего набора передавайте отдельные вопросы в `RolloutRunner.iter_episodes`; источником может быть `Iterable` или `AsyncIterable`. Полный пример двух разных вопросов по тестовому корпусу:

```python
import asyncio
import json
from contextlib import aclosing
from pathlib import Path

from search_rl_env.agent import RolloutRequest, RolloutRunner, save_episode
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit


async def main():
    base = json.loads(Path("examples/fixtures/task.json").read_text())
    hits = [SearchHit.model_validate(hit) for hit in json.loads(
        Path("examples/fixtures/search_hits.json").read_text()
    )]
    rows = [
        {**base, "task_id": "handoff-1", "question": "Какие документы сообщают новую дату?",
         "reference_document_ids": ["doc-001", "doc-002"]},
        {**base, "task_id": "handoff-2", "question": "Где описана причина переноса?",
         "reference_document_ids": ["doc-001"]},
    ]

    def requests():
        for record in rows:  # Заменить своим источником согласованного формата.
            record = dict(record)
            reference = record.pop("reference_document_ids", None)
            yield RolloutRequest(AgentTask.model_validate(record), reference_document_ids=reference)

    search = MockSearch(hits, corpus_id=base["corpus_id"], corpus_revision=base["corpus_revision"])
    async with RolloutRunner(MockModel(), search, EpisodeConfig(max_steps=4), concurrency=2) as runner:
        async with aclosing(runner.iter_episodes(requests())) as results:
            async for result in results:
                await asyncio.to_thread(save_episode, result, f"runs/handoff/tasks/{result.episode_id}.json")
                print(result.task_id, result.stop_reason.value, result.reward)


asyncio.run(main())
```

Результаты приходят **по готовности**, а не в порядке входа. Сопоставляйте их по id. Модель и клиенты бэкендов общие, состояние диалога отдельное. Досрочный выход из `aclosing` отменяет остаток этого batch; выход из runner освобождает все его работы. Не нужно создавать задачу asyncio сразу на каждый вопрос большого набора.

Библиотечный runner отвечает за исполнение. Текстовый CLI поверх него записывает манифест, события и JSON по задачам, а команда `evaluate` строит отчёт из сохранённых файлов. Возобновление пакетного запуска и MLflow остаются в плане.

## Диме: заменить бэкенды

Интерфейсы находятся в [contracts/types.py](../src/search_rl_env/contracts/types.py). Наследование от конкретного класса не требуется:

| Интерфейс | Вход | Выход и ошибки |
|---|---|---|
| `SearchBackend.search` | `query`, keyword `corpus_id`, `corpus_revision`, `limit` | `list[SearchHit]`: `doc_id`, `snippet`, `rank` от 1 |
| `SqlBackend.schema` | Свойство выбранного SQL-бэкенда | Строка схемы, видимая модели |
| `SqlBackend.query` | Строка SQL | `SqlResult(columns, rows, truncated)`; безопасная ошибка запроса — `SqlQueryError` |

Подключение — передать свои объекты вместо `MockSearch` и `MockSql` в `run_episode`/`RolloutRunner`. Внутри процесса используются прямые async-вызовы. Клиенты должны выдерживать конкурентные вызовы и иметь собственные таймауты/пулы; их закрывает владелец, не runner. Поиск выбирает корпус по задаче; SQL-объект привязан к конкретной БД и схеме. SQL-сервис обеспечивает только чтение и ограничения времени/объёма результата. Для разных БД можно передать выбранный `sql` в отдельные `runner.run`; `iter_episodes` сейчас использует общий SQL-бэкенд runner.

SQL-демо использует SQLite. `db_id`, SQL-диалект и ревизия схемы задаются через бэкенд. Для поиска есть HTTP-клиент Python-сервиса и отдельный `ReferenceSearch` для публичного фиксированного индекса. [Поисковые контракты](SEARCH_BACKEND.md).

Проверка подключения: нужный корпус/БД и ревизия, постоянные id, правильный порядок выдачи, одна разрешимая задача, ошибка запроса, сбой сервиса и два независимых параллельных эпизода. Исключение сервиса становится `infrastructure_error`; его текст редактируется, такой эпизод не оценивается нулём.

## Конфиги модели и инструментов

`EpisodeConfig` задаёт необязательные бюджеты `max_steps`/`max_output_tokens`, размер одного ответа `max_tokens_per_step`, `search_limit`, `max_snippet_chars`, `seed`. Последние два поисковых параметра ограничивают размер наблюдения; числа поисков агенту они не задают. По умолчанию лимиты шагов и суммарных токенов выключены. Порядок действий принадлежит модели.

Для Chat Completions есть `ChatCompletionsClient`: endpoint, model id, авторизация, температура и deadline одного запроса. CLI/API-режимы читают локальные `MODEL_API_URL`, `MODEL_ID`, `MODEL_API_KEY` либо `MODEL_API_KEY_FILE`, необязательный `MODEL_AUTH_SCHEME`. Образец — [.env.example](../.env.example). Для обращения к модели из приведённых команд `--model mock` заменяется на `--model api`; поиск/SQL при этом остаются тестовыми. Реальный вызов потребляет согласованный API-бюджет.

Локальный пакетный прогон запускается через `examples/run_rollouts.py`. Реальный диагностический прогон описан в [HYDRA_PILOT.md](HYDRA_PILOT.md).

Текстовый CLI включает `retrieval_rounds` и использует один готовый поисковый снимок на параллельную серию. В новый раунд передаются выбранные свидетельства и цель, а весь ход поиска сохраняется в истории. [Как устроен поиск](RAG_SEARCH.md).

## Как читать результат и ошибки

```bash
python -m search_rl_env.profiling runs/handoff/hybrid/episode.json
```

| Поле EpisodeResult | Использование |
|---|---|
| `task_id`, `episode_id`, `schema_version` | Связь с задачей, попыткой и версией формата |
| `stop_reason`, `truncated`, `error` | Завершение, бюджетная остановка, инфраструктурный сбой |
| `selected_document_ids`, `answer` | Выбор документов и id ответа в SQL/hybrid-демо |
| `trajectory` | Ответы модели, действия, аргументы, результаты инструментов, ошибки, usage и времена шагов |
| `history`, `native_trace` | Диалог эпизода и архивное поле для прежнего формата |
| `usage`, `duration_seconds`, `profile` | Токены, доступная стоимость, модель/инструменты/обвязка |
| `reward` | Встроенная оценка при наличии подходящего приватного эталона; иначе `null` |

Ошибочные аргументы и неизвестный документ возвращаются модели для исправления. `invalid_actions > 0` не означает, что окончательный ответ неверен. `finished` означает завершение протокола; корректность оценивается отдельно. Текст без `finish` даёт `invalid_action/missing_finish`. Бюджетная остановка даёт `truncated=true`; финальный список/ответ очищаются. Неведомые usage/cost сохраняются как `null`, без подстановки нуля.

CLI возвращает `0` для `finished`, `1` для другой причины, `130` при Ctrl+C. Одиночный CLI сохраняет доступный частичный эпизод с суффиксом `.interrupted.json`. Для библиотечного `runner.run` есть callback `on_interrupted`; при отмене большого batch уже выданные результаты остаются у consumer, отменённые задачи не становятся обычными завершёнными результатами.

Время активного эпизода не включает ожидание семафора и разовый старт общего runner. Для пропускной способности измеряйте полное время серии и число **корректно завершённых** эпизодов. `run_rollouts.py` пишет `summary.json`; пилот HyDRA дополнительно пишет манифест и вердикты.

Эталоны передаются оценщику отдельно. Для retrieval используется Recall@10, для синтетических hybrid-задач — совпадение множеств id, для HyDRA — [отдельный оценщик](HYDRA_PILOT.md).

## После передачи: учитель, GEPA и обучение

| Работа Сергея и Вовы в первую неделю | Проверяемый результат | Текущий статус |
|---|---|---|
| `train-задача → rollout учителя → проверка → SFT` | Принятые/отклонённые траектории, версии учителя и проверяющего, причины отбраковки, экспорт под chat template | Есть rollout и траектория; общего проверяющего и SFT-экспортера пока нет |
| GEPA параллельно учителю | Адаптер задачи/feedback, кандидаты промпта, расходы, выбор на внутреннем dev | Адаптер и пилот ещё предстоят |
| Совместимость с тренировкой | Инструменты, token ids, logprobs, нулевая loss mask на наблюдениях инструментов | Есть синтетический контрактный тест native TrainClient; реальная модель не проверена |
| Техническое обновление весов | Изменившиеся параметры, сохранённый checkpoint, загрузка и повторный rollout | Требуются подготовленный тренировочный стек, GPU и разрешённый бюджет |

До генерации фиксируются разрешённые train/internal dev, источники и ревизии. Внешние оценочные вопросы и производные от них траектории исключены из GEPA/SFT/RL. SFT использует проверенные траектории; RL-пул задач не ограничивается успешными ответами учителя. Сохранённый текст API-диалога и положительный reward не заменяют проверку tokenizer/chat template/масок и фактического обновления параметров.

## Что подтвердить при передаче 3–4 ноября

- Коля самостоятельно запускает RAG, SQL и hybrid-примеры и читает сохранённый результат.
- Его список разных задач проходит пакетный интерфейс; ошибки одного эпизода и отмена обработаны.
- Дима подтвердил Python-интерфейсы, выбор источника и формат ошибок; согласованы следующие изменения для реальных БД.
- Вова подтвердил формат задачи/эталона и критерий проверки результата.
- Общие расширения финального ответа и конфигов зафиксированы до первого реального SQL/RAG/hybrid-бенчмарка.

Это пункты совместной приёмки, а не утверждение, что участники уже её провели. Поддержка харнесса остаётся у Сергея; экспериментальный runner, возобновление, MLflow и таблицы — у Коли.

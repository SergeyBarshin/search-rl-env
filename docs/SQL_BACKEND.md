# SQL-бэкенд для пакетного прогона команды

Цель: runner команды берёт 1000 задач и выполняет 16/32/64 эпизода одновременно.
Каждая задача выбирает исходную SQLite-БД и её ревизию. SQL только для чтения.
Бэкенд предоставляет данные и инструменты; запуск модели, очередь эпизодов,
сохранение результатов и оценщики остаются у харнесса и runner команды.

```text
1000 задач → runner: 16 / 32 / 64 активных агента
                           ↓ HTTP SQL
                    один Go-бэкенд
                           ↓ db_id + db_revision
                    набор SQLite-снимков
```

1000 задач не означают 1000 серверов БД. Задачи с одинаковой БД используют один
неизменяемый файл; история агента и состояние попытки отдельны у каждого эпизода.
Между задачами не переключается общая «текущая БД».

## Что передаёт Дима

- Go-программу с SQL API и каталогом нескольких БД/ревизий;
- Python SQLClientPool / SQLClient, совместимые с существующим SqlBackend;
- ограничения нагрузки, очереди, времени и результата;
- команды запуска и проверку 1000 SQL RPC.

Здесь нет деплоя, Docker/Kubernetes, аренды хоста или запуска платной модели.
Ответственный за размещение позже запускает тот же бинарник рядом с файлами
данных и сообщает команде base URL. Публикация исходников в Git сама не запускает сервис.

## Локальный запуск без поиска

Для примера BIRD hockey подготовьте существующий артефакт; свой набор может
использовать произвольные заранее подготовленные SQLite-файлы:

```bash
python examples/prepare_hydra.py
go -C services/search build -o ../../build/search-service ./cmd/search
./build/search-service serve-sql -sql-catalog configs/backend/sql-catalog.example.json
```

Не нужны поисковый индекс, FAISS, embedding-модель или GPU. Сервис по умолчанию
слушает 127.0.0.1:8090. Схема и ревизии готовятся до сообщения sql_ready.

Каталог version=1:

```json
{
  "version": 1,
  "databases": [
    {
      "db_id": "hockey",
      "path": "../../data/hydra/hockey.sqlite",
      "revision": "2987cba3921e2544fcd4603eccf5ff97b17f63103d8feb6fb908d55eb5bd9ba2"
    }
  ]
}
```

Пути относительны каталогу JSON-файла, не текущей директории. revision — SHA-256
исходного файла, а не произвольное имя. Для нового файла добавьте запись с его
db_id, путём и хешем (`sha256sum FILE` в Linux или `shasum -a 256 FILE` в macOS).
Несколько ревизий одного db_id могут находиться в одном каталоге.

При старте проверяется каждый файл. Неправильный SHA, отсутствующий файл,
повторная пара db_id/revision или неверная схема каталога не дают сервису
объявить готовность. Большой каталог требует времени на чтение и хеширование БД.
Пределы: 4096 записей, 1 МиБ JSON, 64 МиБ суммарных текстов SQL-схем.

Чтобы сменить данные, подготовьте новый файл/ревизию и новый каталог, затем
перезапустите сервис. Нельзя менять обслуживаемый файл на месте. Старые эпизоды
не должны незаметно начать обращаться к новой версии данных.

## API и выбор БД

| Метод | Назначение |
|---|---|
| GET /healthz | Готовность процесса |
| GET /v1/sql/databases | Список db_id/db_revision без путей файлов |
| GET /v1/sql/schema?db_id=...&db_revision=... | Схема выбранной БД |
| POST /v1/sql | Выполнение запроса на выбранном снимке |

POST содержит `db_id`, `db_revision`, `query`. Ответ — `columns`, `rows`,
`truncated`, `db_id`, `db_revision`. Имена колонок могут повторяться и сохраняются
списком. SQLClient проверяет, что ответ относится к закреплённой БД.

Если единственный снимок можно выбрать однозначно, старый SQLClient без db_id
продолжает работать. При нескольких подходящих снимках schema не выбирает
первый/последний: нужно явно указать id и ревизию. В запросах SQL оба поля обязательны.
Несуществующая пара возвращает unknown_id/404. Пути к файлам через API не принимаются.

SQL-only режим не открывает FTS-индекс и не предоставляет search/read API.
Совместный запуск поиска и каталога также поддержан:

```bash
./build/search-service serve -snapshot indexes/hydra.sqlite \
  -sql-catalog configs/backend/sql-catalog.example.json
```

Прежние `-sql-snapshot PATH -db-id ID` сохранены для одного снимка; смешивать их
с sql-catalog нельзя.

## Подключение к эпизодам

Один пул создаётся на весь прогон. Клиент конкретного снимка можно использовать
параллельно: в нём нет истории диалога и изменяемого выбора БД.

```python
from search_rl_env.search import SQLClientPool

async with SQLClientPool("http://127.0.0.1:8090", concurrency=64) as sql_pool:
    catalog = await sql_pool.databases()
    database = catalog[0]  # В реальном прогоне выбирается из метаданных задачи.
    sql = await sql_pool.bind(database["db_id"], database["db_revision"])
    result = await sql.query("SELECT name FROM sqlite_schema WHERE type='table'")
    # await runner.run(task, sql=sql)
```

Для набора с разными БД привязка выбирается для каждой задачи. Предлагаемый вход
на границе runner: `{"task": <AgentTask>, "sql": {"db_id": "...", "db_revision": "..."}}`.
Это внешний манифест запуска: поля db_id не добавляются в AgentTask и не передаются
модели как аргументы SQL. Модель получает выбранную публичную схему.

Пример использования существующего RolloutRunner с разными SQL-привязками:

```python
import asyncio
from search_rl_env.agent import RolloutRunner
from search_rl_env.contracts import AgentTask
from search_rl_env.search import SQLClientPool

# rows — итератор подготовленного манифеста, model/search/config задаёт команда.
# persist — асинхронная функция runner Коли, сохраняющая результат и версии.
async def run_rows(rows, model, search, config, persist, url, concurrency=16):
    source = iter(rows)
    async with SQLClientPool(url, concurrency=concurrency) as sqls:
        async with RolloutRunner(model, search, config, concurrency=concurrency) as runner:
            async def worker():
                for row in source:
                    sql = await sqls.bind(row["sql"]["db_id"], row["sql"]["db_revision"])
                    task = AgentTask.model_validate(row["task"])
                    result = await runner.run(task, sql=sql)
                    await persist(result, sql.info)
            async with asyncio.TaskGroup() as group:
                for _ in range(concurrency):
                    group.create_task(worker())
```

Создаются только N workers, а не 1000 активных корутин эпизодов. Существующий
iter_episodes использует один общий sql; для разных БД показан уже доступный
runner.run(..., sql=...). Код харнесса не изменяется. Настройка доступных агенту
инструментов и вызовы настоящей модели принадлежат команде, эти команды их не запускают.

## Нагрузка и ошибки

По умолчанию общие workers=16, max-pending=128, deadline=5s с очередью.
Параллельность агентов 16/32/64 задаёт runner, а workers ограничивает одновременные
операции бэкенда. Это разные параметры. Их не требуется делать одинаковыми.

Выполняемые и ожидающие операции ограничены на весь процесс, а не отдельно
для каждой БД. SQL-пулы не оставляют idle-соединения открытыми: число файловых
дескрипторов не растёт как число зарегистрированных БД × размер пула.
Тексты схем кешируются, сами базы не загружаются целиком в память.

Превышение очереди — busy/503; deadline — timeout/504; неверная БД — unknown_id/404;
невалидный запрос — sql_query_error/400; технический сбой — storage_error/500.
Автоматических повторов клиента нет. SQLite только для чтения; ограничения
запросов/результатов описаны в HYDRA_PILOT.md. Изменяемые БД и другие SQL-диалекты
не входят в этот контракт.

Логи операций содержат статус, длительности, db_id/db_revision и доступные
поисковые версии. В SQL-only режиме поисковые поля равны null. SQL-тексты,
табличные данные и файловые пути в лог не включаются.

## Проверка передачи бэкенда

```bash
python examples/sql_backend_smoke.py --operations 1000 --concurrency 16 --output runs/backend/sql-16.json
python examples/sql_backend_smoke.py --operations 1000 --concurrency 32 --output runs/backend/sql-32.json
python examples/sql_backend_smoke.py --operations 1000 --concurrency 64 --output runs/backend/sql-64.json
go -C services/search test -race ./...
go -C services/search vet ./...
python -m pytest -q
```

Smoke создаёт ограниченное число workers и читает схему зарегистрированных БД;
проверяет версии и стабильность ответов. Проверены 1000/1000 RPC без ошибок для
16, 32 и 64. Go-тесты проверяют маршрутизацию на четырёх снимках, в том числе
двух ревизиях одного id. Сквозные HTTP-тесты чередуют два разных файла БД и
проверяют содержимое ответов при тех же трёх уровнях параллельности.

Это проверка бэкенда, не запуск 1000 агентов и не оценка качества benchmark.
Сложность реальных SQL, модель, её квоты и лимиты влияют на скорость полного
прогона; её измеряют после подключения runner и данных команды.

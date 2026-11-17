# Python SQL-бэкенд

> SQL — инструмент самообучающегося харнесса; постоянная память — [MEMORY_CORE.md](MEMORY_CORE.md). Действующие команды и ограничения ниже относятся к SQL-бэкенду.

Сервис на aiohttp использует SQLiteSnapshot из search/local.py; тот же SQL-движок
используется изолированными средами. Дополнительный компилятор не нужен.

```bash
python examples/prepare_hydra.py
python -m search_rl_env.search.server serve-sql --sql-catalog configs/backend/sql-catalog.example.json
```

Каталог version=1 содержит databases: список записей db_id, path, revision.
Путь относителен каталогу, revision — SHA-256 SQLite. При старте проверяются
ревизии и схемы. Неизвестные поля и повторные пары БД/ревизия отклоняются.
GET /v1/sql/databases возвращает список привязок; GET /v1/sql/schema — выбранную
схему. POST /v1/sql принимает db_id, db_revision, query и возвращает SqlResult
с теми же id/ревизией. SQLClientPool использует этот API без изменения интерфейса.

SQLite authorizer запрещает запись, ATTACH, PRAGMA и посторонние/виртуальные
таблицы, включая pragma_database_list в любых кавычках. Обычные функции replace()
и идентификаторы вроде "delete" работают. Снимок должен быть самостоятельным:
непустые WAL/журнал отклоняются. Используются mode=ro, immutable и query_only.

Ограничения: 100 строк, 100 колонок, 256 КиБ содержимого результата,
значение до 1 МБ, SQL до 80 КБ, deadline и VM budget. Отмена вызывает interrupt
и дожидается worker до закрытия соединения. Ошибки запроса — sql_query_error/400,
ошибки снимка — storage_error/500; неизвестная БД — unknown_id/404.

```bash
python examples/sql_backend_smoke.py --operations 1000 --concurrency 16
python -m pytest tests/unit/search -q
```

Это команды проверки, а не утверждение об уже выполненном сетевом нагрузочном
прогоне новой реализации. Передача собственных сред и метрик — в
[ISOLATED_ENVIRONMENTS.md](ISOLATED_ENVIRONMENTS.md).

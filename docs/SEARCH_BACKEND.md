# Python-сервис поиска

> Этот сервис ищет документы RAG. История агента, долговременные записи и skills относятся к [MEMORY_CORE.md](MEMORY_CORE.md) и не становятся корпусом задачи автоматически.

HTTP-сервис реализован в `src/search_rl_env/search/server.py` на aiohttp.
SQLite FTS5 unicode61 обеспечивает BM25; исходные doc_id сохраняются.
Python SDK SearchClient реализует SearchBackend харнесса.

```bash
python -m search_rl_env.search.server build --input examples/fixtures/corpus.jsonl --output indexes/demo.sqlite --corpus-id demo
python -m search_rl_env.search.server serve --snapshot indexes/demo.sqlite
```

Сервис слушает 127.0.0.1:8090. Маршруты: GET /healthz, /v1/manifest;
POST /v1/search, /v1/read, /v1/chunks. Форматы ответов определены в search/models.py.
SDK закрепляет корпус и ревизии при открытии клиента. Поиск объединяет
экранированные слова запроса через OR; равные score сортируются по doc_id.

Сборка проверяет уникальность id и Unicode, создаёт FTS и чанки chars-v1,
публикует готовый файл без перезаписи существующего снимка.
Python-снимки имеют собственную версию алгоритма и требуют пересборки из JSONL.
Манифест содержит SQLite, алгоритм, content/index/search revisions.

HTTP body ограничен 64 КиБ; одновременно выполняются 16 операций,
выполняемых и ожидающих не больше 128. Deadline с очередью — 5 секунд.
SQLite вычисления идут в threads, соединение закрывается после операции;
отмена дожидается native-вызова. FTS progress handler ограничивает вычисление.
Ошибки не раскрывают SQL и внутренние пути. Аутентификацию/TLS предоставляет
отдельное размещение; по умолчанию сервис доступен только на loopback.

Read возвращает исходный текст с offset/next_offset в Unicode code points;
chunks возвращает страницы стабильных chunk_id. Сниппеты ограничены 500 символами.

```bash
python -m pytest tests/unit/search -q
```

Изолированные SQLite/FAISS и приёмка 1/40 описаны в
[ISOLATED_ENVIRONMENTS.md](ISOLATED_ENVIRONMENTS.md).

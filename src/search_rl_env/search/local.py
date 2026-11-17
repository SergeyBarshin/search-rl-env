"""Локальные инструменты изолированного эпизода: настоящие SQLite и FAISS."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
from time import monotonic
import unicodedata

from search_rl_env.contracts import SearchHit, SqlQueryError, SqlResult
from .models import SearchProtocolError, StorageError, UnknownId


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def require_standalone(path: Path):
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(path) + suffix)
        try:
            size = sidecar.stat().st_size
        except FileNotFoundError:
            continue
        except OSError:
            raise StorageError("Не удалось проверить журнал SQLite-снимка.") from None
        if size:
            raise StorageError("Нужен самостоятельный SQLite-снимок без WAL/журнала; используйте SQLite backup.")


async def joined_thread(function, *args, on_cancel=None):
    """Отмена не освобождает файлы/индекс раньше завершения native-вызова."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        if on_cancel is not None:
            on_cancel()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


class SQLiteSnapshot:
    """Один read-only снимок на эпизод; authorizer работает на уровне SQLite."""

    def __init__(self, path: Path, db_id: str, revision: str, *, timeout_seconds=1.0,
                 max_vm_steps=100_000, max_rows=100, max_result_bytes=256 * 1024):
        if (not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or
                min(max_vm_steps, max_rows, max_result_bytes) < 1):
            raise ValueError("Ограничения SQL должны быть положительными.")
        self.path = Path(path).resolve()
        require_standalone(self.path)
        self._signature = self._stat()
        if digest_file(self.path) != revision or self._signature != self._stat():
            raise StorageError("SQLite-снимок не соответствует ожидаемой ревизии.")
        self._db = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1",
                                   uri=True, check_same_thread=False)
        try:
            self._db.execute("PRAGMA query_only=1")
            self._db.execute("PRAGMA trusted_schema=0")
            schema = self._db.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name").fetchall()
            if not schema:
                raise StorageError("SQLite-схема пуста.")
            self.schema = "\n".join(statement + ";" for _, statement in schema)
            tables = {name for name, statement in schema
                      if not statement.lstrip().upper().startswith("CREATE VIRTUAL TABLE")}
            self._tables = tables | {"sqlite_schema", "sqlite_master"}
            self._db.set_authorizer(self._authorize)
            for limit, maximum in ((sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000),
                                   (sqlite3.SQLITE_LIMIT_COLUMN, 100),
                                   (sqlite3.SQLITE_LIMIT_SQL_LENGTH, 80_000),
                                   (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 100)):
                self._db.setlimit(limit, maximum)
        except sqlite3.Error:
            self._db.close()
            raise StorageError("Не удалось прочитать SQLite-снимок.") from None
        except BaseException:
            self._db.close()
            raise
        self.info = {"db_id": db_id, "db_revision": revision, "schema": self.schema}
        self.timeout_seconds, self.max_vm_steps = timeout_seconds, max_vm_steps
        self.max_rows, self.max_result_bytes = max_rows, max_result_bytes
        self._lock = asyncio.Lock()
        self.closed = False

    def _stat(self):
        try:
            stat = self.path.stat()
        except OSError:
            raise StorageError("SQLite-снимок недоступен.") from None
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _authorize(self, action, first, second, database, trigger):
        if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}:
            return sqlite3.SQLITE_OK
        # SQLite сообщает database=None/column="" для оптимизации count(*).
        if (action == sqlite3.SQLITE_READ and first in self._tables and
                (database == "main" or database is None and second == "")):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (second or "").lower() not in {
                "load_extension", "readfile", "writefile", "eval", "zipfile", "fts3_tokenizer"}:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def _query(self, query, cancelled):
        deadline = monotonic() + self.timeout_seconds
        remaining = self.max_vm_steps

        def progress():
            nonlocal remaining
            remaining -= 100
            return int(cancelled.is_set() or remaining <= 0 or monotonic() >= deadline)

        self._db.set_progress_handler(progress, 100)
        cursor = None
        try:
            cursor = self._db.execute(query)
            if cursor.description is None:
                raise SqlQueryError("Разрешён один запрос на чтение с табличным результатом.")
            columns = [column[0] for column in cursor.description]
            result, used, truncated = [], 0, False
            for row in cursor:
                if len(result) == self.max_rows:
                    truncated = True
                    break
                if any(isinstance(cell, bytes) or isinstance(cell, float) and not math.isfinite(cell)
                       for cell in row):
                    raise SqlQueryError("Бинарные значения и неконечные числа не поддерживаются.")
                size = sum(len(cell.encode()) if isinstance(cell, str) else 32 for cell in row)
                if used + size > self.max_result_bytes:
                    truncated = True
                    break
                used += size
                result.append(list(row))
            return SqlResult(columns=columns, rows=result, truncated=truncated)
        except sqlite3.Error as error:
            code = getattr(error, "sqlite_errorcode", 0) & 255
            if isinstance(error, sqlite3.ProgrammingError) or code in {
                    sqlite3.SQLITE_ERROR, sqlite3.SQLITE_AUTH, sqlite3.SQLITE_READONLY,
                    sqlite3.SQLITE_TOOBIG, sqlite3.SQLITE_MISMATCH, sqlite3.SQLITE_INTERRUPT}:
                raise SqlQueryError("SQL не разрешён, некорректен или превысил ограничения SQLite.") from None
            raise StorageError("Техническая ошибка SQLite-снимка.") from None
        finally:
            if cursor is not None:
                cursor.close()
            self._db.set_progress_handler(None, 0)

    async def query(self, query: str) -> SqlResult:
        if not isinstance(query, str) or not query.strip() or "\0" in query:
            raise SqlQueryError("Нужен непустой SQL-запрос без NUL.")
        try:
            query.encode("utf-8", errors="strict")
        except UnicodeError:
            raise SqlQueryError("Некорректный Unicode в SQL-запросе.") from None
        async with self._lock:
            if self.closed:
                raise StorageError("SQL-среда закрыта.")
            require_standalone(self.path)
            if self._stat() != self._signature:
                raise StorageError("SQLite-снимок изменился.")
            cancelled = threading.Event()

            def interrupt():
                cancelled.set()
                self._db.interrupt()

            result = await joined_thread(self._query, query, cancelled, on_cancel=interrupt)
            require_standalone(self.path)
            if self._stat() != self._signature:
                raise StorageError("SQLite-снимок изменился во время запроса.")
            return result

    async def aclose(self):
        async with self._lock:
            self.close()

    def close(self):
        """Синхронное закрытие подготовительного соединения без активного query."""
        if not self.closed:
            self.closed = True
            self._db.close()


class HashingEncoder:
    """Фиксированный lexical encoder для технической приёмки без весов модели."""

    def __init__(self, dimensions=128):
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 16 <= dimensions <= 4096:
            raise ValueError("Размерность должна быть целым числом от 16 до 4096.")
        self.dimensions = dimensions
        self.manifest = {"id": "sha256-token-hashing-v1", "dimensions": dimensions,
                         "tokenizer": "unicode-word-casefold-v1", "normalization": "l2",
                         "unicode_version": unicodedata.unidata_version}

    def encode(self, texts):
        import numpy as np

        vectors = np.zeros((len(texts), self.dimensions), dtype="float32")
        for index, text in enumerate(texts):
            for word in re.findall(r"\w+", text.casefold()):
                digest = hashlib.sha256(word.encode()).digest()
                vectors[index, int.from_bytes(digest[:8], "big") % self.dimensions] += 1 if digest[8] & 1 else -1
            norm = np.linalg.norm(vectors[index])
            if norm:
                vectors[index] /= norm
        return vectors


class FaissSearch:
    def __init__(self, index, documents, encoder, manifest):
        self._index, self._documents, self._encoder = index, documents, encoder
        self._manifest = deepcopy(manifest)
        self._lock = asyncio.Lock()
        self.closed = False

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def _search(self, query, limit):
        scores, positions = self._index.search(self._encoder.encode([query]), len(self._documents))
        candidates = [(float(score), self._documents[int(position)])
                      for score, position in zip(scores[0], positions[0]) if position >= 0]
        candidates.sort(key=lambda item: (-item[0], item[1]["doc_id"]))
        return [SearchHit(doc_id=document["doc_id"], snippet=document["text"][:500], rank=rank)
                for rank, (_, document) in enumerate(candidates[:limit], 1)]

    async def search(self, query, *, corpus_id, corpus_revision, limit):
        if (corpus_id != self._manifest["corpus_id"] or
                corpus_revision not in (None, self._manifest["corpus_revision"])):
            raise UnknownId("Неизвестный корпус или ревизия.")
        if (not isinstance(query, str) or not query.strip() or len(query) > 2000 or "\0" in query or
                isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100):
            raise ValueError("Недопустимый query или limit.")
        async with self._lock:
            if self.closed:
                raise SearchProtocolError("Поисковая среда закрыта.")
            return await joined_thread(self._search, query, limit)

    async def aclose(self):
        async with self._lock:
            self.closed = True
            self._index = None

"""Поиск по документам с ограниченным пулом и отдельным соединением на вызов."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import logging
from pathlib import Path
import re
import sqlite3
import threading
from time import perf_counter
import unicodedata

from .models import (InvalidRequest, Match, SearchBusy, SearchConfig, SearchError,
                     SearchResult, SearchTimeout, StorageError, TextResult, UnknownId)
from .snapshot import canonical, connect_readonly, fingerprint, load_manifest

logger = logging.getLogger(__name__)


class LocalSearch:
    """Один экземпляр принадлежит одному event loop; закрывается через async with.

    max_pending включает выполняемые и ожидающие операции. Отмена не освобождает
    место до фактического завершения worker. Опубликованный файл нельзя заменять.
    """

    def __init__(self, path: str | Path, config: SearchConfig | None = None):
        self.path = Path(path).resolve()
        self.config = config or SearchConfig()
        try:
            db = connect_readonly(self.path)
            try:
                self._manifest = load_manifest(db)
            finally:
                db.close()
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
            raise StorageError("Не удалось открыть снимок поиска.") from None
        self.search_revision = fingerprint(["literal-or-bm25-v1", asdict(self.config)])
        self._executor = ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix="corpus-search")
        self._pending = {}
        self._loop = None
        self._closed = False

    @property
    def manifest(self):
        return dict(self._manifest)

    async def __aenter__(self):
        if self._closed:
            raise StorageError("Поиск уже закрыт.")
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        if not self._closed:
            self._closed = True
            for stop in self._pending.values():
                stop.set()
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    def _integer(self, value, maximum, name, *, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise InvalidRequest(f"Недопустимое значение {name}.")
        return value

    async def search(self, query: str, *, limit: int = 10) -> SearchResult:
        def operation(db):
            self._integer(limit, self.config.max_results, "limit")
            if not isinstance(query, str) or not query.strip() or len(query) > self.config.max_query_chars or "\x00" in query:
                raise InvalidRequest("Недопустимый поисковый запрос.")
            terms = list(dict.fromkeys(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", query))))
            if not terms or len(terms) > self.config.max_query_terms:
                raise InvalidRequest("Запрос не содержит слов или превышен лимит терминов.")
            expression = " OR ".join('"' + term + '"' for term in terms)
            rows = db.execute("""
                SELECT d.rowid, d.doc_id, bm25(document_fts)
                FROM document_fts JOIN documents d ON d.rowid=document_fts.rowid
                WHERE document_fts MATCH ?
                ORDER BY bm25(document_fts), d.doc_id LIMIT ?
            """, (expression, limit)).fetchall()
            # Сниппет вычисляется только для top-k, а не для всех совпадений.
            hits = []
            for rank, (rowid, doc_id, score) in enumerate(rows, 1):
                snippet = db.execute("""SELECT snippet(document_fts, 1, '', '', ' … ', 48)
                    FROM document_fts WHERE rowid=? AND document_fts MATCH ?""", (rowid, expression)).fetchone()[0]
                hits.append(Match(doc_id, snippet[:self.config.snippet_chars], rank, score))
            return SearchResult(tuple(hits), self._manifest["corpus_revision"],
                                self._manifest["index_revision"], self.search_revision)

        return await self._run("search", operation)

    async def read_document(self, doc_id: str, *, offset: int = 0, limit: int | None = None) -> TextResult:
        return await self._read(doc_id, chunk=False, offset=offset, limit=limit)

    async def read_chunk(self, chunk_id: str, *, offset: int = 0, limit: int | None = None) -> TextResult:
        return await self._read(chunk_id, chunk=True, offset=offset, limit=limit)

    async def _read(self, identifier, *, chunk, offset, limit):
        def operation(db):
            if not isinstance(identifier, str) or not identifier.strip():
                raise InvalidRequest("Нужен непустой идентификатор.")
            actual_limit = self.config.max_read_chars if limit is None else limit
            self._integer(actual_limit, self.config.max_read_chars, "limit")
            self._integer(offset, 2**63 - 1, "offset", minimum=0)
            if chunk:
                row = db.execute("""SELECT doc_id, start_char, end_char FROM chunks
                                    WHERE chunk_id=?""", (identifier,)).fetchone()
                if row is None:
                    raise UnknownId("Неизвестный chunk_id.")
                doc_id, start, end = row
                total = end - start
            else:
                row = db.execute("SELECT length(text) FROM documents WHERE doc_id=?", (identifier,)).fetchone()
                if row is None:
                    raise UnknownId("Неизвестный doc_id.")
                doc_id, start, total = identifier, 0, row[0]
            if offset > total:
                raise InvalidRequest("Смещение выходит за границы текста.")
            text = db.execute("SELECT substr(text, ?, ?) FROM documents WHERE doc_id=?",
                              (start + offset + 1, min(actual_limit, total - offset), doc_id)).fetchone()[0]
            next_offset = offset + len(text) if offset + len(text) < total else None
            return TextResult(doc_id, identifier if chunk else None, text, offset, next_offset,
                              total, self._manifest["corpus_revision"])

        return await self._run("read_chunk" if chunk else "read_document", operation)

    async def list_chunks(self, doc_id: str, *, offset: int = 0, limit: int = 10) -> tuple[str, ...]:
        """Ограниченная страница id фрагментов в порядке исходного текста."""
        def operation(db):
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise InvalidRequest("Нужен непустой doc_id.")
            self._integer(limit, self.config.max_results, "limit")
            self._integer(offset, 2**63 - 1, "offset", minimum=0)
            if db.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone() is None:
                raise UnknownId("Неизвестный doc_id.")
            return tuple(row[0] for row in db.execute("""SELECT chunk_id FROM chunks WHERE doc_id=?
                         ORDER BY start_char LIMIT ? OFFSET ?""", (doc_id, limit, offset)))

        return await self._run("list_chunks", operation)

    async def _run(self, name, operation):
        started = perf_counter()
        timing = {"queue_seconds": None, "execution_seconds": None}
        status = "ok"
        result_count = None
        stop = threading.Event()
        deadline = started + self.config.timeout_seconds

        def worker():
            worker_started = perf_counter()
            timing["queue_seconds"] = worker_started - started
            db = None
            try:
                if stop.is_set() or perf_counter() >= deadline:
                    raise SearchTimeout("Истёк срок выполнения операции.")
                db = connect_readonly(self.path)
                db.set_progress_handler(lambda: int(stop.is_set() or perf_counter() >= deadline), 1000)
                if load_manifest(db) != self._manifest:
                    raise StorageError("Снимок был заменён во время работы.")
                result = operation(db)
                if stop.is_set() or perf_counter() >= deadline:
                    raise SearchTimeout("Истёк срок выполнения операции.")
                return result
            except sqlite3.OperationalError as error:
                if error.sqlite_errorcode == sqlite3.SQLITE_INTERRUPT:
                    raise SearchTimeout("Операция поиска прервана.") from None
                raise StorageError("Ошибка чтения индекса.") from None
            except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
                raise StorageError("Ошибка чтения индекса.") from None
            finally:
                if db is not None:
                    db.close()
                timing["execution_seconds"] = perf_counter() - worker_started

        try:
            loop = asyncio.get_running_loop()
            if self._closed:
                raise StorageError("Поиск уже закрыт.")
            if self._loop is not None and self._loop is not loop:
                raise StorageError("Экземпляр поиска принадлежит другому event loop.")
            self._loop = loop
            if len(self._pending) >= self.config.max_pending:
                raise SearchBusy("Очередь поиска заполнена.")
            future = self._executor.submit(worker)
            self._pending[future] = stop
            future.add_done_callback(lambda done: loop.call_soon_threadsafe(self._pending.pop, done, None))
            wrapped = asyncio.wrap_future(future)
            try:
                result = await asyncio.wait_for(wrapped, timeout=max(0, deadline - perf_counter()))
            except TimeoutError:
                stop.set()
                raise SearchTimeout("Истёк срок выполнения операции.") from None
            except asyncio.CancelledError:
                stop.set()
                raise
            result_count = len(result.hits) if isinstance(result, SearchResult) else None
            return result
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except SearchError as error:
            status = type(error).__name__
            raise
        except Exception as error:
            status = type(error).__name__
            raise
        finally:
            logger.info(canonical(dict(operation=name, status=status, duration_seconds=perf_counter() - started,
                        result_count=result_count, corpus_id=self._manifest["corpus_id"],
                        corpus_revision=self._manifest["corpus_revision"], index_revision=self._manifest["index_revision"],
                        search_revision=self.search_revision, **timing)))


class HarnessSearch:
    """Адаптер существующего SearchBackend; ресурсом LocalSearch владеет вызывающий код."""

    def __init__(self, backend: LocalSearch):
        self.backend = backend

    async def search(self, query, *, corpus_id, corpus_revision, limit):
        from search_rl_env.contracts import SearchHit

        manifest = self.backend.manifest
        if corpus_id != manifest["corpus_id"] or corpus_revision not in (None, manifest["corpus_revision"]):
            raise UnknownId("Неизвестный корпус или ревизия.")
        result = await self.backend.search(query, limit=limit)
        return [SearchHit(doc_id=hit.doc_id, snippet=hit.snippet, rank=hit.rank) for hit in result.hits]

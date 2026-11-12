"""Асинхронный SearchBackend для HTTP API Go-сервиса."""

import asyncio
import math

import httpx
from pydantic import ValidationError

from search_rl_env.contracts import SearchHit, SqlQueryError, SqlResult
from .models import (ChunksResult, ErrorEnvelope, InvalidRequest, Manifest, SearchBusy,
                     SearchProtocolError, SearchResult, SearchTimeout, StorageError,
                     TextResult, UnknownId, SQLInfo, SQLResponse)

ERRORS = {
    "sql_query_error": (400, SqlQueryError, "SQL не разрешён или превышает ограничения SQLite."),
    "invalid_request": (400, InvalidRequest, "Недопустимый запрос к поиску."),
    "unknown_id": (404, UnknownId, "Неизвестный корпус, ревизия или идентификатор."),
    "busy": (503, SearchBusy, "Очередь поискового сервиса заполнена."),
    "timeout": (504, SearchTimeout, "Истёк срок выполнения поиска."),
    "cancelled": (408, SearchTimeout, "Сервис отменил операцию."),
    "storage_error": (500, StorageError, "Ошибка поискового сервиса."),
}


class SearchClient:
    """Один клиент на batch; манифест фиксируется при async with.

    Все ответы проверяются на те же версии. Клиент не запускает Go-процесс,
    не хранит корпус и не повторяет запросы. Внешним AsyncClient владеет вызывающий код.
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 10,
                 max_response_bytes: int = 2 * 1024 * 1024, http_client: httpx.AsyncClient | None = None):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("Нужен HTTP(S)-адрес без credentials, query и fragment.")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds должен быть конечным положительным числом.")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes < 1:
            raise ValueError("max_response_bytes должен быть положительным целым числом.")
        self._url = str(url).rstrip("/")
        self._timeout = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout_seconds,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16), trust_env=False,
            follow_redirects=False)
        self._manifest = None
        self._closed = False

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            raise SearchProtocolError("Сначала откройте SearchClient через async with.")
        return self._manifest.model_dump()

    async def __aenter__(self):
        if self._closed or self._manifest is not None:
            raise SearchProtocolError("Клиент уже открыт или закрыт.")
        try:
            self._manifest = await self._request("GET", "/v1/manifest", Manifest)
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        self._closed = True
        if self._owns_client:
            await self._http.aclose()

    async def _request(self, method, path, model, payload=None):
        if self._closed:
            raise SearchProtocolError("Клиент закрыт.")
        try:
            async with asyncio.timeout(self._timeout):
                async with self._http.stream(method, self._url + path, json=payload) as response:
                    data = bytearray()
                    async for part in response.aiter_bytes():
                        if len(data) + len(part) > self._max_response_bytes:
                            raise SearchProtocolError("Превышен размер ответа поиска.")
                        data.extend(part)
                    if response.status_code != 200:
                        error = ErrorEnvelope.model_validate_json(data).error
                        mapping = ERRORS.get(error.code)
                        if mapping is None or mapping[0] != response.status_code:
                            raise SearchProtocolError("Неизвестная ошибка HTTP-протокола поиска.")
                        raise mapping[1](mapping[2])
                    return model.model_validate_json(data)
        except (TimeoutError, httpx.TimeoutException):
            raise SearchTimeout("Истёк срок HTTP-запроса к поиску.") from None
        except httpx.HTTPError:
            raise StorageError("Не удалось обратиться к поисковому сервису.") from None
        except ValidationError:
            raise SearchProtocolError("Ответ поиска не соответствует HTTP-контракту.") from None

    def _scope(self, corpus_id, corpus_revision):
        if self._manifest is None:
            raise SearchProtocolError("Сначала откройте SearchClient через async with.")
        if corpus_id != self._manifest.corpus_id or corpus_revision not in (None, self._manifest.corpus_revision):
            raise UnknownId("Неизвестный корпус или ревизия.")
        return {"corpus_id": corpus_id, "corpus_revision": self._manifest.corpus_revision}

    def _check_versions(self, result):
        expected = self._manifest
        if (result.corpus_id, result.corpus_revision, result.index_revision, result.search_revision) != (
                expected.corpus_id, expected.corpus_revision, expected.index_revision, expected.search_revision):
            raise SearchProtocolError("Поисковое окружение изменилось во время прогона.")
        return result

    async def search(self, query: str, *, corpus_id: str, corpus_revision: str | None, limit: int) -> list[SearchHit]:
        payload = {**self._scope(corpus_id, corpus_revision), "query": query, "limit": limit}
        result = self._check_versions(await self._request("POST", "/v1/search", SearchResult, payload))
        if len(result.hits) > limit or len({hit.doc_id for hit in result.hits}) != len(result.hits):
            raise SearchProtocolError("Неверный размер или повторные id в выдаче.")
        if any(hit.rank != index or len(hit.snippet) > self._manifest.config.snippet_chars for index, hit in enumerate(result.hits, 1)):
            raise SearchProtocolError("Неверные позиции или размер сниппета.")
        return [SearchHit(doc_id=hit.doc_id, snippet=hit.snippet, rank=hit.rank) for hit in result.hits]

    async def read(self, *, doc_id=None, chunk_id=None, offset=0, limit=None) -> TextResult:
        if self._manifest is None:
            raise SearchProtocolError("Сначала откройте SearchClient через async with.")
        payload = {**self._scope(self._manifest.corpus_id, None), "offset": offset}
        if doc_id is not None:
            payload["doc_id"] = doc_id
        if chunk_id is not None:
            payload["chunk_id"] = chunk_id
        if limit is not None:
            payload["limit"] = limit
        result = self._check_versions(await self._request("POST", "/v1/read", TextResult, payload))
        if result.offset != offset or (doc_id is not None and result.doc_id != doc_id) or result.chunk_id != chunk_id:
            raise SearchProtocolError("Чтение вернуло другой идентификатор или offset.")
        end = result.offset + len(result.text)
        maximum = self._manifest.config.max_read_chars if limit is None else limit
        if end > result.total_chars or result.next_offset != (end if end < result.total_chars else None) or len(result.text) > maximum or (result.next_offset is not None and not result.text):
            raise SearchProtocolError("Нарушены границы чтения.")
        return result

    async def chunks(self, doc_id: str, *, offset=0, limit=10) -> ChunksResult:
        if self._manifest is None:
            raise SearchProtocolError("Сначала откройте SearchClient через async with.")
        payload = {**self._scope(self._manifest.corpus_id, None), "doc_id": doc_id, "offset": offset, "limit": limit}
        result = self._check_versions(await self._request("POST", "/v1/chunks", ChunksResult, payload))
        if len(result.chunk_ids) > limit or len(set(result.chunk_ids)) != len(result.chunk_ids) or any(not identifier for identifier in result.chunk_ids):
            raise SearchProtocolError("Неверная страница фрагментов.")
        if result.next_offset is not None and (result.next_offset != offset + len(result.chunk_ids) or len(result.chunk_ids) != limit):
            raise SearchProtocolError("Неверное продолжение страницы фрагментов.")
        return result


class SQLClient:
    """SqlBackend для исходной SQLite-базы в том же Go-сервисе."""

    def __init__(self, base_url: str, **kwargs):
        self._transport = SearchClient(base_url, **kwargs)
        self._info = None

    async def __aenter__(self):
        try:
            self._info = await self._transport._request("GET", "/v1/sql/schema", SQLInfo)
        except BaseException:
            await self._transport.aclose()
            raise
        return self

    async def __aexit__(self, *exc):
        await self._transport.aclose()

    @property
    def schema(self) -> str:
        if self._info is None:
            raise SearchProtocolError("Сначала откройте SQLClient через async with.")
        return self._info.schema_

    @property
    def info(self) -> dict:
        if self._info is None:
            raise SearchProtocolError("Сначала откройте SQLClient через async with.")
        return self._info.model_dump(by_alias=True)

    async def query(self, query: str) -> SqlResult:
        if self._info is None:
            raise SearchProtocolError("Сначала откройте SQLClient через async with.")
        response = await self._transport._request("POST", "/v1/sql", SQLResponse, {
            "db_id": self._info.db_id, "db_revision": self._info.db_revision, "query": query,
        })
        if response.db_id != self._info.db_id or response.db_revision != self._info.db_revision:
            raise SearchProtocolError("SQL-база изменилась во время прогона.")
        return SqlResult(columns=response.columns, rows=response.rows, truncated=response.truncated)

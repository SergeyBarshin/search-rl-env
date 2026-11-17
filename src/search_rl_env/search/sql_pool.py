"""Общий HTTP-пул и неизменяемые SQL-привязки для задач бенчмарка."""

import asyncio
from contextlib import AsyncExitStack

import httpx

from .client import SQLClient
from .models import SearchProtocolError


class SQLClientPool:
    """Один пул на прогон; bind возвращает SqlBackend конкретной БД/ревизии.

    Клиенты не переключают общую «текущую БД». Все привязки заимствуют один HTTP-пул;
    состояние модели и эпизода остаётся у харнесса.
    """

    def __init__(self, base_url: str, *, concurrency: int = 64, timeout_seconds: float = 10):
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency должен быть положительным целым числом.")
        self._url, self._concurrency, self._timeout = base_url, concurrency, timeout_seconds
        self._clients = {}
        self._lock = asyncio.Lock()
        self._stack = AsyncExitStack()
        self._http = None
        self._closed = False

    async def __aenter__(self):
        if self._http is not None or self._closed:
            raise SearchProtocolError("SQLClientPool уже открыт или закрыт.")
        self._http = httpx.AsyncClient(timeout=self._timeout, trust_env=False, follow_redirects=False,
            limits=httpx.Limits(max_connections=self._concurrency, max_keepalive_connections=self._concurrency))
        return self

    async def __aexit__(self, *exc):
        self._closed = True
        async with self._lock:
            try:
                await self._stack.aclose()
            finally:
                if self._http is not None:
                    await self._http.aclose()

    def _check_open(self):
        if self._http is None or self._closed:
            raise SearchProtocolError("Используйте SQLClientPool через async with.")

    async def databases(self) -> list[dict]:
        self._check_open()
        return await SQLClient.list_databases(self._url, http_client=self._http, timeout_seconds=self._timeout)

    async def bind(self, db_id: str, db_revision: str) -> SQLClient:
        self._check_open()
        if not isinstance(db_id, str) or not db_id or not isinstance(db_revision, str) or not db_revision:
            raise ValueError("Для задачи нужны явные db_id и db_revision.")
        key = (db_id, db_revision)
        async with self._lock:
            self._check_open()
            if key not in self._clients:
                client = SQLClient(self._url, db_id=db_id, db_revision=db_revision,
                                   http_client=self._http, timeout_seconds=self._timeout)
                self._clients[key] = await self._stack.enter_async_context(client)
            return self._clients[key]

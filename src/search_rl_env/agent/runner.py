"""Ограниченный пул rollout с переиспользованием транспорта SDK."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Literal

import httpx
from openai import AsyncOpenAI

from search_rl_env.contracts.types import AgentTask, ChatModel, EpisodeConfig, EpisodeResult, SearchBackend, SqlBackend

from .sdk import dispatch_completion, run_sdk_episode


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    task: AgentTask
    config: EpisodeConfig | None = None
    reference_document_ids: list[str] | None = None
    reference_answer: list[str] | None = None


class RolloutRunner:
    """Ресурсы живут один async with; результаты — в порядке готовности.

    model/search/sql заимствуются: runner не закрывает внешние клиенты.
    Бэкенды должны поддерживать конкурентные вызовы без состояния диалога.
    Ограничение concurrency действует и на несколько одновременных batch/run.
    """

    def __init__(self, model: ChatModel | None = None, search: SearchBackend | None = None,
                 config: EpisodeConfig | None = None, *, sql: SqlBackend | None = None, concurrency: int = 8,
                 execution: Literal["in_process"] = "in_process"):
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError('concurrency должен быть положительным целым числом.')
        if execution != "in_process":
            raise ValueError("Agents SDK поддерживает execution='in_process'.")
        self.model, self.search, self.sql = model, search, sql
        self.config = config or EpisodeConfig()
        self.concurrency = concurrency
        self._gate = asyncio.Semaphore(concurrency)
        self._active, self._workers = set(), set()
        self._client = None
        self._closed = False

    async def __aenter__(self):
        if self._client is not None or self._closed:
            raise RuntimeError('Runner уже запущен или закрыт.')
        self._client = AsyncOpenAI(
            api_key="local-sdk-adapter",
            base_url="https://agent-transport.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch_completion)),
            max_retries=0,
        )
        return self

    async def __aexit__(self, *exc):
        self._closed = True
        tasks = (self._active | self._workers) - {asyncio.current_task()}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._client.close()

    async def run(self, task: AgentTask, model=None, search=None, config=None, *, sql=None,
                  on_event=None, on_interrupted=None, reference_document_ids=None, reference_answer=None) -> EpisodeResult:
        if self._client is None or self._closed:
            raise RuntimeError('Используйте async with RolloutRunner(...).')
        model = self.model if model is None else model
        search = self.search if search is None else search
        if model is None or search is None:
            raise ValueError('Нужны model и search в runner или run().')
        operation = asyncio.current_task()
        self._active.add(operation)
        try:
            async with self._gate:
                return await run_sdk_episode(task, model, search, config or self.config, client=self._client,
                    sql=self.sql if sql is None else sql, on_event=on_event, on_interrupted=on_interrupted,
                    reference_document_ids=reference_document_ids, reference_answer=reference_answer)
        finally:
            self._active.discard(operation)

    async def iter_episodes(self, requests: Iterable[AgentTask | RolloutRequest] | AsyncIterable[AgentTask | RolloutRequest],
                            *, on_event=None) -> AsyncIterator[EpisodeResult]:
        """O(concurrency) задач и буфера; медленный consumer тормозит выдачу новых работ.

        Закрывайте iterator через aclose()/contextlib.aclosing при досрочном выходе.
        Это отменяет незавершённые работы этого batch, не затрагивая другие batch.
        """
        if self._client is None or self._closed:
            raise RuntimeError('Используйте async with RolloutRunner(...).')
        asynchronous = isinstance(requests, AsyncIterable)
        source = aiter(requests) if asynchronous else iter(requests)
        source_lock = asyncio.Lock()
        queue = asyncio.Queue(maxsize=self.concurrency)
        done = object()

        async def worker():
            while True:
                try:
                    async with source_lock:
                        try:
                            item = await anext(source) if asynchronous else next(source)
                        except (StopIteration, StopAsyncIteration):
                            break
                    request = item if isinstance(item, RolloutRequest) else RolloutRequest(item)
                    result = await self.run(request.task, config=request.config, on_event=on_event,
                        reference_document_ids=request.reference_document_ids, reference_answer=request.reference_answer)
                    await queue.put((result, None))
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    await queue.put((None, error))
                    return
            await queue.put((done, None))

        workers = {asyncio.create_task(worker(), name=f'rollout-worker-{index}')
                   for index in range(self.concurrency)}
        self._workers.update(workers)
        try:
            remaining = len(workers)
            while remaining:
                result, error = await queue.get()
                if error is not None:
                    raise error
                if result is done:
                    remaining -= 1
                else:
                    yield result
        finally:
            for operation in workers:
                operation.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self._workers.difference_update(workers)
            if asynchronous and hasattr(source, 'aclose'):
                await source.aclose()

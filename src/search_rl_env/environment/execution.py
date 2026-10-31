"""Вызовы доверенных инструментов без отдельного MCP-процесса на каждый rollout."""

from __future__ import annotations

import os
from time import perf_counter

import aiohttp

from search_rl_env.agent.tools import ToolCall, execute_tool
from search_rl_env.contracts.types import SearchHit, SqlQueryError, SqlResult, ToolExecution


class ToolExecutor:
    def __init__(self, task, config, state, *, search=None, sql=None, token=None):
        self.task, self.config, self.state = task, config, state
        self.search_backend, self.sql_backend = search, sql
        self.token = token or os.environ.get('SEARCH_RL_BRIDGE_TOKEN', '')
        self.http = None

    async def __aenter__(self):
        if self.config.backend_url and (self.search_backend is None or
                                       self.config.enable_sql and self.sql_backend is None):
            self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        if self.config.enable_sql and self.sql_backend is None and not self.config.backend_url:
            from .sql import MockSql
            self.sql_backend = MockSql()
        return self

    async def __aexit__(self, *exc):
        if self.http is not None:
            await self.http.close()

    async def execute(self, raw_call):
        call = ToolCall.model_validate(raw_call)
        started = perf_counter()
        result = await execute_tool(call, task=self.task, search=self, config=self.config,
                                    state=self.state, sql=self if self.config.enable_sql else None)
        self.state.pending_calls = [item for item in self.state.pending_calls if item['id'] != call.id]
        self.state.actions.append(ToolExecution(
            tool_call_id=call.id, name=call.function.name, arguments=call.function.arguments,
            result=result, duration_seconds=perf_counter() - started,
        ))
        return result

    async def _post(self, route, payload):
        async with self.http.post(self.config.backend_url + route,
                                  headers={'Authorization': 'Bearer ' + self.token}, json=payload) as response:
            response.raise_for_status()
            return await response.json()

    async def search(self, query, *, corpus_id, corpus_revision, limit):
        if self.search_backend is not None:
            return await self.search_backend.search(query, corpus_id=corpus_id,
                                                    corpus_revision=corpus_revision, limit=limit)
        if self.config.backend_url is None:
            return self.config.hits[:limit]
        data = await self._post('/search', dict(query=query, corpus_id=corpus_id,
                                               corpus_revision=corpus_revision, limit=limit))
        return [SearchHit.model_validate(item) for item in data]

    async def query(self, query):
        if self.sql_backend is not None:
            return await self.sql_backend.query(query)
        data = await self._post('/sql', {'query': query})
        if 'query_error' in data:
            raise SqlQueryError(data['query_error'])
        return SqlResult.model_validate(data)

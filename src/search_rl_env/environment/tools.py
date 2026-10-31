"""MCP-инструменты задачи; изменяемое состояние синхронизирует Verifiers."""

import json
import os
from time import perf_counter
from typing import Literal

import httpx
import verifiers.v1 as vf
from pydantic import Field

from search_rl_env.agent.tools import ToolCall, execute_tool
from search_rl_env.contracts.types import SearchHit, ToolExecution, SqlResult, SqlQueryError
from .state import SearchState


class SearchToolsetConfig(vf.ToolsetConfig):
    execution: Literal['in_process', 'mcp'] = 'in_process'
    colocated: bool = True
    backend_url: str | None = None
    search_limit: int = Field(default=10, ge=1)
    max_snippet_chars: int = Field(default=2000, ge=1)
    hits: list[SearchHit] = Field(default_factory=list)
    enable_sql: bool = False


class SearchToolset(vf.Toolset[SearchToolsetConfig, SearchState]):
    TOOL_PREFIX = None

    async def setup_task(self, task):
        self.task = task.agent_task

    async def _execute(self, name, arguments):
        def matches(call):
            try:
                return call['function']['name'] == name and json.loads(call['function']['arguments']) == arguments
            except ValueError:
                return False

        pending = next((call for call in self.state.pending_calls if matches(call)), None)
        if pending is None:
            raise ValueError("Нет вызова инструмента в состоянии эпизода.")
        self.state.pending_calls.remove(pending)
        call = ToolCall.model_validate(pending)
        started = perf_counter()
        backend = BackendView(self)
        result = await execute_tool(call, task=self.task, search=backend, config=self.config,
                                    state=self.state, sql=backend if self.config.enable_sql else None)
        self.state.actions.append(ToolExecution(
            tool_call_id=call.id, name=name, arguments=call.function.arguments,
            result=result, duration_seconds=perf_counter() - started,
        ))
        return result

    @vf.tool
    async def search(self, query: str) -> dict:
        """Найти документы по запросу в корпусе задачи."""
        return await self._execute('search', {'query': query})

    @vf.tool
    async def finish(self, document_ids: list[str]) -> dict:
        """Завершить с ранжированными id ранее показанных документов."""
        return await self._execute('finish', {'document_ids': document_ids})

    # SearchBackend для общей проверки и обрезки выдачи в execute_tool.
    async def search_backend(self, query, *, corpus_id, corpus_revision, limit):
        if self.config.backend_url is None:
            return self.config.hits[:limit]
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.config.backend_url + '/search',
                headers={'Authorization': 'Bearer ' + os.environ.get('SEARCH_RL_BRIDGE_TOKEN', '')},
                json={'query': query, 'corpus_id': corpus_id,
                      'corpus_revision': corpus_revision, 'limit': limit})
            response.raise_for_status()
            return [SearchHit.model_validate(item) for item in response.json()]

    async def sql_backend(self, query):
        if self.config.backend_url is None:
            from .sql import MockSql
            return await MockSql().query(query)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.config.backend_url + '/sql',
                headers={'Authorization': 'Bearer ' + os.environ.get('SEARCH_RL_BRIDGE_TOKEN', '')},
                json={'query': query})
            response.raise_for_status()
            data = response.json()
            if 'query_error' in data:
                raise SqlQueryError(data['query_error'])
            return SqlResult.model_validate(data)


class BackendView:
    def __init__(self, tools):
        self.tools = tools

    async def search(self, *args, **kwargs):
        return await self.tools.search_backend(*args, **kwargs)

    async def query(self, query):
        return await self.tools.sql_backend(query)


if __name__ == '__main__':
    SearchToolset.run()

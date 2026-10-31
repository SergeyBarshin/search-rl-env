"""Детерминированные замены модели и поиска для интеграции без сети."""

import json
from typing import Any

from search_rl_env.contracts.types import ModelResponse, SearchHit, Usage


class MockSearch:
    def __init__(self, hits: list[SearchHit], *, corpus_id: str, corpus_revision: str | None = None) -> None:
        self.hits = list(hits)
        self.corpus_id = corpus_id
        self.corpus_revision = corpus_revision
        self.calls: list[dict[str, Any]] = []

    async def search(
        self, query: str, *, corpus_id: str, corpus_revision: str | None, limit: int
    ) -> list[SearchHit]:
        if (corpus_id, corpus_revision) != (self.corpus_id, self.corpus_revision):
            raise ValueError("Неизвестный корпус или ревизия.")
        self.calls.append({"query": query, "corpus_id": corpus_id, "corpus_revision": corpus_revision, "limit": limit})
        return self.hits[:limit]


class MockModel:
    @property
    def metadata(self) -> dict[str, Any]:
        return {"adapter": "mock", "model": "search-then-finish-v1"}

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        seed: int | None,
    ) -> ModelResponse:
        results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if not results:
            task = json.loads(messages[1]["content"])
            name, arguments = "search", {"query": task["question"]}
        else:
            hits = [hit for result in results for hit in result.get("hits", [])]
            ids = list(dict.fromkeys(hit["doc_id"] for hit in hits))
            name, arguments = "finish", {"document_ids": ids}
        return ModelResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"mock-call-{len(results) + 1}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }],
            },
            usage=Usage(prompt_tokens=0, completion_tokens=min(1, max_output_tokens), total_tokens=min(1, max_output_tokens), cost=0.0),
            finish_reason="tool_calls",
            model="search-then-finish-v1",
        )


class MockSqlRagModel(MockModel):
    @property
    def metadata(self):
        return {"adapter": "mock", "model": "search-sql-finish-v1"}

    async def complete(self, messages, *, tools, max_output_tokens, seed):
        results = [json.loads(message['content']) for message in messages if message['role'] == 'tool']
        table = next((result for result in results if result.get('ok') and 'columns' in result), None)
        if table is not None:
            column = table['columns'].index('order_id')
            answer = [row[column] for row in table['rows']]
            hits = [hit for result in results for hit in result.get('hits', [])]
            name = 'finish'
            arguments = {'answer': answer, 'document_ids': [hits[0]['doc_id']] if hits else []}
        elif results:
            name = 'sql'
            arguments = {'query': "SELECT order_id FROM orders WHERE region = 'RU' AND status = 'delivered' AND delay_days > 3 ORDER BY order_id"}
        else:
            name = 'search'
            arguments = {'query': 'регламент компенсаций заказов'}
        return ModelResponse(
            message={'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': f'hybrid-call-{len(results) + 1}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)},
            }]},
            usage=Usage(prompt_tokens=0, completion_tokens=1, total_tokens=1, cost=0.0),
            finish_reason='tool_calls', model=self.metadata['model'],
        )

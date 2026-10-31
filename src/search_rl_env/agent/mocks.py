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

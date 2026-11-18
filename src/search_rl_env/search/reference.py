"""Адаптер общего HTTP-индекса для нашего агента и T-Search."""

from copy import deepcopy

import httpx
from pydantic import BaseModel, Field

from search_rl_env.contracts import SearchHit

from .models import SearchProtocolError, UnknownId


class ReferenceManifest(BaseModel):
    corpus_id: str = Field(min_length=1)
    corpus_revision: str = Field(min_length=1)
    search_revision: str = Field(min_length=1)


class ReferenceHit(BaseModel):
    doc_id: str = Field(min_length=1)
    snippet: str


class ReferenceResult(BaseModel):
    result: list[ReferenceHit]


class ReferenceSearch:
    def __init__(self, http: httpx.AsyncClient, base_url: str, manifest: dict):
        self._identity = ReferenceManifest.model_validate(manifest)
        self._manifest = deepcopy(manifest)
        self._http = http
        self._endpoint = base_url.rstrip("/") + "/search"

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    async def search(self, query, *, corpus_id, corpus_revision, limit):
        if (corpus_id, corpus_revision) != (self._identity.corpus_id, self._identity.corpus_revision):
            raise UnknownId("Другой корпус или ревизия reference-индекса.")
        response = await self._http.post(self._endpoint, json={"query": query, "top_k": limit})
        response.raise_for_status()
        hits = ReferenceResult.model_validate_json(response.content).result
        if len(hits) > limit or len({hit.doc_id for hit in hits}) != len(hits):
            raise SearchProtocolError("Размер или ID выдачи не соответствуют контракту общего индекса.")
        revision = response.headers.get("x-search-revision")
        if revision is not None and revision != self._identity.search_revision:
            raise SearchProtocolError("Ревизия reference-индекса изменилась.")
        return [SearchHit(doc_id=hit.doc_id, snippet=hit.snippet, rank=rank) for rank, hit in enumerate(hits, 1)]

import json
from pathlib import Path

import httpx
import pytest

from search_rl_env.contracts.types import ModelTimeoutError, ModelTransportError
from search_rl_env.search.acceptance import ReservedBudgetModel
from search_rl_env.search.models import SearchProtocolError
from search_rl_env.search.reference import ReferenceSearch
from search_rl_env.search.snippets import SnippetTokenSearch


async def test_reference_search_preserves_raw_chunk_ids_and_checks_revision():
    revision = "version-1"

    def serve(request):
        assert request.url.path == "/search"
        assert json.loads(request.content) == {"query": "Oslo", "top_k": 2}
        return httpx.Response(200, headers={"x-search-revision": revision}, json={"result": [
            {"doc_id": "12_1", "snippet": "Oslo", "score": 0.9},
            {"doc_id": "12_0", "snippet": "Norway", "score": 0.8},
        ]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        search = ReferenceSearch(http, "http://index.test", {
            "corpus_id": "trust", "corpus_revision": "corpus-1", "search_revision": revision,
        })
        hits = await search.search("Oslo", corpus_id="trust", corpus_revision="corpus-1", limit=2)
        assert [(hit.doc_id, hit.rank) for hit in hits] == [("12_1", 1), ("12_0", 2)]

        revision = "version-2"
        with pytest.raises(SearchProtocolError, match="Ревизия"):
            await search.search("Oslo", corpus_id="trust", corpus_revision="corpus-1", limit=2)


@pytest.mark.parametrize(("http_error", "model_error"), [
    (httpx.ReadTimeout, ModelTimeoutError),
    (httpx.ReadError, ModelTransportError),
])
async def test_qwen_reference_can_continue_after_transport_failure(monkeypatch, http_error, model_error):
    pytest.importorskip("numpy")
    examples = Path(__file__).resolve().parents[3] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    from serve_browsecomp_dense import DIMENSIONS, MODEL, QUERY_PREFIX, ElizaQueryEncoder

    requests = []

    def serve(request):
        requests.append(request)
        if len(requests) == 1:
            raise http_error("transport failure", request=request)
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        assert payload["input"] == QUERY_PREFIX + "Oslo"
        return httpx.Response(200, json={
            "model": "Qwen/Qwen3-Embedding-8B",
            "data": [{"index": 0, "embedding": [1.0] + [0.0] * (DIMENSIONS - 1)}],
            "usage": {"prompt_tokens": 20, "total_tokens": 20, "cost": 0.0000002},
        })

    async with httpx.AsyncClient(base_url="https://index.test/v1/", transport=httpx.MockTransport(serve)) as http:
        encoder = ElizaQueryEncoder(http, "model-case-test")
        billing = ReservedBudgetModel(encoder, 0.012, 0.004, recoverable_errors=(ModelTransportError,))
        with pytest.raises(model_error) as caught:
            await billing.complete("Oslo")
        assert type(caught.value) is model_error
        result = await billing.complete("Oslo")

    assert result.vector.shape == (1, DIMENSIONS)
    assert result.usage.cost == 0.0000002
    assert encoder.reported_cost_usd == result.usage.cost
    assert billing.reserved == pytest.approx(0.004)


async def test_token_limit_preserves_retrieval_order_and_document_prefix(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "река": 1, "桥": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    text = "река 桥 " * 150

    def serve(_request):
        return httpx.Response(200, json={"result": [
            {"doc_id": "d1", "snippet": text}, {"doc_id": "d2", "snippet": "река"},
        ]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        reference = ReferenceSearch(http, "http://index.test", {
            "corpus_id": "fixture", "corpus_revision": "v1", "search_revision": "v1",
        })
        search = SnippetTokenSearch(reference, tokenizer_path, 256)
        hits = await search.search("река", corpus_id="fixture", corpus_revision="v1", limit=2)

    assert [(hit.doc_id, hit.rank) for hit in hits] == [("d1", 1), ("d2", 2)]
    assert text.startswith(hits[0].snippet) and len(tokenizer.encode(hits[0].snippet).ids) == 256
    assert hits[1].snippet == "река"

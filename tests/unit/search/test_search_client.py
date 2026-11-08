import asyncio
import json

import httpx
import pytest

from search_rl_env.search import (InvalidRequest, SearchBusy, SearchClient,
    SearchProtocolError, SearchTimeout, StorageError, UnknownId)


INFO = {
    "schema_version": 2, "corpus_id": "demo", "corpus_revision": "v1", "content_hash": "a" * 64,
    "chunk_chars": 2000, "chunk_algorithm": "chars-v1", "documents": 1,
    "index_algorithm": "go-fts5-unicode61-document-bm25-v1", "sqlite_version": "3.51.2",
    "driver_version": "modernc.org/sqlite@v1.44.3", "index_revision": "b" * 64,
    "api_version": "1", "search_revision": "c" * 64,
    "config": {"max_results": 100, "snippet_chars": 500, "max_read_chars": 8000,
        "max_query_chars": 2000, "max_query_terms": 32, "workers": 4, "max_pending": 32,
        "timeout_nanoseconds": 5_000_000_000},
}
VERSIONS = {key: INFO[key] for key in ("corpus_id", "corpus_revision", "index_revision", "search_revision")}
HIT = {"doc_id": "doc", "snippet": "Original text.", "rank": 1, "score": -1.0}
RESULT = {**VERSIONS, "hits": [HIT]}


async def test_protocol_pins_snapshot_and_keeps_external_client_open():
    calls = []
    def handler(request):
        calls.append(request)
        if request.url.path.endswith("manifest"):
            return httpx.Response(200, json=INFO)
        payload = json.loads(request.content)
        assert payload["corpus_revision"] == "v1"
        return httpx.Response(200, json=RESULT)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with SearchClient("http://search", http_client=http) as search:
            hits = await search.search("text", corpus_id="demo", corpus_revision=None, limit=10)
            assert hits[0].model_dump() == {"doc_id": "doc", "snippet": "Original text.", "rank": 1}
            assert search.manifest == INFO
            with pytest.raises(UnknownId):
                await search.search("text", corpus_id="wrong", corpus_revision=None, limit=10)
        assert not http.is_closed
    assert len(calls) == 2


@pytest.mark.parametrize("status,code,error", [(400, "invalid_request", InvalidRequest),
    (404, "unknown_id", UnknownId), (503, "busy", SearchBusy), (504, "timeout", SearchTimeout),
    (500, "storage_error", StorageError), (408, "cancelled", SearchTimeout)])
async def test_errors_sanitized_without_retry(status, code, error):
    calls = []
    def handler(request):
        calls.append(request)
        if request.url.path.endswith("manifest"):
            return httpx.Response(200, json=INFO)
        return httpx.Response(status, json={"error": {"code": code, "message": "private-secret-body"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with SearchClient("http://search", http_client=http) as search:
            with pytest.raises(error) as caught:
                await search.search("private-query", corpus_id="demo", corpus_revision=None, limit=10)
            assert "private" not in str(caught.value)
    assert len(calls) == 2


@pytest.mark.parametrize("result", [
    {**RESULT, "corpus_revision": "other"}, {**RESULT, "index_revision": "other"},
    {**RESULT, "search_revision": "other"}, {**RESULT, "hits": [HIT, HIT]},
    {**RESULT, "hits": [{**HIT, "rank": 2}]}, {**RESULT, "hits": [{**HIT, "snippet": "x" * 501}]},
    {**RESULT, "unknown": True}, {**RESULT, "hits": [{**HIT, "score": float("inf")}]},
])
async def test_rejects_changed_environment_and_invalid_results(result):
    def handler(request):
        data = INFO if request.url.path.endswith("manifest") else result
        return httpx.Response(200, content=json.dumps(data).encode())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with SearchClient("http://search", http_client=http) as search:
            with pytest.raises(SearchProtocolError):
                await search.search("text", corpus_id="demo", corpus_revision=None, limit=10)


async def test_stream_deadline_and_cancellation():
    started = asyncio.Event()
    async def handler(request):
        if request.url.path.endswith("manifest"):
            return httpx.Response(200, json=INFO)
        started.set()
        await asyncio.Event().wait()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with SearchClient("http://search", timeout_seconds=0.02, http_client=http) as search:
            with pytest.raises(SearchTimeout):
                await search.search("text", corpus_id="demo", corpus_revision=None, limit=10)
        async with SearchClient("http://search", http_client=http) as search:
            started.clear()
            task = asyncio.create_task(search.search("text", corpus_id="demo", corpus_revision=None, limit=10))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("body,status", [(b"not json", 200), (b"private-error", 502),
    (b'{"error":{"code":"unknown","message":"secret"}}', 500),
    (b'{"error":{"code":"busy","message":"secret"}}', 400)])
async def test_malformed_response(body, status):
    def handler(request):
        return httpx.Response(200, json=INFO) if request.url.path.endswith("manifest") else httpx.Response(status, content=body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with SearchClient("http://search", http_client=http) as search:
            with pytest.raises(SearchProtocolError):
                await search.search("text", corpus_id="demo", corpus_revision=None, limit=10)


async def test_oversized_manifest_and_network_error():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b" " * 100))) as http:
        with pytest.raises(SearchProtocolError):
            async with SearchClient("http://search", max_response_bytes=50, http_client=http):
                pass
    def handler(request):
        raise httpx.ConnectError("private-server-address")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(StorageError) as caught:
            async with SearchClient("http://search", http_client=http):
                pass
        assert "private" not in str(caught.value)


async def test_reads_validate_identity_offsets_and_versions():
    response = {**VERSIONS, "doc_id": "doc", "chunk_id": None, "text": "Ori", "offset": 0, "next_offset": 3, "total_chars": 14}
    def handler(request):
        return httpx.Response(200, json=INFO if request.url.path.endswith("manifest") else response)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with SearchClient("http://search", http_client=http) as search:
            assert (await search.read(doc_id="doc", limit=3)).text == "Ori"
            response["next_offset"] = None
            with pytest.raises(SearchProtocolError):
                await search.read(doc_id="doc", limit=3)


@pytest.mark.parametrize("url", ["file:///tmp/index", "http://user:secret@search", "http://search?key=value"])
def test_rejects_invalid_service_address(url):
    with pytest.raises(ValueError):
        SearchClient(url)

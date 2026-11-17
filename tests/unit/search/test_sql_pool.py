import asyncio
import json

import httpx
import pytest

from search_rl_env.search import SQLClientPool, SearchProtocolError, SearchTimeout
from search_rl_env.search import sql_pool


def install_http(monkeypatch, handler):
    original = httpx.AsyncClient
    clients = []
    def factory(**kwargs):
        client = original(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client
    monkeypatch.setattr(sql_pool.httpx, "AsyncClient", factory)
    return clients


async def test_pool_has_fixed_bindings_one_schema_fetch_and_one_http_pool(monkeypatch):
    catalog = [{"db_id": "a", "db_revision": "a" * 64}, {"db_id": "b", "db_revision": "b" * 64}]
    schema_calls = []
    def handler(request):
        if request.url.path.endswith("databases"):
            return httpx.Response(200, json={"databases": catalog})
        if request.method == "GET":
            identifier = request.url.params["db_id"]
            revision = request.url.params["db_revision"]
            schema_calls.append(identifier)
            return httpx.Response(200, json={"db_id": identifier, "db_revision": revision,
                                            "schema": f"CREATE TABLE {identifier}(value TEXT);"})
        payload = json.loads(request.content)
        return httpx.Response(200, json={"db_id": payload["db_id"], "db_revision": payload["db_revision"],
                                        "columns": ["value"], "rows": [[payload["db_id"]]], "truncated": False})
    clients = install_http(monkeypatch, handler)
    async with SQLClientPool("http://backend", concurrency=64) as pool:
        assert await pool.databases() == catalog
        first = await asyncio.gather(*(pool.bind("a", "a" * 64) for _ in range(16)))
        assert all(binding is first[0] for binding in first)
        second = await pool.bind("b", "b" * 64)
        assert first[0].schema != second.schema
        async def query(index):
            sql = first[0] if index % 2 else second
            result = await sql.query("SELECT value")
            assert result.rows == [["a" if index % 2 else "b"]]
        await asyncio.gather(*(query(index) for index in range(64)))
    assert schema_calls == ["a", "b"]
    assert len(clients) == 1 and clients[0].is_closed
    with pytest.raises(SearchProtocolError):
        await pool.bind("a", "a" * 64)


async def test_pool_closes_http_when_binding_fails(monkeypatch):
    clients = install_http(monkeypatch, lambda request: httpx.Response(504,
        json={"error": {"code": "timeout", "message": "safe"}}))
    with pytest.raises(SearchTimeout):
        async with SQLClientPool("http://backend") as pool:
            await pool.bind("a", "a" * 64)
    assert clients[0].is_closed


async def test_pool_rejects_schema_of_other_database(monkeypatch):
    install_http(monkeypatch, lambda request: httpx.Response(200, json={
        "db_id": "wrong", "db_revision": "a" * 64, "schema": "CREATE TABLE x(a);"}))
    async with SQLClientPool("http://backend") as pool:
        with pytest.raises(SearchProtocolError):
            await pool.bind("a", "a" * 64)

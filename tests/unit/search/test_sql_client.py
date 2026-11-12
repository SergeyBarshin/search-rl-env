import httpx
import pytest

from search_rl_env.contracts import SqlQueryError
from search_rl_env.search import SQLClient, SearchProtocolError


INFO = {"db_id": "hockey", "db_revision": "a" * 64, "schema": "CREATE TABLE teams(name TEXT);"}
RESULT = {"db_id": "hockey", "db_revision": "a" * 64,
          "columns": ["name", "name"], "rows": [["Toronto Arenas", "Toronto Arenas"]], "truncated": False}


async def test_sql_adapter_preserves_schema_duplicate_columns_and_rows():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json=INFO if request.method == "GET" else RESULT))) as http:
        async with SQLClient("http://search", http_client=http) as sql:
            assert sql.schema == INFO["schema"]
            assert sql.info == INFO
            result = await sql.query("SELECT name,name FROM teams")
            assert result.columns == ["name", "name"]
            assert result.rows == [["Toronto Arenas", "Toronto Arenas"]]
        assert not http.is_closed


async def test_sql_error_is_recoverable_without_raw_server_message():
    def handler(request):
        return httpx.Response(200, json=INFO) if request.method == "GET" else httpx.Response(400,
            json={"error": {"code": "sql_query_error", "message": "private path and SQL"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with SQLClient("http://search", http_client=http) as sql:
            with pytest.raises(SqlQueryError) as caught:
                await sql.query("DELETE FROM teams")
            assert "private" not in str(caught.value)


async def test_sql_revision_cannot_change_during_run():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json=INFO if request.method == "GET" else {**RESULT, "db_revision": "b" * 64}))) as http:
        async with SQLClient("http://search", http_client=http) as sql:
            with pytest.raises(SearchProtocolError):
                await sql.query("SELECT name FROM teams")

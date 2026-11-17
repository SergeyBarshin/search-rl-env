import asyncio
from contextlib import closing
import shutil
import sqlite3
import threading

import pytest

from search_rl_env.contracts import SqlQueryError
from search_rl_env.search.local import SQLiteSnapshot, digest_file
from search_rl_env.search.models import StorageError


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(path)) as db:
        db.executescript("CREATE TABLE facts(value TEXT); INSERT INTO facts VALUES ('a-b'), ('second'), ('third');")
        db.commit()
    return path


@pytest.fixture
async def sql(source):
    backend = SQLiteSnapshot(source, "facts", digest_file(source))
    yield backend
    await backend.aclose()


async def test_readonly_functions_identifiers_and_duplicate_columns(sql):
    result = await sql.query('WITH "delete" AS (SELECT value AS "update" FROM facts) '
                             'SELECT replace("update", \'-\', \'\') AS "drop", "update" AS "drop" FROM "delete"')
    assert result.columns == ["drop", "drop"]
    assert result.rows == [["ab", "a-b"], ["second", "second"], ["third", "third"]]
    assert (await sql.query("SELECT 'pragma_database_list' AS value")).rows == [["pragma_database_list"]]


@pytest.mark.parametrize("query", [
    "DELETE FROM facts", "UPDATE facts SET value='changed'", "DROP TABLE facts",
    "ATTACH ':memory:' AS other", "PRAGMA database_list", "SELECT file FROM pragma_database_list",
    "SELECT file FROM 'pragma_database_list'", 'SELECT file FROM "pragma_database_list"',
    "SELECT file FROM `pragma_database_list`", "SELECT file FROM [pragma_database_list]",
    "WITH p AS (SELECT * FROM 'pragma_database_list') SELECT * FROM p",
    "SELECT 1 FROM facts, 'pragma_database_list'", "SELECT 1; DELETE FROM facts",
    "SELECT load_extension('private-path')", "SELECT randomblob(20)", "SELECT 1e999",
])
async def test_rejects_unsafe_sql_without_mutating_snapshot(sql, query):
    with pytest.raises(SqlQueryError) as error:
        await sql.query(query)
    assert "private-path" not in str(error.value)
    assert (await sql.query("SELECT count(*) FROM facts")).rows == [[3]]


async def test_result_and_vm_budgets(source):
    sql = SQLiteSnapshot(source, "facts", digest_file(source), max_rows=2, max_vm_steps=1000)
    try:
        result = await sql.query("SELECT value FROM facts")
        assert result.rows == [["a-b"], ["second"]] and result.truncated
        with pytest.raises(SqlQueryError):
            await sql.query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n")
        assert (await sql.query("SELECT 1")).rows == [[1]]
    finally:
        await sql.aclose()
    with pytest.raises(StorageError):
        await sql.query("SELECT 1")


async def test_cancellation_joins_sql_worker_and_allows_next_query(source, monkeypatch):
    sql = SQLiteSnapshot(source, "facts", digest_file(source), max_vm_steps=10**12, timeout_seconds=5)
    entered = threading.Event()
    original = sql._query

    def observe(*args):
        entered.set()
        return original(*args)

    monkeypatch.setattr(sql, "_query", observe)
    try:
        operation = asyncio.create_task(sql.query(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n"))
        assert await asyncio.to_thread(entered.wait, 1)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 1)
        assert (await sql.query("SELECT count(*) FROM facts")).rows == [[3]]
    finally:
        await sql.aclose()


def test_rejects_snapshot_with_uncheckpointed_wal(tmp_path):
    source = tmp_path / "writer.sqlite"
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.executescript("CREATE TABLE facts(value TEXT); INSERT INTO facts VALUES ('base');")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        revisions = []
        for value in ["first", "second"]:
            writer.execute("UPDATE facts SET value=?", (value,))
            writer.commit()
            path = tmp_path / (value + ".sqlite")
            shutil.copyfile(source, path)
            shutil.copyfile(str(source) + "-wal", str(path) + "-wal")
            revisions.append(digest_file(path))
            with pytest.raises(StorageError, match="WAL"):
                SQLiteSnapshot(path, "facts", digest_file(path))
        assert revisions[0] == revisions[1]


async def test_rejects_changed_file_and_wrong_revision(source):
    with pytest.raises(StorageError):
        SQLiteSnapshot(source, "facts", "0" * 64)
    sql = SQLiteSnapshot(source, "facts", digest_file(source))
    try:
        with source.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(StorageError):
            await sql.query("SELECT value FROM facts")
    finally:
        await sql.aclose()


async def test_forty_parallel_sql_snapshots_do_not_share_rows(source, tmp_path):
    barrier = asyncio.Barrier(40)
    backends = []

    async def episode(index):
        path = tmp_path / f"episode-{index}.sqlite"
        shutil.copyfile(source, path)
        with closing(sqlite3.connect(path)) as writer:
            writer.execute("UPDATE facts SET value=?", (str(index),))
            writer.commit()
        backend = SQLiteSnapshot(path, "facts", digest_file(path))
        backends.append(backend)
        try:
            await barrier.wait()
            result = await backend.query("SELECT DISTINCT value FROM facts")
            assert result.rows == [[str(index)]]
        finally:
            await backend.aclose()
            path.unlink()

    await asyncio.gather(*(episode(index) for index in range(40)))
    assert len(backends) == 40 and all(backend.closed for backend in backends)

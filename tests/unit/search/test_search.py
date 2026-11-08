import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import sqlite3
import threading

import pytest

from search_rl_env.search import (
    Document, HarnessSearch, InvalidRequest, LocalSearch, SearchBusy, SearchConfig,
    SearchTimeout, SnapshotConflict, StorageError, UnknownId, build_snapshot,
)


DOCUMENTS = [
    Document("ru", "Поиск документов. Срок проекта перенесён на ноябрь. Русский текст 😀."),
    Document("en", "Equipment delivery delayed. The release was postponed until November."),
    Document("other", "The library opens on Monday."),
]


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "corpus.sqlite"
    build_snapshot(DOCUMENTS, path, corpus_id="demo", chunk_chars=17)
    return path


def test_rebuild_reordering_and_conflict(snapshot, tmp_path):
    before = snapshot.read_bytes()
    manifest = build_snapshot(reversed(DOCUMENTS), snapshot, corpus_id="demo", chunk_chars=17)
    other = build_snapshot(reversed(DOCUMENTS), tmp_path / "other.sqlite", corpus_id="demo", chunk_chars=17)
    assert manifest == other
    assert snapshot.read_bytes() == before
    with pytest.raises(SnapshotConflict):
        build_snapshot([Document("ru", "Новый текст")], snapshot, corpus_id="demo")
    assert snapshot.read_bytes() == before


def test_versions_separate_content_and_chunking(snapshot, tmp_path):
    with sqlite3.connect(snapshot) as db:
        first = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
    second = build_snapshot(DOCUMENTS, tmp_path / "chunks.sqlite", corpus_id="demo", chunk_chars=9)
    third = build_snapshot(DOCUMENTS[:1], tmp_path / "smaller.sqlite", corpus_id="demo")
    assert first["corpus_revision"] == second["corpus_revision"]
    assert first["index_revision"] != second["index_revision"]
    assert third["corpus_revision"] != first["corpus_revision"]


@pytest.mark.parametrize("documents", [[], [Document("x", "")], [Document("", "text")],
    [Document("x", "a"), Document("x", "b")], [Document("x", "a\x00b")],
    [Document("x", "\ud800")], [Document("x", 123)]])
def test_invalid_build_never_publishes(documents, tmp_path):
    path = tmp_path / "bad.sqlite"
    with pytest.raises(InvalidRequest):
        build_snapshot(documents, path, corpus_id="demo")
    assert not path.exists()
    assert not list(tmp_path.glob(".search-build-*"))


def test_failed_input_does_not_publish(tmp_path):
    def broken():
        yield Document("a", "valid")
        raise RuntimeError("source failed")
    with pytest.raises(RuntimeError):
        build_snapshot(broken(), tmp_path / "bad.sqlite", corpus_id="demo")
    assert not list(tmp_path.iterdir())


def test_concurrent_builders_publish_same_snapshot(tmp_path):
    path = tmp_path / "same.sqlite"
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(build_snapshot, DOCUMENTS, path, corpus_id="demo") for _ in range(2)]
        assert futures[0].result() == futures[1].result()


async def test_search_read_and_unicode_chunks_survive_reopen(snapshot):
    async with LocalSearch(snapshot) as backend:
        assert [hit.doc_id for hit in (await backend.search("перенесён")).hits] == ["ru"]
        assert [hit.doc_id for hit in (await backend.search("POSTPONED")).hits] == ["en"]
        assert not (await backend.search("unfindableword")).hits
        ids = await backend.list_chunks("ru", limit=100)
        fragments = [await backend.read_chunk(identifier) for identifier in ids]
        assert "".join(fragment.text for fragment in fragments) == DOCUMENTS[0].text
        first = await backend.read_document("ru", limit=5)
        second = await backend.read_document("ru", offset=first.next_offset, limit=5)
        assert first.text + second.text == DOCUMENTS[0].text[:10]
        end = await backend.read_document("ru", offset=len(DOCUMENTS[0].text))
        assert end.text == "" and end.next_offset is None
    async with LocalSearch(snapshot) as reopened:
        assert await reopened.list_chunks("ru", limit=100) == ids
        assert await reopened.read_chunk(ids[0]) == fragments[0]


async def test_ranking_unique_documents_and_limits(tmp_path):
    path = tmp_path / "rank.sqlite"
    build_snapshot([Document("b", "needle"), Document("a", "needle"),
                    Document("c", "hay " * 100 + "needle")], path, corpus_id="rank", chunk_chars=5)
    async with LocalSearch(path, SearchConfig(snippet_chars=4, max_read_chars=3)) as backend:
        result = await backend.search("needle", limit=2)
        assert [hit.doc_id for hit in result.hits] == ["a", "b"]
        assert [hit.rank for hit in result.hits] == [1, 2]
        assert all(len(hit.snippet) <= 4 for hit in result.hits)
        assert len((await backend.read_document("c")).text) == 3
        assert (await backend.read_document("c")).next_offset == 3


async def test_title_matches_return_readable_body_snippet(tmp_path):
    path = tmp_path / "title.sqlite"
    build_snapshot([Document("doc", "The original body text.", title="uniquetitle")], path, corpus_id="title")
    async with LocalSearch(path) as backend:
        hit = (await backend.search("uniquetitle")).hits[0]
        assert hit.doc_id == "doc"
        assert hit.snippet == (await backend.read_document(hit.doc_id)).text


@pytest.mark.parametrize("query", ["", "   ", "()", "\x00", None, 1, "a" * 2001, " ".join(f"word{i}" for i in range(33))])
async def test_invalid_query(snapshot, query):
    async with LocalSearch(snapshot) as backend:
        with pytest.raises(InvalidRequest):
            await backend.search(query)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 101])
async def test_invalid_limit(snapshot, limit):
    async with LocalSearch(snapshot) as backend:
        with pytest.raises(InvalidRequest):
            await backend.search("text", limit=limit)


async def test_query_operators_are_plain_text(snapshot):
    async with LocalSearch(snapshot) as backend:
        assert (await backend.search('"postponed" OR (unfindableword)')).hits[0].doc_id == "en"
        assert (await backend.search("postponed'; DROP TABLE documents; --")).hits[0].doc_id == "en"
        assert (await backend.read_document("ru")).text == DOCUMENTS[0].text


async def test_unknown_ids_and_offsets(snapshot):
    async with LocalSearch(snapshot) as backend:
        for method in (backend.read_document, backend.read_chunk, backend.list_chunks):
            with pytest.raises(UnknownId):
                await method("missing")
        for offset in (-1, True, len(DOCUMENTS[0].text) + 1):
            with pytest.raises(InvalidRequest):
                await backend.read_document("ru", offset=offset)


async def test_sixteen_parallel_requests_keep_corpora_separate(snapshot, tmp_path):
    other = tmp_path / "other.sqlite"
    build_snapshot([Document("ru", "Другой корпус: поиск")], other, corpus_id="other")
    async with LocalSearch(snapshot) as first, LocalSearch(other) as second:
        async def run(index):
            backend = first if index % 2 else second
            for _ in range(3):
                result = await backend.search("поиск")
                assert result.corpus_revision == backend.manifest["corpus_revision"]
                text = await backend.read_document(result.hits[0].doc_id)
                assert text.text == (DOCUMENTS[0].text if index % 2 else "Другой корпус: поиск")
        await asyncio.gather(*(run(i) for i in range(16)))


async def test_adapter_pins_revision_and_validates_corpus(snapshot):
    async with LocalSearch(snapshot) as backend:
        adapter = HarnessSearch(backend)
        hits = await adapter.search("postponed", corpus_id="demo", corpus_revision=None, limit=10)
        assert hits[0].model_dump() == {"doc_id": "en", "snippet": DOCUMENTS[1].text, "rank": 1}
        for corpus, revision in [("wrong", None), ("demo", "wrong")]:
            with pytest.raises(UnknownId):
                await adapter.search("postponed", corpus_id=corpus, corpus_revision=revision, limit=10)


async def test_cancellation_keeps_worker_slot_and_event_loop_live(snapshot):
    started, release = threading.Event(), threading.Event()
    async with LocalSearch(snapshot, SearchConfig(workers=1, max_pending=1)) as backend:
        def held(db):
            started.set()
            release.wait(2)
        task = asyncio.create_task(backend._run("held", held))
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(SearchBusy):
                await backend.search("text")
        finally:
            release.set()


async def test_deadline_interrupts_actual_sqlite_work(snapshot):
    async with LocalSearch(snapshot, SearchConfig(workers=1, timeout_seconds=0.02)) as backend:
        with pytest.raises(SearchTimeout):
            await backend._run("expensive", lambda db: db.execute("""
                WITH RECURSIVE numbers(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM numbers WHERE x<100000000)
                SELECT sum(x) FROM numbers
            """).fetchone())
        assert (await backend.search("postponed")).hits[0].doc_id == "en"


async def test_deadline_includes_queue_wait(snapshot):
    started, release = threading.Event(), threading.Event()
    async with LocalSearch(snapshot, SearchConfig(workers=1, timeout_seconds=0.05)) as backend:
        def held(db):
            started.set()
            release.wait(2)
        task = asyncio.create_task(backend._run("held", held))
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            with pytest.raises(SearchTimeout):
                await backend.search("postponed")
            with pytest.raises(SearchTimeout):
                await task
        finally:
            release.set()


async def test_telemetry_does_not_include_query_or_document(snapshot, caplog):
    caplog.set_level(logging.INFO, logger="search_rl_env.search.backend")
    async with LocalSearch(snapshot) as backend:
        await backend.search("private-query-marker")
        with pytest.raises(UnknownId):
            await backend.read_document("private-document-marker")
    records = [json.loads(record.message) for record in caplog.records]
    assert [record["status"] for record in records] == ["ok", "UnknownId"]
    assert records[0]["result_count"] == 0
    assert records[0]["queue_seconds"] >= 0
    assert records[0]["execution_seconds"] >= 0
    assert "private-" not in caplog.text


async def test_missing_corrupt_replaced_and_closed_snapshot(snapshot, tmp_path):
    with pytest.raises(StorageError):
        LocalSearch(tmp_path / "absent.sqlite")
    assert not (tmp_path / "absent.sqlite").exists()
    bad = tmp_path / "bad.sqlite"
    bad.write_text("not sqlite")
    with pytest.raises(StorageError):
        LocalSearch(bad)
    backend = LocalSearch(snapshot)
    snapshot.unlink()
    build_snapshot([Document("new", "new content")], snapshot, corpus_id="demo")
    with pytest.raises(StorageError):
        await backend.search("new")
    await backend.aclose()
    with pytest.raises(StorageError):
        await backend.search("new")

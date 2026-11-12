import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import pytest

from search_rl_env.agent import RolloutRunner
from search_rl_env.agent.mocks import MockModel
from search_rl_env.contracts import AgentTask, EpisodeConfig
from search_rl_env.search import SearchClient, SQLClient, UnknownId

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def go_binary(tmp_path_factory):
    if shutil.which("go") is None:
        pytest.skip("Для сквозной проверки Go-сервиса установите Go 1.25+.")
    binary = tmp_path_factory.mktemp("go-binary") / "search"
    environment = os.environ.copy()
    environment.setdefault("GOCACHE", str(Path(tempfile.gettempdir()) / "search-go-buildcache"))
    subprocess.run(["go", "build", "-o", str(binary), "./cmd/search"], cwd=ROOT / "services/search",
                   env=environment, check=True, capture_output=True, timeout=180)
    return binary


@pytest.fixture
def go_service(go_binary, tmp_path, request):
    snapshot = tmp_path / "corpus.sqlite"
    hydra = getattr(request, "param", None) == "hydra"
    source = ROOT / "examples/fixtures/hydra/corpus.jsonl" if hydra else ROOT / "examples/fixtures/corpus.jsonl"
    if hydra and not (ROOT / "data/hydra/hockey.sqlite").exists():
        pytest.skip("Для настоящего SQL replay выполните examples/prepare_hydra.py.")
    corpus = "hydra-a2s4-kilt-pilot" if hydra else "integration"
    subprocess.run([str(go_binary), "build", "-input", str(source),
                    "-output", str(snapshot), "-corpus-id", corpus, "-chunk-chars", "17"],
                   check=True, capture_output=True, timeout=20)
    log = tmp_path / "service.jsonl"
    with log.open("wb") as stream:
        command = [str(go_binary), "serve", "-snapshot", str(snapshot), "-listen", "127.0.0.1:0"]
        if hydra:
            command += ["-sql-snapshot", str(ROOT / "data/hydra/hockey.sqlite"), "-db-id", "hockey"]
        process = subprocess.Popen(command,
                                   stdout=subprocess.DEVNULL, stderr=stream)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("Go-сервис завершился до готовности: " + log.read_text())
            lines = log.read_text().splitlines()
            if lines:
                ready = json.loads(lines[0])
                if ready.get("msg") == "search_ready":
                    yield "http://" + ready["address"]
                    return
            time.sleep(0.01)
        pytest.fail("Go-сервис не сообщил о готовности.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


async def test_real_go_search_read_and_chunks(go_service):
    async with SearchClient(go_service) as search:
        hits = await search.search("перенесён", corpus_id="integration", corpus_revision=None, limit=10)
        assert [hit.doc_id for hit in hits] == ["doc-001"]
        text = await search.read(doc_id=hits[0].doc_id)
        assert text.text == hits[0].snippet
        ids = []
        offset = 0
        while True:
            page = await search.chunks(hits[0].doc_id, offset=offset, limit=2)
            ids.extend(page.chunk_ids)
            if page.next_offset is None:
                break
            offset = page.next_offset
        fragments = [await search.read(chunk_id=identifier) for identifier in ids]
        assert "".join(fragment.text for fragment in fragments) == text.text
        with pytest.raises(UnknownId):
            await search.read(doc_id="missing")


async def test_sixteen_harness_episodes_use_go_service(go_service):
    async with SearchClient(go_service) as search:
        async with RolloutRunner(MockModel(), search, EpisodeConfig(max_steps=4), concurrency=16) as runner:
            async def episode(index):
                task = AgentTask(task_id=f"task-{index}", question="перенесён", language="ru",
                                 corpus_id="integration", corpus_revision=search.manifest["corpus_revision"])
                result = await runner.run(task)
                assert result.stop_reason == "finished", result.error
                assert result.selected_document_ids == ["doc-001"]
                assert result.search_calls == 1
                return result.episode_id
            ids = await asyncio.gather(*(episode(index) for index in range(16)))
            assert len(set(ids)) == 16


@pytest.mark.parametrize("go_service", ["hydra"], indirect=True)
async def test_real_hydra_sql_and_search_use_same_task(go_service):
    reference = json.loads((ROOT / "examples/fixtures/hydra/reference.json").read_text())
    provenance = json.loads((ROOT / "examples/fixtures/hydra/provenance.json").read_text())
    async with SearchClient(go_service) as search, SQLClient(go_service) as sql:
        assert sql.info["db_revision"] == provenance["sql_database"]["sha256"]
        results = await asyncio.gather(*(sql.query(reference["sql_query"]) for _ in range(16)))
        assert all(result.rows == reference["expected_rows"] and not result.truncated for result in results)
        result = results[0]
        assert result.columns == reference["expected_columns"]
        assert result.rows == reference["expected_rows"]
        assert not result.truncated
        hits = await search.search("Toronto Arenas suspending operations 1918 1919",
                                   corpus_id="hydra-a2s4-kilt-pilot", corpus_revision=None, limit=10)
        assert {hit.doc_id for hit in hits} & set(provenance["manual_reference_doc_ids"])

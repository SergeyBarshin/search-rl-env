from contextlib import AsyncExitStack
import json
from pathlib import Path

import pytest

from search_rl_env.search.environments import EnvironmentFactory, prepare_dataset
from search_rl_env.search.local import HashingEncoder, digest_file
from search_rl_env.search.models import StorageError, UnknownId

FIXTURE = Path(__file__).resolve().parents[3] / "examples/fixtures/environments"
MANIFESTS = [FIXTURE / name / "manifest.json" for name in ("alpha", "beta")]


def test_dataset_revisions_schema_and_stable_ids(tmp_path):
    alpha = prepare_dataset(MANIFESTS[0], tmp_path / "alpha")
    beta = prepare_dataset(MANIFESTS[1], tmp_path / "beta")
    assert alpha.manifest["dataset_revision"] != beta.manifest["dataset_revision"]
    assert alpha.manifest["sql"]["db_revision"] != beta.manifest["sql"]["db_revision"]
    assert "CREATE TABLE facts" in alpha.manifest["sql"]["schema"]
    assert {doc["doc_id"] for doc in alpha.documents} & {doc["doc_id"] for doc in beta.documents} == {"doc-shared"}


def test_dataset_rejects_wrong_hash_and_duplicate_ids(tmp_path):
    spec = json.loads(MANIFESTS[0].read_text())
    spec["sql_path"] = str(FIXTURE / "alpha/source.sql")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id":"same","text":"first"}\n{"doc_id":"same","text":"second"}\n')
    spec["corpus_path"] = str(corpus)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    with pytest.raises(StorageError):
        prepare_dataset(manifest, tmp_path / "data")
    spec["corpus_sha256"] = digest_file(corpus)
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="doc_id"):
        prepare_dataset(manifest, tmp_path / "data")


def test_encoder_is_deterministic_normalized_and_records_unicode_version():
    np = pytest.importorskip("numpy")
    first, second = HashingEncoder(), HashingEncoder()
    vectors = first.encode(["Срок Project", "срок project"])
    assert np.array_equal(vectors, second.encode(["Срок Project", "срок project"]))
    assert np.array_equal(vectors[0], vectors[1])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert first.manifest["unicode_version"]


@pytest.fixture
def faiss():
    return pytest.importorskip("faiss", reason="Установите .[environments] для настоящего FAISS.")


async def test_environments_have_own_sqlite_and_faiss_and_release_resources(faiss, tmp_path):
    async with EnvironmentFactory(MANIFESTS) as factory:
        async with AsyncExitStack() as stack:
            alpha = await stack.enter_async_context(factory.open("alpha"))
            beta = await stack.enter_async_context(factory.open("beta"))
            assert factory.active_count == factory.peak_active == 2
            assert alpha.sql.path != beta.sql.path
            assert alpha.search._index is not beta.search._index
            assert (await alpha.sql.query("SELECT value FROM facts WHERE fact_id='target'")).rows == [["ALPHA-17"]]
            assert (await beta.sql.query("SELECT value FROM facts WHERE fact_id='target'")).rows == [["BETA-29"]]
            for environment, keyword in [(alpha, "alphakey"), (beta, "betakey")]:
                scope = environment.manifest["search"]
                hits = await environment.search.search(keyword, corpus_id=scope["corpus_id"],
                                                       corpus_revision=scope["corpus_revision"], limit=1)
                assert [(hit.doc_id, hit.snippet) for hit in hits] == [("doc-shared", keyword)]
            with pytest.raises(UnknownId):
                await alpha.search.search("betakey", corpus_id="beta-corpus", corpus_revision=None, limit=1)
            with pytest.raises(UnknownId):
                await alpha.search.search("alphakey", corpus_id="alpha-corpus", corpus_revision="wrong", limit=1)
            await alpha.aclose()
            assert factory.active_count == 1 and not alpha.directory.exists()
            assert (await beta.sql.query("SELECT count(*) FROM facts")).rows == [[2]]
        assert factory.active_count == 0
        assert beta.closed and not beta.directory.exists()
        async with factory.open("alpha") as restarted:
            assert restarted.environment_id != alpha.environment_id
            assert (await restarted.sql.query("SELECT count(*) FROM facts")).rows == [[2]]
        exported = await factory.export(tmp_path / "artifacts")
        for relative in exported.values():
            manifest_path = tmp_path / "artifacts" / relative
            manifest = json.loads(manifest_path.read_text())
            for name, revision in manifest["artifacts"].items():
                assert digest_file(manifest_path.parent / name) == revision
            assert manifest["artifacts"]["index.faiss"] == manifest["search"]["index_content_sha256"]
            assert faiss.read_index(str(manifest_path.parent / "index.faiss")).ntotal == 2
    assert factory.closed and not Path(factory._temporary.name).exists()


async def test_error_and_factory_shutdown_close_environments(faiss):
    async with EnvironmentFactory(MANIFESTS) as factory:
        with pytest.raises(RuntimeError, match="probe"):
            async with factory.open("alpha") as failed:
                raise RuntimeError("probe")
        assert failed.closed and factory.active_count == 0 and not failed.directory.exists()
        manager = factory.open("beta")
        forgotten = await manager.__aenter__()
    assert forgotten.closed and not forgotten.directory.exists()
    await manager.__aexit__(None, None, None)

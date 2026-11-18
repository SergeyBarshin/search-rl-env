import json

import httpx
import pytest

from search_rl_env.search.embedding_index import EmbeddingSearch, build_embedding_index
from search_rl_env.search.embeddings import JINA_DIMENSIONS, JinaEmbeddings
from search_rl_env.search.models import StorageError
from search_rl_env.search.server import SearchStore, build_snapshot


async def test_reopen_faiss_snapshot_and_query_without_document_encoding(tmp_path):
    pytest.importorskip("faiss")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id":"a","text":"Paris France"}\n{"doc_id":"b","text":"London UK"}\n')
    snapshot = tmp_path / "bm25.sqlite"
    build_snapshot(corpus, snapshot, "fixture")
    store = SearchStore(snapshot, sealed=True)
    requests = []

    def embeddings(request):
        assert request.url.path == "/v1/embeddings"
        texts = json.loads(request.content)["input"]
        requests.extend(texts)
        data = []
        for i, text in enumerate(texts):
            embedding = [0.0] * JINA_DIMENSIONS
            embedding[1 if "London" in text else 0] = 2.0
            data.append({"index": i, "embedding": embedding})
        return httpx.Response(200, json={"data": data[::-1]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(embeddings)) as http:
        encoder = JinaEmbeddings(http)
        directory = tmp_path / "vectors"
        manifest = await build_embedding_index(store, encoder, directory)
        assert manifest["documents"] == 2
        search = EmbeddingSearch(store, encoder, directory, mode="dense")
        hits = await search.search("London", corpus_id="fixture", corpus_revision=manifest["corpus_revision"], limit=2)
        assert [hit.doc_id for hit in hits] == ["b", "a"]
        assert requests == ["Document: Paris France", "Document: London UK", "Query: London"]

        with (directory / "index.faiss").open("ab") as stream:
            stream.write(b"invalid")
        with pytest.raises(StorageError):
            EmbeddingSearch(store, encoder, directory)

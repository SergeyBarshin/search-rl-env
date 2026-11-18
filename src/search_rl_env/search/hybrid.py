"""BM25 и FAISS по одному локальному снимку корпуса."""

from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version

from search_rl_env.contracts import SearchHit

from .local import FaissSearch, HashingEncoder, fingerprint, joined_thread
from .models import StorageError
from .semantic import fuse_rankings
from .server import SearchRequest, SearchStore


class HybridIndex:
    """Строит FAISS один раз и открывает независимый поиск для каждого эпизода."""

    def __init__(self, store: SearchStore):
        import faiss

        self._faiss = faiss
        self._encoder = HashingEncoder(dimensions=1024)
        self._bm25_revision = store.revision
        with store.connection() as db:
            rows = db.execute("SELECT doc_id, title, text FROM documents ORDER BY doc_id").fetchall()
        self._documents = tuple({"doc_id": doc_id, "text": text} for doc_id, _, text in rows)
        self._passages = {doc_id: f"{title}\n{text}" if title else text for doc_id, title, text in rows}
        vectors = self._encoder.encode([title + "\n" + text for _, title, text in rows])
        self._index = faiss.IndexFlatIP(self._encoder.dimensions)
        self._index.add(vectors)
        self._manifest = {
            "corpus_id": store.manifest["corpus_id"],
            "corpus_revision": store.manifest["corpus_revision"],
            "documents": len(rows),
            "index_algorithm": "bm25-faiss-rrf-v2",
            "bm25": {**store.manifest, "snapshot_sha256": store.revision},
            "faiss": {
                "index": "IndexFlatIP", "encoder": self._encoder.manifest,
                "version": version("faiss-cpu"),
                "numpy_version": version("numpy"),
                "index_sha256": sha256(faiss.serialize_index(self._index).tobytes()).hexdigest(),
            },
            "fusion": {"algorithm": "rrf", "constant": 60, "candidates_per_index": "request_limit"},
        }
        self._manifest["search_revision"] = fingerprint(self._manifest)

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def open(self, store: SearchStore):
        if store.revision != self._bm25_revision:
            raise StorageError("Снимок BM25 не соответствует индексу FAISS.")
        dense = FaissSearch(self._faiss.clone_index(self._index), self._documents, self._encoder, self._manifest)
        return HybridSearch(store, dense, self._passages)


class HybridSearch:
    def __init__(self, bm25: SearchStore, dense: FaissSearch, passages: dict[str, str]):
        self._bm25, self._dense, self._passages = bm25, dense, passages

    @property
    def manifest(self):
        return self._dense.manifest

    async def search(self, query, *, corpus_id, corpus_revision, limit):
        scope = dict(corpus_id=corpus_id, corpus_revision=corpus_revision, limit=limit)
        lexical = await joined_thread(self._bm25.search, SearchRequest(query=query, **scope))
        dense = await self._dense.search(query, **scope)
        hits = fuse_rankings(lexical["hits"], [hit.model_dump() for hit in dense], limit)
        return [SearchHit(doc_id=hit["doc_id"], snippet=self._passages[hit["doc_id"]], rank=hit["rank"])
                for hit in hits]

    async def aclose(self):
        await self._dense.aclose()

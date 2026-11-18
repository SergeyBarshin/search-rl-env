"""Строит и открывает общий неизменяемый снимок FAISS поверх локального корпуса."""

from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from search_rl_env.contracts import SearchHit

from .embeddings import JinaEmbeddings
from .local import digest_file, fingerprint, joined_thread
from .models import StorageError
from .semantic import fuse_rankings
from .server import SearchRequest, SearchStore


def _documents(store, after, size):
    with store.connection() as db:
        return db.execute(
            "SELECT rowid, title, text FROM documents WHERE rowid > ? ORDER BY rowid LIMIT ?", (after, size)
        ).fetchall()


async def build_embedding_index(store: SearchStore, encoder: JinaEmbeddings, destination: Path, *, batch_size=8):
    import faiss
    import numpy as np

    if not 1 <= batch_size <= 128:
        raise ValueError("batch_size должен быть в диапазоне 1–128.")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(encoder.manifest["dimensions"]))
    with TemporaryDirectory(prefix=".faiss-", dir=destination.parent) as folder:
        build = Path(folder) / "snapshot"
        build.mkdir()
        rowid = 0
        while documents := await joined_thread(_documents, store, rowid, batch_size):
            rows, titles, texts = zip(*documents, strict=True)
            vectors = await encoder.encode([f"{title}\n{text}" if title else text
                                            for title, text in zip(titles, texts, strict=True)], query=False)
            await joined_thread(index.add_with_ids, vectors, np.asarray(rows, dtype=np.int64))
            rowid = rows[-1]
        store.assert_snapshot()
        index_file = build / "index.faiss"
        await joined_thread(faiss.write_index, index, str(index_file))
        manifest = {
            "version": 1, "corpus_id": store.manifest["corpus_id"],
            "corpus_revision": store.manifest["corpus_revision"], "bm25_sha256": store.revision,
            "index_sha256": await joined_thread(digest_file, index_file), "documents": index.ntotal,
            "encoder": encoder.manifest, "index": "IndexIDMap2(IndexFlatIP)", "faiss_version": version("faiss-cpu"),
        }
        (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(destination)
        build.rename(destination)
    return manifest


class EmbeddingSearch:
    """Один экземпляр на прогон. Возвращает стабильные ID снимка для оценки."""

    def __init__(self, store: SearchStore, encoder: JinaEmbeddings, directory: Path, *, mode="hybrid"):
        import faiss

        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode должен быть dense или hybrid.")
        self._store, self._encoder = store, encoder
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_bytes())
        index_file = directory / "index.faiss"
        if (manifest["version"] != 1 or manifest["bm25_sha256"] != store.revision
                or manifest["corpus_id"] != store.manifest["corpus_id"]
                or manifest["corpus_revision"] != store.manifest["corpus_revision"]
                or manifest["encoder"] != encoder.manifest
                or manifest["index_sha256"] != digest_file(index_file)):
            raise StorageError("FAISS, эмбеддер и BM25 должны принадлежать одному снимку.")
        self._index = faiss.read_index(str(index_file))
        if (not isinstance(self._index, faiss.IndexIDMap2)
                or self._index.ntotal != manifest["documents"] or self._index.ntotal != store.manifest["documents"]
                or self._index.d != encoder.manifest["dimensions"]
                or self._index.metric_type != faiss.METRIC_INNER_PRODUCT):
            raise StorageError("FAISS не соответствует формату манифеста.")
        self._manifest = {**manifest, "mode": mode, "rrf_constant": 60 if mode == "hybrid" else None}
        self._manifest["search_revision"] = fingerprint(self._manifest)

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def _lookup(self, ids):
        if not ids:
            return {}
        marks = ", ".join("?" for _ in ids)
        with self._store.connection() as db:
            return {rowid: (doc_id, (title + "\n" if title else "") + text) for rowid, doc_id, title, text in db.execute(
                f"SELECT rowid, doc_id, title, text FROM documents WHERE rowid IN ({marks})", ids
            )}

    def _retrieve(self, vector, request):
        self._store.scope(request)
        scores, positions = self._index.search(vector, request.limit)
        order = [(float(score), int(rowid)) for score, rowid in zip(scores[0], positions[0], strict=True) if rowid >= 0]
        documents = self._lookup([rowid for _, rowid in order])
        dense = [{"doc_id": documents[rowid][0], "snippet": documents[rowid][1]} for _, rowid in
                 sorted(order, key=lambda row: (-row[0], documents[row[1]][0]))]
        hits = dense
        if self._manifest["mode"] == "hybrid":
            hits = fuse_rankings(dense, self._store.search(request)["hits"], request.limit)
        return [SearchHit(doc_id=hit["doc_id"], snippet=hit["snippet"], rank=rank)
                for rank, hit in enumerate(hits, 1)]

    async def search(self, query, *, corpus_id, corpus_revision, limit):
        request = SearchRequest(query=query, corpus_id=corpus_id, corpus_revision=corpus_revision, limit=limit)
        if (corpus_id, corpus_revision) != (self._manifest["corpus_id"], self._manifest["corpus_revision"]):
            self._store.scope(request)
        vectors = await self._encoder.encode([query], query=True)
        return await joined_thread(self._retrieve, vectors, request)


class Bm25Search:
    def __init__(self, store: SearchStore):
        self._store = store

    @property
    def manifest(self):
        return {**deepcopy(self._store.manifest), "mode": "bm25", "bm25_sha256": self._store.revision}

    async def search(self, query, *, corpus_id, corpus_revision, limit):
        request = SearchRequest(query=query, corpus_id=corpus_id, corpus_revision=corpus_revision, limit=limit)
        result = await joined_thread(self._store.search, request)
        return [SearchHit(doc_id=hit["doc_id"], snippet=hit["snippet"], rank=hit["rank"]) for hit in result["hits"]]

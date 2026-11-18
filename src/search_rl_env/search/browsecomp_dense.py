"""Локальный FAISS из опубликованных векторов BrowseComp-Plus EN."""

import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

from .local import digest_file, fingerprint

MODEL = "qwen/qwen3-embedding-8b"
DIMENSIONS = 4096
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
INDEX_REVISION = "b3f37f70c33829eb09d04784a54277a31871fd63"
SHARDS = {
    "corpus.shard1_of_4.pkl": "4bfad4863cc4f38d2a6bef1ee54f81b8021a0cad985abd153b91fc0e29f22c7e",
    "corpus.shard2_of_4.pkl": "0285996dd37e58f5ad21ea4ed12d61d933c7e223c885955174e71ee30002dbd8",
    "corpus.shard3_of_4.pkl": "0701099f6c806c086b0a1eda1b3dc6f76449236f0339bc5cc153c79eb56140ae",
    "corpus.shard4_of_4.pkl": "b117e2d9f08c2dcf6f50229f34e1302f10a3b6960ef5dfa7f3719c21aaf0024a",
}


def build_published_dense(shards, destination, bm25_manifest):
    import faiss
    import numpy as np

    if bm25_manifest["corpus_id"] != "browsecomp-plus-en" or bm25_manifest["corpus_revision"] != INDEX_REVISION:
        raise ValueError("Опубликованные векторы и Lucene должны принадлежать одному релизу BrowseComp-Plus.")
    destination = Path(destination)
    if destination.exists():
        return load_dense_manifest(destination, bm25_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(DIMENSIONS)
    identifiers = []
    for filename, digest in SHARDS.items():
        path = Path(shards) / filename
        if digest_file(path) != digest:
            raise ValueError(f"SHA-256 не соответствует опубликованному шарду: {filename}")
        # Десериализация только после проверки точного SHA-256 официального файла.
        with path.open("rb") as source:
            embeddings, ids = pickle.load(source)
        if (embeddings.shape != (len(ids), DIMENSIONS) or embeddings.dtype != np.float32
                or not np.isfinite(embeddings).all()):
            raise ValueError(f"Некорректные векторы в шарде {filename}.")
        index.add(embeddings)
        identifiers.extend(map(str, ids))
        del embeddings
    if len(identifiers) != 100195 or len(set(identifiers)) != len(identifiers):
        raise ValueError("Индекс должен содержать 100 195 уникальных ID.")

    with TemporaryDirectory(prefix=".qwen-index-", dir=destination.parent) as folder:
        build = Path(folder) / "snapshot"
        build.mkdir()
        faiss.write_index(index, str(build / "index.faiss"))
        (build / "docids.json").write_text(json.dumps(identifiers) + "\n", encoding="utf-8")
        manifest = {
            "corpus_id": bm25_manifest["corpus_id"], "corpus_revision": INDEX_REVISION,
            "corpus_index_sha256": bm25_manifest["index_sha256"], "model": MODEL, "dimensions": DIMENSIONS,
            "query_prefix": QUERY_PREFIX, "index": "IndexFlatIP", "documents": len(identifiers),
            "shards": SHARDS, "snippet_chars": bm25_manifest["snippet_chars"],
            "files": {name: digest_file(build / name) for name in ("index.faiss", "docids.json")},
        }
        manifest["search_revision"] = fingerprint(manifest)
        (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        build.rename(destination)
    return manifest


def load_dense_manifest(directory, bm25_manifest):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if (
        manifest["corpus_id"] != bm25_manifest["corpus_id"]
        or manifest["corpus_revision"] != INDEX_REVISION
        or manifest["corpus_index_sha256"] != bm25_manifest["index_sha256"]
        or manifest["shards"] != SHARDS
        or manifest["model"] != MODEL
        or manifest["dimensions"] != DIMENSIONS
        or manifest["query_prefix"] != QUERY_PREFIX
        or fingerprint({key: value for key, value in manifest.items() if key != "search_revision"}) != manifest["search_revision"]
        or any(digest_file(directory / name) != manifest["files"][name] for name in ("index.faiss", "docids.json"))
    ):
        raise ValueError("FAISS-снимок Qwen несовместим с опубликованным корпусом или повреждён.")
    return manifest

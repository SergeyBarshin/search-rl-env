"""Раздать опубликованный Lucene/BM25-индекс BrowseComp-Plus локально."""

import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from search_rl_env.search.local import digest_file, fingerprint, joined_thread


class SearchBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(ge=1, le=100)


def load_manifest(path, index):
    manifest = json.loads(path.read_bytes())
    files = {entry.name: digest_file(entry) for entry in sorted(index.iterdir()) if entry.is_file()}
    if files != manifest["files"] or fingerprint(files) != manifest["index_sha256"]:
        raise ValueError("Файлы Lucene не соответствуют манифесту.")
    revision = manifest["search_revision"]
    if fingerprint({key: value for key, value in manifest.items() if key != "search_revision"}) != revision:
        raise ValueError("Хеш манифеста не совпадает с search_revision.")
    return manifest


def create_app(retrieve, manifest):
    app = web.Application(client_max_size=32 * 1024)
    slots = asyncio.Semaphore(4)
    revision = {"X-Search-Revision": manifest["search_revision"]}

    async def health(_):
        return web.json_response({"status": "ok"}, headers=revision)

    async def search(request):
        try:
            body = SearchBody.model_validate_json(await request.read())
        except ValidationError:
            raise web.HTTPBadRequest(text="Неверные параметры поиска.") from None
        async with slots:
            results = await retrieve(body.query, body.top_k)
        return web.json_response({"result": results}, headers=revision)

    app.router.add_get("/health", health)
    app.router.add_post("/search", search)
    return app


def bm25_hits(searcher, query, top_k, snippet_chars):
    return [
        {
            "doc_id": str(hit.docid),
            "snippet": json.loads(str(hit.lucene_document.get("raw")))["contents"][:snippet_chars],
            "score": float(hit.score),
        }
        for hit in searcher.search(query, top_k)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anserini-jar", type=Path, required=True)
    parser.add_argument("--jvm", type=Path)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, args.index)

    import jpype

    jpype.startJVM("-Xmx1g", jvmpath=args.jvm, classpath=[args.anserini_jar])
    searcher = jpype.JClass("io.anserini.search.SimpleSearcher")(str(args.index))

    async def retrieve(query, top_k):
        return await joined_thread(bm25_hits, searcher, query, top_k, manifest["snippet_chars"])

    try:
        web.run_app(create_app(retrieve, manifest), host="127.0.0.1", port=args.port)
    finally:
        searcher.close()


if __name__ == "__main__":
    main()

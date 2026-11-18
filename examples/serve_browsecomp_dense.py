"""Serve the frozen BrowseComp-Plus Qwen index with query embeddings via Eliza."""

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from aiohttp import web
from dotenv import load_dotenv
import httpx

from search_rl_env.agent.auth import load_api_key
from search_rl_env.contracts.types import ModelAuthenticationError, ModelTimeoutError, ModelTransportError, Usage
from search_rl_env.search.acceptance import ReservedBudgetModel, write_json
from search_rl_env.search.browsecomp_dense import (
    DIMENSIONS, MODEL, QUERY_PREFIX, build_published_dense, load_dense_manifest,
)
from search_rl_env.search.local import fingerprint, joined_thread

from serve_browsecomp_bm25 import create_app, load_manifest

MAX_PRICE_PER_MILLION = 0.05
REQUEST_RESERVE_USD = 0.004


@dataclass
class EmbeddingResponse:
    vector: object
    usage: Usage


class ElizaQueryEncoder:
    def __init__(self, http, session_id):
        self.http = http
        self.session_id = session_id
        self.requests = 0
        self.tokens = 0
        self.price_estimates = 0
        self.reported_cost_usd = 0.0

    @property
    def metadata(self):
        return {"model": MODEL, "via": "eliza-openrouter", "max_price_usd_per_million": MAX_PRICE_PER_MILLION}

    async def complete(self, query):
        import numpy as np

        try:
            response = await self.http.post("embeddings", json={
                "model": MODEL, "input": QUERY_PREFIX + query, "encoding_format": "float",
                "provider": {"max_price": {"prompt": MAX_PRICE_PER_MILLION}}, "session_id": self.session_id,
            })
        except httpx.TimeoutException as error:
            raise ModelTimeoutError("Таймаут запроса эмбеддинга.") from error
        except (httpx.NetworkError, httpx.RemoteProtocolError) as error:
            raise ModelTransportError("Сетевой сбой запроса эмбеддинга.") from error
        if response.status_code == 401:
            raise ModelAuthenticationError("Embeddings HTTP 401.")
        response.raise_for_status()
        payload = response.json()
        model_id = payload["model"]
        if (
            not isinstance(model_id, str) or model_id.casefold() != MODEL
            or len(payload["data"]) != 1 or payload["data"][0]["index"] != 0
        ):
            raise ValueError("Ответ не относится к запрошенному эмбеддингу.")
        vector = np.asarray(payload["data"][0]["embedding"], dtype=np.float32)
        if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("Ответ должен содержать 4096 конечных значений Qwen.")
        usage = payload["usage"]
        tokens = usage["total_tokens"]
        if type(tokens) is not int or not 0 < tokens <= 40960:
            raise ValueError("Эндпоинт не сообщил корректный расход токенов.")
        cost = usage.get("cost")
        if cost is None:
            self.price_estimates += 1
            cost = tokens * MAX_PRICE_PER_MILLION / 1_000_000
        else:
            self.reported_cost_usd += cost
        self.requests += 1
        self.tokens += tokens
        return EmbeddingResponse(vector.reshape(1, DIMENSIONS),
                                 Usage(prompt_tokens=tokens, total_tokens=tokens, cost=cost))


def open_http():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    endpoint = urlsplit(os.getenv("MODEL_API_URL", ""))
    if (endpoint.scheme, endpoint.netloc, endpoint.path) != (
        "https", "api.eliza.yandex.net", "/raw/openrouter/v1/chat/completions"
    ):
        raise ValueError("Для токена Eliza нужен MODEL_API_URL маршрута Eliza → OpenRouter.")
    return httpx.AsyncClient(
        base_url="https://api.eliza.yandex.net/raw/openrouter/v1/", timeout=40,
        headers={"Authorization": f"{os.getenv('MODEL_AUTH_SCHEME', 'OAuth')} {load_api_key()}"},
    )


async def application(args, bm25, dense):
    import faiss
    import jpype

    usage_id = f"browsecomp-qwen-{uuid4().hex}"
    index = faiss.read_index(str(args.snapshot / "index.faiss"))
    ids = json.loads((args.snapshot / "docids.json").read_bytes())
    if index.d != DIMENSIONS or index.ntotal != len(ids) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Формат FAISS не соответствует манифесту.")
    jpype.startJVM("-Xmx1g", jvmpath=args.jvm, classpath=[args.anserini_jar])
    corpus = jpype.JClass("io.anserini.search.SimpleSearcher")(str(args.bm25_index))
    http = open_http()
    encoder = ElizaQueryEncoder(http, usage_id)
    billing = ReservedBudgetModel(encoder, args.budget_usd, REQUEST_RESERVE_USD,
                                  recoverable_errors=(ModelTransportError,), retry_auth_once=True)
    embedding_slots = min(args.embedding_concurrency, int(args.budget_usd / REQUEST_RESERVE_USD))
    embedding_gate = asyncio.Semaphore(embedding_slots)
    transport_retries = Counter()

    def rank(vector, top_k):
        scores, positions = index.search(vector, top_k)
        results = []
        for score, position in zip(scores[0], positions[0], strict=True):
            doc_id = ids[position]
            contents = json.loads(str(corpus.doc_raw(doc_id)))["contents"]
            results.append({"doc_id": doc_id, "snippet": contents[:bm25["snippet_chars"]], "score": float(score)})
        return results

    async def retrieve(query, top_k):
        async with embedding_gate:
            try:
                response = await billing.complete(query)
            except ModelTransportError as error:
                transport_retries[error.code] += 1
                response = await billing.complete(query)
        return await joined_thread(rank, response.vector, top_k)

    manifest = {**dense, "embedding_service": {**billing.metadata, "concurrency": embedding_slots},
                "usage_id": usage_id}
    manifest.pop("search_revision")
    manifest["search_revision"] = fingerprint(manifest)
    write_json(args.service_manifest, manifest)
    app = create_app(retrieve, manifest)

    async def usage(_):
        return web.json_response({
            "usage_id": usage_id, "requests": encoder.requests, "total_tokens": encoder.tokens,
            "priced_by_cap": encoder.price_estimates, "cost_or_upper_bound_usd": billing.spent,
            "reported_cost_usd": encoder.reported_cost_usd,
            "unsettled_reservation_usd": billing.reserved, "blocked": billing.blocked,
            "auth_retries": billing.auth_retries, "timeout_retries": transport_retries[ModelTimeoutError.code],
            "transport_retries": transport_retries[ModelTransportError.code],
        })

    async def close(_):
        await http.aclose()
        corpus.close()

    app.router.add_get("/usage", usage)
    app.on_cleanup.append(close)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-manifest", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--anserini-jar", type=Path)
    parser.add_argument("--jvm", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--embedding-concurrency", type=int, default=4)
    parser.add_argument("--service-manifest", type=Path)
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()
    if args.build_only:
        if args.shards is None:
            parser.error("--build-only требует --shards.")
        bm25 = json.loads(args.bm25_manifest.read_bytes())
        manifest = build_published_dense(args.shards, args.snapshot, bm25)
        print(f"Подготовлено {manifest['documents']} векторов Qwen.")
        return
    if not all((args.bm25_index, args.anserini_jar, args.budget_usd, args.service_manifest)):
        parser.error("Для сервера укажите --bm25-index, --anserini-jar, --budget-usd и --service-manifest.")
    if args.embedding_concurrency < 1:
        parser.error("embedding-concurrency должен быть положительным.")
    bm25 = load_manifest(args.bm25_manifest, args.bm25_index)
    dense = load_dense_manifest(args.snapshot, bm25)
    web.run_app(application(args, bm25, dense), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

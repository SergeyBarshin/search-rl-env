"""Векторы Jina через совместимый с OpenAI сервер llama.cpp."""

import httpx

JINA_MODEL = "jina-v5-small-retrieval-q4_k_m"
JINA_SHA256 = "9440cf89f3e8a7a31a42e11b87e106dd5b344af4e0e3b6b21a96136cc8686e21"
JINA_DIMENSIONS = 1024
JINA_URL = "http://127.0.0.1:8081/v1"


class JinaEmbeddings:
    def __init__(self, http: httpx.AsyncClient, base_url: str = JINA_URL):
        self._http = http
        self._endpoint = base_url.rstrip("/") + "/embeddings"

    @property
    def manifest(self):
        return {"model": JINA_MODEL, "model_sha256": JINA_SHA256, "dimensions": JINA_DIMENSIONS,
                "pooling": "last", "normalization": "l2", "prefixes": ["Query: ", "Document: "]}

    async def encode(self, texts: list[str], *, query: bool):
        import numpy as np

        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Для эмбеддинга нужны непустые строки.")
        prefix = "Query: " if query else "Document: "
        response = await self._http.post(self._endpoint, json={
            "model": JINA_MODEL, "input": [prefix + text for text in texts], "encoding_format": "float",
        })
        response.raise_for_status()
        try:
            data = response.json()["data"]
            indexes = [row["index"] for row in data]
            if any(type(i) is not int for i in indexes) or sorted(indexes) != list(range(len(texts))):
                raise ValueError("Индексы ответа не совпадают с порядком запроса.")
            vectors = np.asarray([row["embedding"] for row in sorted(data, key=lambda row: row["index"])],
                                 dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Некорректный ответ эндпоинта эмбеддингов.") from error
        if vectors.shape != (len(texts), JINA_DIMENSIONS) or not np.isfinite(vectors).all():
            raise ValueError("Размерность или значения эмбеддингов не соответствуют модели.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("Невалидная норма эмбеддинга.")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

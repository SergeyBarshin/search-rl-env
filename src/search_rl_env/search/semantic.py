"""Диагностический FAISS-бейзлайн: компактный ONNX-энкодер, без LLM и GPU.

Это эксперимент сравнения с Go BM25, не замена работающего HTTP-сервиса.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_SHA256 = "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DenseBaseline:
    def __init__(self, documents, model_directory):
        import faiss
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._faiss, self._np = faiss, np
        folder = Path(model_directory)
        if file_hash(folder / "encoder.onnx") != MODEL_SHA256 or file_hash(folder / "tokenizer.json") != TOKENIZER_SHA256:
            raise ValueError("Хеш embedding-модели или токенизатора не совпадает с зафиксированным.")
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(str(folder / "encoder.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.chunks = []
        ids = set()
        for document in documents:
            if document["doc_id"] in ids or not document["text"].strip():
                raise ValueError("Повторный id или пустой документ.")
            ids.add(document["doc_id"])
            text = document.get("title", "") + "\n" + document["text"]
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, len(tokens.ids), 192):
                end = min(start + 224, len(tokens.ids))
                snippet = text[tokens.offsets[start][0]:tokens.offsets[end - 1][1]]
                self.chunks.append({"doc_id": document["doc_id"], "text": snippet})
                if end == len(tokens.ids):
                    break
        if not self.chunks:
            raise ValueError("Корпус не содержит фрагментов.")
        self._tokenizer.enable_truncation(max_length=256)
        self._tokenizer.enable_padding()
        vectors = self._encode([chunk["text"] for chunk in self.chunks])
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        digest = hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.manifest = {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                         "model_sha256": MODEL_SHA256, "tokenizer_sha256": TOKENIZER_SHA256,
                         "corpus_sha256": digest, "window_tokens": 224, "stride_tokens": 192,
                         "pooling": "attention-mask-mean+l2", "index": "IndexFlatIP",
                         "dimension": vectors.shape[1], "chunks": len(self.chunks), "documents": len(ids),
                         "language": "en", "providers": ["CPUExecutionProvider"],
                         "versions": {name: importlib.metadata.version(name) for name in
                                      ["faiss-cpu", "onnxruntime", "tokenizers", "numpy"]}}

    def _encode(self, texts):
        np = self._np
        batches = []
        for start in range(0, len(texts), 16):
            encoded = self._tokenizer.encode_batch(texts[start:start + 16])
            inputs = {"input_ids": np.array([item.ids for item in encoded], dtype=np.int64),
                      "attention_mask": np.array([item.attention_mask for item in encoded], dtype=np.int64),
                      "token_type_ids": np.array([item.type_ids for item in encoded], dtype=np.int64)}
            hidden = self._session.run(["last_hidden_state"], inputs)[0]
            mask = inputs["attention_mask"][..., None].astype(np.float32)
            vectors = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
                raise ValueError("Энкодер вернул недопустимый вектор.")
            self._faiss.normalize_L2(vectors)
            batches.append(vectors)
        return np.concatenate(batches)

    def search(self, query, limit=10):
        if not isinstance(query, str) or not query.strip() or not 1 <= limit <= 100:
            raise ValueError("Недопустимый query или limit.")
        if len(self._tokenizer.encode(query, add_special_tokens=False).ids) > 254:
            raise ValueError("Запрос превышает контекст embedding-модели.")
        # Для малого среза считаем все фрагменты: top-k документов после дедупликации точен.
        scores, indexes = self._index.search(self._encode([query]), len(self.chunks))
        candidates = {}
        for score, index in zip(scores[0], indexes[0]):
            chunk = self.chunks[int(index)]
            if chunk["doc_id"] not in candidates:
                candidates[chunk["doc_id"]] = {"doc_id": chunk["doc_id"], "score": float(score),
                                                "snippet": chunk["text"][:500]}
        hits = sorted(candidates.values(), key=lambda hit: (-hit["score"], hit["doc_id"]))[:limit]
        return [{**hit, "rank": index} for index, hit in enumerate(hits, 1)]

    def save(self, directory):
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(folder / "dense.faiss"))
        (folder / "dense-manifest.json").write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n")
        (folder / "dense-chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False) + "\n")


def fuse_rankings(first, second, limit=10, constant=60):
    """Фиксированный reciprocal rank fusion без подбора веса по эталону."""
    scores, snippets = {}, {}
    for ranking in (first, second):
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            identifier = hit["doc_id"]
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] = scores.get(identifier, 0) + 1 / (constant + rank)
            snippets.setdefault(identifier, hit["snippet"])
    order = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:limit]
    return [{"doc_id": identifier, "rank": rank, "score": scores[identifier], "snippet": snippets[identifier]}
            for rank, identifier in enumerate(order, 1)]

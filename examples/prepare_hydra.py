"""Скачать только исходную SQL-базу и опциональный компактный CPU-энкодер."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from search_rl_env.search.semantic import MODEL_REVISION, MODEL_SHA256, TOKENIZER_SHA256

DB_URL = "https://huggingface.co/datasets/ibm-research/VAKRA/resolve/1388b9f1aaed887a73957eba651a6c2034010476/databases/hockey/hockey.sqlite"
DB_HASH = "2987cba3921e2544fcd4603eccf5ff97b17f63103d8feb6fb908d55eb5bd9ba2"


def download(url, path, expected, maximum):
    path = Path(path)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Существующий артефакт имеет другой хеш; выберите новый каталог.")
        return
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Размер или SHA-256 скачанного артефакта не совпадает.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/hydra"))
    parser.add_argument("--with-vectors", action="store_true")
    parser.add_argument("--verify-corpus-source", action="store_true",
                        help="Перечитать только зафиксированную Parquet-группу KILT и сверить корпус")
    args = parser.parse_args()
    if args.verify_corpus_source:
        import fsspec
        import pyarrow.parquet as pq
        fixture = Path(__file__).parent / "fixtures/hydra"
        source = json.loads((fixture / "provenance.json").read_text())["corpus"]
        url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{source['file']}"
        with fsspec.open(url, "rb", block_size=65536, cache_type="blockcache") as stream:
            rows = pq.ParquetFile(stream).read_row_group(source["row_group"]).to_pylist()
        documents = [{"doc_id": row["id"], "text": row["contents"], "title": row["title"], "language": "en"}
                     for row in rows if row["contents"].strip()]
        text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in documents)
        if hashlib.sha256(text.encode()).hexdigest() != source["sha256"]:
            raise ValueError("Опубликованный срез KILT не совпал с зафиксированным.")
    download(DB_URL, args.output / "hockey.sqlite", DB_HASH, 8 * 1024 * 1024)
    if args.with_vectors:
        base = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{MODEL_REVISION}/"
        download(base + "onnx/model_qint8_arm64.onnx", args.output / "encoder.onnx", MODEL_SHA256, 24 * 1024 * 1024)
        download(base + "tokenizer.json", args.output / "tokenizer.json", TOKENIZER_SHA256, 2 * 1024 * 1024)
    print("HyDRA-артефакты подготовлены; SHA-256 проверены.")


if __name__ == "__main__":
    main()

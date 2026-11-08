"""Небольшой воспроизводимый замер искусственного RU/EN-корпуса без моделей."""

import argparse
import asyncio
import json
import math
from pathlib import Path
import platform
import sqlite3
import tempfile
from time import perf_counter

from search_rl_env.search import Document, LocalSearch, SearchConfig, SearchError, build_snapshot


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Нужно положительное число.")
    return value


def documents(count, chars):
    for index in range(count):
        base = ("Поставка оборудования перенесена на ноябрь. " if index % 2 else
                "Equipment delivery postponed until November. ")
        # Различные термины дают редкие запросы; весь текст имеет заданную длину.
        text = (base * (chars // len(base) + 1))[:chars] + f" marker{index}"
        yield Document(f"doc-{index:08d}", text, language="ru" if index % 2 else "en")


async def measure(path, args):
    config = SearchConfig(workers=args.workers, max_pending=max(32, args.concurrency))
    async with LocalSearch(path, config) as backend:
        queries = ["поставка", "equipment", "notincorpus", f"marker{args.documents // 2}", "November ноябрь"]
        for query in queries:
            await backend.search(query)
        latencies, failures = [], []
        next_request = 0
        started = perf_counter()

        async def worker():
            nonlocal next_request
            while next_request < args.requests:
                index = next_request
                next_request += 1
                request_started = perf_counter()
                try:
                    await backend.search(queries[index % len(queries)])
                except SearchError as error:
                    failures.append(type(error).__name__)
                else:
                    latencies.append(perf_counter() - request_started)

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = perf_counter() - started
        latencies.sort()

        def percentile(fraction):
            return 1000 * latencies[max(0, math.ceil(len(latencies) * fraction) - 1)] if latencies else None

        return dict(requests=args.requests, concurrency=args.concurrency, workers=args.workers,
                    successful=len(latencies), errors=failures, wall_seconds=elapsed,
                    successful_requests_per_second=len(latencies) / elapsed,
                    p50_ms=percentile(0.5), p95_ms=percentile(0.95),
                    manifest=backend.manifest, search_revision=backend.search_revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=positive, default=1000)
    parser.add_argument("--chars", type=positive, default=2000)
    parser.add_argument("--requests", type=positive, default=160)
    parser.add_argument("--concurrency", type=positive, default=16)
    parser.add_argument("--workers", type=positive, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="search-benchmark-") as directory:
        path = Path(directory) / "corpus.sqlite"
        started = perf_counter()
        build_snapshot(documents(args.documents, args.chars), path, corpus_id="synthetic-benchmark")
        build_seconds = perf_counter() - started
        result = asyncio.run(measure(path, args))
        result.update(documents=args.documents, base_document_chars=args.chars, build_seconds=build_seconds,
                      index_bytes=path.stat().st_size, python=platform.python_version(),
                      sqlite=sqlite3.sqlite_version, platform=platform.platform(),
                      note="Искусственные тексты, прогретый индекс, без модели; не оценка реального качества.")
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())

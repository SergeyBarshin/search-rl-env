"""HTTP-замер запущенного Go-сервиса без модели; корпус фиксируется манифестом."""

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter

from search_rl_env.search import SearchClient, SearchError


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Нужно положительное число.")
    return value


async def measure(args):
    queries = args.query or ["Toronto Arenas", "suspend operations", "notincorpus"]
    async with SearchClient(args.url) as search:
        manifest = search.manifest
        async def query(text):
            return await search.search(text, corpus_id=manifest["corpus_id"],
                                       corpus_revision=manifest["corpus_revision"], limit=10)
        for text in queries:
            await query(text)
        latencies, errors = [], []
        next_request = 0
        started = perf_counter()
        async def worker():
            nonlocal next_request
            while next_request < args.requests:
                index = next_request
                next_request += 1
                request_started = perf_counter()
                try:
                    await query(queries[index % len(queries)])
                except SearchError as error:
                    errors.append(type(error).__name__)
                else:
                    latencies.append(perf_counter() - request_started)
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = perf_counter() - started
        latencies.sort()
        def percentile(fraction):
            return 1000 * latencies[max(0, math.ceil(len(latencies) * fraction) - 1)] if latencies else None
        return {"manifest": manifest, "requests": args.requests, "concurrency": args.concurrency,
                "successful": len(latencies), "errors": errors, "wall_seconds": elapsed,
                "successful_requests_per_second": len(latencies) / elapsed,
                "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
                "note": "Прогретый сервис, closed-loop HTTP batch без модели; не оценка качества."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--requests", type=positive, default=160)
    parser.add_argument("--concurrency", type=positive, default=16)
    parser.add_argument("--query", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = asyncio.run(measure(args))
    except SearchError as error:
        parser.exit(1, f"Ошибка: {type(error).__name__}.\n")
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())

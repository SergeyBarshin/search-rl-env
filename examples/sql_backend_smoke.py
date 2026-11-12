"""Проверка готовности SQL-бэкенда: ограниченный поток RPC, без модели и оценок."""

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter

from search_rl_env.contracts import SqlQueryError
from search_rl_env.search import SQLClientPool, SearchError

QUERY = "SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name"


async def run(args):
    async with SQLClientPool(args.url, concurrency=args.concurrency) as pool:
        databases = await pool.databases()
        if not databases:
            raise ValueError("SQL-каталог пуст.")
        bindings, expected = [], []
        for database in databases:
            binding = await pool.bind(database["db_id"], database["db_revision"])
            baseline = await binding.query(QUERY)
            if baseline.truncated:
                raise ValueError("Схема превышает лимит smoke-запроса; задайте меньший каталог.")
            bindings.append(binding)
            expected.append(baseline.model_dump())
        next_index = 0
        durations, errors = [], []
        started = perf_counter()

        async def worker():
            nonlocal next_index
            while next_index < args.operations:
                index = next_index
                next_index += 1
                selected = index % len(bindings)
                before = perf_counter()
                try:
                    result = await bindings[selected].query(QUERY)
                    if result.model_dump() != expected[selected]:
                        errors.append("response_mismatch")
                    else:
                        durations.append(perf_counter() - before)
                except (SearchError, SqlQueryError) as error:
                    errors.append(type(error).__name__)

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = perf_counter() - started
        durations.sort()

        def percentile(fraction):
            return durations[max(0, math.ceil(len(durations) * fraction) - 1)] * 1000 if durations else None

        return {"mode": "sql_backend_rpc_smoke_without_agents", "operations": args.operations,
                "concurrency": args.concurrency, "databases": databases, "successful": len(durations),
                "errors": errors, "wall_seconds": elapsed, "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
                "successful_operations_per_second": len(durations) / elapsed,
                "note": "Проверка SQL RPC и стабильности выбранных снимков; не прогон агентов и не метрика бенчмарка."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--operations", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("runs/backend/sql-smoke.json"))
    args = parser.parse_args()
    if args.operations < 1 or args.concurrency < 16 or args.concurrency % 16:
        parser.error("Нужны operations>0 и concurrency=16/32/64/... .")
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["operations", "concurrency", "successful", "errors", "p95_ms"]}, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())

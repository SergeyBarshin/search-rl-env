"""Проверка SQL/search на одной настоящей train-задаче HyDRA; без модели агента."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from search_rl_env.agent.tools import ToolCall, execute_tool
from search_rl_env.contracts import AgentTask, EpisodeConfig
from search_rl_env.environment.state import SearchState
from search_rl_env.search import SearchClient, SQLClient


async def run(args):
    fixture = Path(__file__).parent / "fixtures/hydra"
    task = AgentTask.model_validate_json((fixture / "task.json").read_text())
    reference = json.loads((fixture / "reference.json").read_text())
    corpus = [json.loads(line) for line in (fixture / "corpus.jsonl").read_text().splitlines() if line]
    provenance = json.loads((fixture / "provenance.json").read_text())
    state, config = SearchState(), EpisodeConfig(search_limit=10)
    async with SearchClient(args.url) as search, SQLClient(args.url) as sql:
        if sql.info["db_revision"] != provenance["sql_database"]["sha256"]:
            raise ValueError("Запущена другая SQL-база.")
        if search.manifest["content_hash"] != provenance["corpus"]["content_hash"]:
            raise ValueError("Запущен другой поисковый корпус.")
        call = ToolCall.model_validate({"id": "hydra-sql", "type": "function",
            "function": {"name": "sql", "arguments": json.dumps({"query": reference["sql_query"]})}})
        started = perf_counter()
        result = await execute_tool(call, task=task, search=search, config=config, state=state, sql=sql)
        sql_seconds = perf_counter() - started
        if not result.get("ok") or result["columns"] != reference["expected_columns"] or result["rows"] != reference["expected_rows"] or result["truncated"]:
            raise ValueError("SQL-тул не воспроизвёл исходный результат HyDRA.")
        # Запрос строится из публичного вопроса и фактического SQL-результата, без search_answer.
        bridge = " ".join(str(cell) for row in result["rows"] for cell in row if cell is not None)
        query = task.question + "\nSQL result: " + bridge
        started = perf_counter()
        search_call = ToolCall.model_validate({"id": "hydra-search", "type": "function",
            "function": {"name": "search", "arguments": json.dumps({"query": query})}})
        observation = await execute_tool(search_call, task=task, search=search, config=config, state=state, sql=sql)
        bm25_seconds = perf_counter() - started
        if not observation.get("ok"):
            raise ValueError("Поисковый тул завершился с ошибкой.")
        bm25 = observation["hits"]
        inventory = {document["doc_id"]: document for document in corpus}
        for hit in bm25:
            text = await search.read(doc_id=hit["doc_id"])
            if text.text != inventory[hit["doc_id"]]["text"][:text.total_chars]:
                raise ValueError("Чтение не соответствует исходному KILT-фрагменту.")
        report = {"mode": "gold_sql_tool_replay_without_agent", "task_id": task.task_id,
                  "source": provenance, "search_manifest": search.manifest, "sql_manifest": sql.info,
                  "sql": {"columns": result["columns"], "rows": result["rows"], "seconds": sql_seconds, "matches_reference": True},
                  "query_recipe": "public_question + actual_sql_values", "query": query,
                  "bm25": {"hits": bm25, "seconds": bm25_seconds},
                  "limitations": ["Одна задача и диагностический подкорпус; не официальный HyDRA score.",
                                  "Эталонный SQL подаётся тестом, агент не решает задачу."]}
        if args.with_vectors:
            from search_rl_env.search.semantic import DenseBaseline, fuse_rankings
            started = perf_counter()
            dense = await asyncio.to_thread(DenseBaseline, corpus, args.assets)
            report["dense_build_seconds"] = perf_counter() - started
            started = perf_counter()
            dense_hits = await asyncio.to_thread(dense.search, query, 10)
            report["faiss"] = {"hits": dense_hits, "seconds_including_query_encoding": perf_counter() - started,
                               "manifest": dense.manifest}
            report["hybrid_rrf"] = {"hits": fuse_rankings(bm25, dense_hits), "constant": 60}
            dense.save(args.output.parent / "faiss")
        reference_ids = set(provenance["manual_reference_doc_ids"])
        for key in ["bm25", "faiss", "hybrid_rrf"]:
            if key in report:
                report[key]["manual_reference_hit_at_10"] = bool(reference_ids & {hit["doc_id"] for hit in report[key]["hits"]})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--assets", type=Path, default=Path("data/hydra"))
    parser.add_argument("--with-vectors", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/hydra/train_a2s_4.json"))
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(json.dumps({"task_id": report["task_id"], "sql_matches": report["sql"]["matches_reference"],
                      "retrieval": {key: report[key]["manual_reference_hit_at_10"] for key in
                                    ["bm25", "faiss", "hybrid_rrf"] if key in report},
                      "report": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

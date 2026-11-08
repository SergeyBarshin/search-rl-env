"""Вызовы отдельного Go-сервиса: поиск, чтение и эпизод с моковой моделью."""

import argparse
import asyncio
import json

from search_rl_env.search import SearchClient, SearchError


async def execute(args):
    async with SearchClient(args.url) as search:
        manifest = search.manifest
        if args.command == "search":
            hits = await search.search(args.query, corpus_id=manifest["corpus_id"],
                                       corpus_revision=manifest["corpus_revision"], limit=args.limit)
            return {"manifest": manifest, "hits": [hit.model_dump() for hit in hits]}
        if args.command == "read":
            return (await search.read(doc_id=args.doc_id, chunk_id=args.chunk_id,
                                      offset=args.offset, limit=args.limit)).model_dump()
        if args.command == "chunks":
            return (await search.chunks(args.doc_id, offset=args.offset, limit=args.limit)).model_dump()
        from search_rl_env.agent import run_episode
        from search_rl_env.agent.mocks import MockModel
        from search_rl_env.contracts import AgentTask, EpisodeConfig

        task = AgentTask(task_id="go-search-demo", question=args.query, language="ru",
                         corpus_id=manifest["corpus_id"], corpus_revision=manifest["corpus_revision"])
        result = await run_episode(task, MockModel(), search, EpisodeConfig(max_steps=4))
        return {"manifest": manifest, "stop_reason": result.stop_reason.value,
                "selected_document_ids": result.selected_document_ids, "search_calls": result.search_calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("search", "read", "chunks", "episode"):
        subparser = commands.add_parser(command)
        if command in {"search", "episode"}:
            subparser.add_argument("--query", required=True)
        if command != "episode":
            subparser.add_argument("--limit", type=int, default=8000 if command == "read" else 10)
        if command in {"read", "chunks"}:
            subparser.add_argument("--offset", type=int, default=0)
            if command == "read":
                identifiers = subparser.add_mutually_exclusive_group(required=True)
                identifiers.add_argument("--doc-id")
                identifiers.add_argument("--chunk-id")
            else:
                subparser.add_argument("--doc-id", required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(execute(args))
    except (SearchError, ValueError) as error:
        parser.exit(1, f"Ошибка: {type(error).__name__}. Проверьте сервис и параметры.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

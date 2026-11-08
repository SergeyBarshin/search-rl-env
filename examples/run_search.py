"""Загрузка JSONL, поиск и чтение без модели; команды описаны в README."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from search_rl_env.search import Document, LocalSearch, SearchError, build_snapshot


def documents(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield Document(**json.loads(line))


async def execute(args):
    async with LocalSearch(args.snapshot) as backend:
        if args.command == "search":
            return asdict(await backend.search(args.query, limit=args.limit))
        if args.command == "chunks":
            return await backend.list_chunks(args.doc_id, offset=args.offset, limit=args.limit)
        method = backend.read_chunk if args.chunk_id else backend.read_document
        return asdict(await method(args.chunk_id or args.doc_id, offset=args.offset, limit=args.limit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--input", type=Path, default=Path(__file__).parent / "fixtures/corpus.jsonl")
    build.add_argument("--output", type=Path, default=Path("indexes/demo.sqlite"))
    build.add_argument("--corpus-id", default="synthetic-demo")
    build.add_argument("--revision", default=None)
    build.add_argument("--chunk-chars", type=int, default=2000)
    for command in ("search", "read", "chunks"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--snapshot", type=Path, default=Path("indexes/demo.sqlite"))
        subparser.add_argument("--limit", type=int, default=10 if command != "read" else 8000)
        if command == "search":
            subparser.add_argument("--query", required=True)
        else:
            subparser.add_argument("--offset", type=int, default=0)
            if command == "read":
                identifiers = subparser.add_mutually_exclusive_group(required=True)
                identifiers.add_argument("--doc-id")
                identifiers.add_argument("--chunk-id")
            else:
                subparser.add_argument("--doc-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_snapshot(documents(args.input), args.output, corpus_id=args.corpus_id,
                                    corpus_revision=args.revision, chunk_chars=args.chunk_chars)
        else:
            result = asyncio.run(execute(args))
    except (SearchError, OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Ошибка: {type(error).__name__}. Проверьте вход и параметры.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

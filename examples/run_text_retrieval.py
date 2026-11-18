"""Сборка снимка, получение публичных траекторий и закрытая оценка Recall@10."""

import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack
from functools import partial
from itertools import islice
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

from search_rl_env.agent import RolloutRunner, save_episode
from search_rl_env.agent.auth import load_api_key
from search_rl_env.agent.client import ChatCompletionsClient
from search_rl_env.agent.mocks import MockModel
from search_rl_env.agent.rounds import BOUNDED_RETRIEVAL_PROMPT, ROUND_PROMPT
from search_rl_env.contracts.types import AgentTask, EpisodeConfig, EpisodeResult, RetrievalRoundConfig
from search_rl_env.evaluation.retrieval import (
    BENCHMARK_LANGUAGES, read_evidence, read_questions, score_retrieval, summarize_retrieval,
)
from search_rl_env.search.acceptance import ReservedBudgetModel, code_revision, write_json
from search_rl_env.search.embedding_index import Bm25Search, EmbeddingSearch, build_embedding_index
from search_rl_env.search.embeddings import JINA_URL, JinaEmbeddings
from search_rl_env.search.local import digest_file, joined_thread
from search_rl_env.search.reference import ReferenceSearch
from search_rl_env.search.server import SearchStore, build_snapshot
from search_rl_env.search.snippets import SnippetTokenSearch

QWEN_CHAT_MODEL = "qwen/qwen3.6-35b-a3b"
QWEN_TOKENIZER = (Path(__file__).resolve().parents[1]
                  / "data/text-retrieval/tokenizers/qwen3.6-35b-a3b/tokenizer.json")
QWEN_TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
DEFAULT_SEARCH_SNIPPET_TOKENS = 256


async def build(args):
    await joined_thread(build_snapshot, args.corpus, args.bm25, args.corpus_id, args.corpus_revision)
    store = await joined_thread(partial(SearchStore, args.bm25, sealed=True))
    if args.faiss is None:
        manifest = Bm25Search(store).manifest
    else:
        async with httpx.AsyncClient(timeout=120, trust_env=False) as http:
            encoder = JinaEmbeddings(http, args.embeddings_url)
            if args.faiss.exists():
                search = await joined_thread(EmbeddingSearch, store, encoder, args.faiss)
                manifest = search.manifest
            else:
                manifest = await build_embedding_index(store, encoder, args.faiss, batch_size=args.batch_size)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


async def open_search(stack, args):
    if args.reference_url is not None:
        manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
        http = await stack.enter_async_context(httpx.AsyncClient(timeout=120))
        return ReferenceSearch(http, args.reference_url, manifest)
    store = await joined_thread(partial(SearchStore, args.bm25, sealed=True))
    if args.mode == "bm25":
        return Bm25Search(store)
    http = await stack.enter_async_context(httpx.AsyncClient(timeout=120, trust_env=False))
    return await joined_thread(partial(EmbeddingSearch, store, JinaEmbeddings(http, args.embeddings_url),
                                       args.faiss, mode=args.mode))


async def open_model(stack, args):
    if args.direct:
        return MockModel()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    endpoint = os.getenv("MODEL_API_URL", "").strip()
    url = urlsplit(endpoint)
    if url.scheme != "https" or not url.hostname or url.username or url.query or url.fragment:
        raise ValueError("Для модельного прогона нужен MODEL_API_URL с HTTPS.")
    reasoning_enabled = True if args.reasoning_effort else None
    if reasoning_enabled is None and args.retrieval_policy == "bounded":
        reasoning_enabled = False
    model = await stack.enter_async_context(ChatCompletionsClient(
        api_key=load_api_key(), endpoint=endpoint, model=args.model_id,
        auth_scheme=os.getenv("MODEL_AUTH_SCHEME", "Bearer"), timeout_seconds=args.request_timeout,
        max_price_usd_per_million=tuple(args.openrouter_max_price),
        openrouter_provider=args.openrouter_provider,
        reasoning_enabled=reasoning_enabled,
        reasoning_effort=args.reasoning_effort,
        force_final_tool=args.retrieval_policy == "bounded",
    ))
    return ReservedBudgetModel(model, args.budget_usd, args.request_reserve_usd,
                               continue_after_timeout=True, retry_auth_once=True)


async def run(args):
    questions = read_questions(args.questions)
    selected = dict(islice(questions.items(), args.offset, args.offset + args.count))
    if len(selected) < args.count:
        raise ValueError(f"Запрошено {args.count} задач, найдено {len(selected)}.")

    async with AsyncExitStack() as stack:
        search = await open_search(stack, args)
        identity = search.manifest
        bounded = not args.direct and args.retrieval_policy == "bounded"
        max_tokens = args.max_tokens if args.max_tokens is not None else 2048 if bounded else 16384
        max_steps = args.max_steps
        if max_steps is None:
            max_steps = 2 if args.direct else 12 if bounded else None
        snippet_chars = args.search_snippet_chars
        snippet_tokens = args.search_snippet_tokens
        observation = {}
        if snippet_tokens is not None:
            tokenizer = args.search_tokenizer or QWEN_TOKENIZER
            if not tokenizer.is_file():
                raise ValueError(f"Токенизатор отсутствует: {tokenizer}. См. инструкцию в docs/EXPERIMENTS.md.")
            search = SnippetTokenSearch(search, tokenizer, snippet_tokens)
            observation = search.observation
            if args.search_tokenizer is None and observation["tokenizer_sha256"] != QWEN_TOKENIZER_SHA256:
                raise ValueError("SHA-256 штатного токенизатора Qwen не совпадает с закреплённой версией.")
        elif snippet_chars is None and args.reference_url is None:
            snippet_chars = 2000
        model = await open_model(stack, args)
        config = EpisodeConfig(
            system_prompt=BOUNDED_RETRIEVAL_PROMPT if bounded else ROUND_PROMPT,
            retrieval_rounds=RetrievalRoundConfig(
                max_rounds=args.rounds, preserve_ranking=bounded,
                final_snippet_chars=None if snippet_tokens is not None else 1000,
            ),
            max_steps=max_steps, max_tokens_per_step=max_tokens,
            max_finish_retries=2, seed=args.seed, search_limit=args.search_limit,
            max_snippet_chars=snippet_chars, rollout_timeout_seconds=900.0,
        )
        tasks = [
            AgentTask(task_id=query_id, question=question, language=BENCHMARK_LANGUAGES[args.benchmark],
                      corpus_id=identity["corpus_id"], corpus_revision=identity["corpus_revision"])
            for query_id, question in selected.items()
        ]
        args.output.mkdir(parents=True, exist_ok=False)
        episode_paths = {task.task_id: f"episodes/{index:04d}.json" for index, task in enumerate(tasks, 1)}
        manifest = {
            "version": 1, "status": "running", "benchmark": args.benchmark,
            "protocol": "closed-retrieval", "questions_sha256": digest_file(args.questions), "dataset_revision": args.dataset_revision,
            "task_ids": list(selected), "episodes": episode_paths, "code": code_revision(), "search": identity,
            "config": config.model_dump(mode="json"), "model": model.metadata,
            **({"search_observation": observation} if observation else {}),
            "concurrency": args.concurrency, "direct": args.direct, "question_offset": args.offset,
        }
        manifest_path = args.output / "manifest.json"
        write_json(manifest_path, manifest)
        started = perf_counter()
        recorded = 0
        status = "interrupted"
        runner = await stack.enter_async_context(RolloutRunner(model, search, config, concurrency=args.concurrency))
        with (args.output / "events.jsonl").open("w", encoding="utf-8", buffering=1) as events:
            try:
                async for result in runner.iter_episodes(tasks, on_event=lambda event: events.write(event.model_dump_json() + "\n")):
                    save_episode(result, args.output / episode_paths[result.task_id])
                    recorded += 1
                    print(f"{recorded}/{len(tasks)} {result.task_id} {result.stop_reason.value}", flush=True)
                status = "complete"
            finally:
                manifest.update(status=status, recorded=recorded, wall_seconds=perf_counter() - started)
                if isinstance(model, ReservedBudgetModel):
                    manifest["billing"] = {
                        "known_cost_usd": model.spent, "unsettled_reservation_usd": model.reserved,
                        "blocked": model.blocked,
                        "auth_retries": model.auth_retries,
                    }
                write_json(manifest_path, manifest)


def evaluate(args):
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    gold = read_evidence(args.qrels)
    if missing := set(manifest["task_ids"]) - gold.keys():
        raise ValueError(f"Для {len(missing)} задач прогона отсутствуют положительные qrels.")
    rows = []
    for task_id in manifest["task_ids"]:
        path = args.run / manifest["episodes"][task_id]
        if not path.exists():
            continue
        episode = EpisodeResult.model_validate_json(path.read_bytes())
        if episode.task_id != task_id or episode.task.corpus_revision != manifest["search"]["corpus_revision"]:
            raise ValueError("Сохранённая траектория не соответствует манифесту.")
        rows.append(score_retrieval(episode, gold[task_id], manifest["benchmark"]))
    summary = summarize_retrieval(rows, len(manifest["task_ids"]))
    report = {
        "benchmark": manifest["benchmark"], "qrels_sha256": digest_file(args.qrels),
        "run_manifest_sha256": digest_file(args.run / "manifest.json"),
        "summary": summary, "episodes": rows,
    }
    usage_file = args.run / "embedding-usage.json"
    if usage_file.exists():
        usage = json.loads(usage_file.read_bytes())
        usage_id = manifest["search"].get("usage_id")
        if not usage_id or usage["usage_id"] != usage_id:
            raise ValueError("Расходы эмбеддера не относятся к этому прогону.")
        report["embedding_usage"] = {**usage, "source_sha256": digest_file(usage_file)}
    write_json(args.report, report)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    builder = modes.add_parser("build")
    builder.add_argument("--corpus", type=Path, required=True)
    builder.add_argument("--corpus-id", required=True)
    builder.add_argument("--corpus-revision")
    builder.add_argument("--bm25", type=Path, required=True)
    builder.add_argument("--faiss", type=Path)
    builder.add_argument("--embeddings-url", default=JINA_URL)
    builder.add_argument("--batch-size", type=int, default=8)

    runner = modes.add_parser("run")
    runner.add_argument("--benchmark", choices=BENCHMARK_LANGUAGES, required=True)
    runner.add_argument("--questions", type=Path, required=True)
    runner.add_argument("--dataset-revision", required=True)
    runner.add_argument("--output", type=Path, required=True)
    index = runner.add_mutually_exclusive_group(required=True)
    index.add_argument("--bm25", type=Path)
    index.add_argument("--reference-url")
    runner.add_argument("--reference-manifest", type=Path)
    runner.add_argument("--faiss", type=Path)
    runner.add_argument("--embeddings-url", default=JINA_URL)
    runner.add_argument("--mode", choices=("hybrid", "dense", "bm25"), default="hybrid")
    llm = runner.add_mutually_exclusive_group(required=True)
    llm.add_argument("--direct", action="store_true")
    llm.add_argument("--model-id")
    runner.add_argument("--budget-usd", type=float)
    runner.add_argument("--request-reserve-usd", type=float)
    runner.add_argument("--openrouter-max-price", type=float, nargs=2, metavar=("INPUT", "OUTPUT"))
    runner.add_argument("--openrouter-provider")
    runner.add_argument("--request-timeout", type=float, default=120)
    runner.add_argument("--max-tokens", type=int)
    runner.add_argument("--max-steps", type=int, help="Общий лимит вызовов модели за агентный эпизод.")
    runner.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    runner.add_argument("--retrieval-policy", choices=("bounded", "baseline"), default="bounded")
    runner.add_argument("--search-limit", type=int, default=10)
    snippet = runner.add_mutually_exclusive_group()
    snippet.add_argument("--search-snippet-chars", type=int)
    snippet.add_argument("--search-snippet-tokens", type=int)
    runner.add_argument("--search-tokenizer", type=Path)
    runner.add_argument("--rounds", type=int, default=5)
    runner.add_argument("--count", type=int, default=100)
    runner.add_argument("--offset", type=int, default=0)
    runner.add_argument("--concurrency", type=int, default=4)
    runner.add_argument("--seed", type=int, default=42)

    evaluator = modes.add_parser("evaluate")
    evaluator.add_argument("--run", type=Path, required=True)
    evaluator.add_argument("--qrels", type=Path, required=True)
    evaluator.add_argument("--report", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args)
    elif args.command == "build":
        asyncio.run(build(args))
    else:
        if args.count < 1 or args.concurrency < 1 or not 1 <= args.search_limit <= 10:
            runner.error("count/concurrency должны быть положительными, search-limit должен быть в диапазоне 1–10.")
        if args.offset < 0:
            runner.error("offset должен быть неотрицательным.")
        if args.max_steps is not None and (args.direct or args.max_steps < 1):
            runner.error("max-steps должен быть положительным и доступен только для агентного режима.")
        if args.search_snippet_chars is not None and args.search_snippet_chars < 1:
            runner.error("search-snippet-chars должен быть положительным.")
        if args.search_snippet_tokens is not None and args.search_snippet_tokens < 1:
            runner.error("search-snippet-tokens должен быть положительным.")
        qwen = (args.model_id or "").casefold() == QWEN_CHAT_MODEL
        if qwen and args.retrieval_policy == "bounded" and args.search_snippet_tokens is None and args.search_snippet_chars is None:
            args.search_snippet_tokens = DEFAULT_SEARCH_SNIPPET_TOKENS
        if args.search_tokenizer is not None and args.search_snippet_tokens is None:
            runner.error("search-tokenizer требует активного ограничения в токенах.")
        if args.search_snippet_tokens is not None and args.search_tokenizer is None and not qwen:
            runner.error("Для токенизации без Qwen3.6-35B-A3B укажите --search-tokenizer.")
        if args.direct and args.reasoning_effort is not None:
            runner.error("reasoning-effort доступен только для агентного режима.")
        if args.reference_url and not args.reference_manifest:
            runner.error("--reference-url требует --reference-manifest.")
        if args.bm25 and args.mode != "bm25" and not args.faiss:
            runner.error("Для hybrid или dense нужен готовый --faiss.")
        if args.model_id and not all((args.budget_usd, args.request_reserve_usd, args.openrouter_max_price)):
            runner.error("Укажите бюджет, резерв на запрос и максимальные тарифы OpenRouter.")
        asyncio.run(run(args))


if __name__ == "__main__":
    main()

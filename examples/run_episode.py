"""Один синтетический эпизод; по умолчанию полностью без сети."""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import TypeAdapter

from search_rl_env.agent import run_episode, save_episode
from search_rl_env.agent.client import OpenRouterClient
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit
from search_rl_env.contracts.types import StopReason

EXAMPLES_DIR = Path(__file__).resolve().parent


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mock", "openrouter"], default="mock")
    parser.add_argument("--model-id", help="OpenRouter model id; также OPENROUTER_MODEL")
    parser.add_argument("--task", type=Path, default=EXAMPLES_DIR / "fixtures/task.json")
    parser.add_argument("--hits", type=Path, default=EXAMPLES_DIR / "fixtures/search_hits.json")
    parser.add_argument("--output", type=Path, default=Path("runs/demo/episode.json"))
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-searches", type=int, default=None, help="Необязательный отдельный лимит поисков")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens-per-step", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = AgentTask.model_validate_json(args.task.read_text(encoding="utf-8"))
    hits = TypeAdapter(list[SearchHit]).validate_json(args.hits.read_text(encoding="utf-8"))
    search = MockSearch(hits, corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)
    config = EpisodeConfig(
        max_steps=args.max_steps,
        max_searches=args.max_searches,
        max_output_tokens=args.max_output_tokens,
        max_tokens_per_step=args.max_tokens_per_step,
        seed=args.seed,
    )
    if args.model == "mock":
        result = await run_episode(task, MockModel(), search, config)
    else:
        load_dotenv(EXAMPLES_DIR.parent / ".env")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model_id = args.model_id or os.getenv("OPENROUTER_MODEL", "")
        if not api_key or not model_id:
            parser.error("Для OpenRouter нужны OPENROUTER_API_KEY и --model-id или OPENROUTER_MODEL.")
        async with OpenRouterClient(api_key=api_key, model=model_id) as model:
            result = await run_episode(task, model, search, config)

    path = save_episode(result, args.output)
    print(f"stop_reason={result.stop_reason.value}; documents={result.selected_document_ids}; result={path}")
    return 0 if result.stop_reason == StopReason.FINISHED else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

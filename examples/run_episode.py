"""Один синтетический эпизод; по умолчанию полностью без сети."""

import argparse
import asyncio
import json
import math
import os
import sys
from contextlib import suppress
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from pydantic import TypeAdapter

from search_rl_env.agent import run_episode, save_episode
from search_rl_env.agent.auth import load_api_key
from search_rl_env.agent.client import ChatCompletionsClient, OpenRouterClient
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, EpisodeResult, SearchHit
from search_rl_env.contracts.types import EpisodeEvent, StopReason

EXAMPLES_DIR = Path(__file__).resolve().parent


class TerminalProgress:
    def __init__(self) -> None:
        self.waiting_for: str | None = None
        self.waiting_since = perf_counter()

    def __call__(self, event: EpisodeEvent) -> None:
        details = event.details
        if event.kind == "model_started":
            self.waiting_for = "ответ модели"
            self.waiting_since = perf_counter()
            print(f"\nШаг {event.step}: отправляю запрос модели; max_tokens={details['max_output_tokens']}", flush=True)
        elif event.kind == "model_finished":
            self.waiting_for = None
            print(f"  Модель {details['model']} ответила за {details['duration_seconds']:.1f} с; "
                  f"выходных токенов={details['usage']['completion_tokens']}", flush=True)
        elif event.kind == "tool_started":
            self.waiting_for = f"результат {details['name']}"
            self.waiting_since = perf_counter()
            print(f"  {details['name']}({details['arguments']})", flush=True)
        elif event.kind == "tool_finished":
            self.waiting_for = None
            print(f"  → {json.dumps(details['result'], ensure_ascii=False)}", flush=True)

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(5)
            if self.waiting_for is not None:
                elapsed = perf_counter() - self.waiting_since
                print(f"  Ожидаю {self.waiting_for}: {elapsed:.0f} с…", flush=True)


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("Нужно конечное положительное число секунд.")
    return seconds


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mock", "api", "openrouter"], default="mock")
    parser.add_argument("--model-id", help="Model id; также MODEL_ID или OPENROUTER_MODEL")
    parser.add_argument("--question", help="Заменить вопрос в задаче своим текстом")
    parser.add_argument("--verbose", action="store_true", help="Показывать шаги и ожидание ответа по ходу эпизода")
    parser.add_argument("--request-timeout", type=positive_seconds, default=30.0, help="Общий таймаут одного запроса модели в секундах (30)")
    parser.add_argument("--task", type=Path, default=EXAMPLES_DIR / "fixtures/task.json")
    parser.add_argument("--hits", type=Path, default=EXAMPLES_DIR / "fixtures/search_hits.json")
    parser.add_argument("--output", type=Path, default=Path("runs/demo/episode.json"))
    parser.add_argument("--max-steps", type=int, default=None, help="Необязательный лимит ответов модели")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Необязательный суммарный бюджет генерации")
    parser.add_argument("--max-tokens-per-step", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None, help="Необязательный seed, если провайдер его поддерживает")
    args = parser.parse_args()

    task = AgentTask.model_validate_json(args.task.read_text(encoding="utf-8"))
    if args.question is not None:
        task = AgentTask.model_validate({**task.model_dump(), "question": args.question})
    hits = TypeAdapter(list[SearchHit]).validate_json(args.hits.read_text(encoding="utf-8"))
    search = MockSearch(hits, corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)
    config = EpisodeConfig(
        max_steps=args.max_steps,
        max_output_tokens=args.max_output_tokens,
        max_tokens_per_step=args.max_tokens_per_step,
        seed=args.seed,
    )
    progress = TerminalProgress() if args.verbose else None
    interrupted = False

    async def execute_episode(model) -> EpisodeResult:
        nonlocal interrupted
        partial: EpisodeResult | None = None

        def remember_partial(result: EpisodeResult) -> None:
            nonlocal partial
            partial = result

        heartbeat = asyncio.create_task(progress.heartbeat()) if progress is not None else None
        try:
            return await run_episode(task, model, search, config, on_event=progress, on_interrupted=remember_partial)
        except asyncio.CancelledError:
            if partial is None:
                raise
            interrupted = True
            return partial
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat

    print(f"Запуск: {args.model}; таймаут запроса модели={args.request_timeout:g} с; поиск=mock", flush=True)
    if args.model == "mock":
        result = await execute_episode(MockModel())
    elif args.model == "api":
        load_dotenv(EXAMPLES_DIR.parent / ".env")
        model_id = args.model_id or os.getenv("MODEL_ID", "")
        endpoint = os.getenv("MODEL_API_URL", "")
        if not model_id or not endpoint:
            parser.error("Для API нужны MODEL_API_URL и --model-id или MODEL_ID.")
        try:
            api_key = load_api_key()
        except ValueError as exc:
            parser.error(str(exc))
        async with ChatCompletionsClient(
            api_key=api_key, model=model_id, endpoint=endpoint,
            auth_scheme=os.getenv("MODEL_AUTH_SCHEME", "Bearer"),
            timeout_seconds=args.request_timeout,
        ) as model:
            result = await execute_episode(model)
    else:
        load_dotenv(EXAMPLES_DIR.parent / ".env")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model_id = args.model_id or os.getenv("OPENROUTER_MODEL", "")
        if not api_key or not model_id:
            parser.error("Для OpenRouter нужны OPENROUTER_API_KEY и --model-id или OPENROUTER_MODEL.")
        async with OpenRouterClient(api_key=api_key, model=model_id, timeout_seconds=args.request_timeout) as model:
            result = await execute_episode(model)

    output = args.output
    if interrupted:
        output = output.with_name(f"{output.stem}.interrupted.json")
    path = save_episode(result, output)
    if interrupted:
        print(f"\nПрервано. Частичная траектория сохранена: {path}", flush=True)
    if args.verbose:
        print(f"\nUsage: {result.usage.model_dump_json()}", flush=True)
    if result.error:
        print(f"error={result.error.code}: {result.error.message}")
    print(f"stop_reason={result.stop_reason.value}; documents={result.selected_document_ids}; result={path}")
    return 130 if interrupted else (0 if result.stop_reason == StopReason.FINISHED else 1)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nПрервано до сохранения результата.", file=sys.stderr, flush=True)
        raise SystemExit(130) from None

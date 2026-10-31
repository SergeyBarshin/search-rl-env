"""Параллельные синтетические эпизоды; default — моки без внешних API."""

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
from contextlib import AsyncExitStack, aclosing
from pathlib import Path
from time import perf_counter

import httpx
from dotenv import load_dotenv

from search_rl_env.agent import RolloutRequest, RolloutRunner, save_episode
from search_rl_env.agent.auth import load_api_key
from search_rl_env.agent.client import ChatCompletionsClient
from search_rl_env.agent.mocks import MockModel, MockSearch, MockSqlRagModel
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit
from search_rl_env.environment.demo import HYBRID_HITS, HYBRID_REFERENCE, hybrid_task
from search_rl_env.environment.sql import MockSql


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Нужно положительное целое число.')
    return value


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['mock', 'api'], default='mock')
    parser.add_argument('--model-id')
    parser.add_argument('--scenario', choices=['retrieval', 'sql_rag'], default='sql_rag')
    parser.add_argument('--episodes', type=positive_int, default=32)
    parser.add_argument('--concurrency', type=positive_int, default=8)
    parser.add_argument('--execution', choices=['in_process', 'subprocess'], default='in_process')
    parser.add_argument('--mock-latency-ms', type=float, default=0., help='Синтетическое ожидание каждого ответа мока.')
    parser.add_argument('--output', type=Path, default=Path('runs/parallel'))
    parser.add_argument('--no-save', action='store_true', help='Не сохранять эпизоды; сводка сохраняется всегда.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=positive_int, default=6)
    parser.add_argument('--max-output-tokens', type=positive_int, default=4096)
    parser.add_argument('--max-tokens-per-step', type=positive_int, default=512)
    args = parser.parse_args()
    if not math.isfinite(args.mock_latency_ms) or args.mock_latency_ms < 0 or (args.mock_latency_ms and args.model != 'mock'):
        parser.error('--mock-latency-ms должен быть конечным >= 0 и используется только с --model mock.')

    hybrid = args.scenario == 'sql_rag'
    task = hybrid_task() if hybrid else AgentTask(task_id='demo', question='Когда перенесли срок?',
                                                language='ru', corpus_id='synthetic-demo', corpus_revision='v1')
    hits = HYBRID_HITS if hybrid else [SearchHit(doc_id='doc-001', snippet='Срок перенесён на ноябрь.', rank=1)]
    search = MockSearch(hits, corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)
    config = EpisodeConfig(max_steps=args.max_steps, max_output_tokens=args.max_output_tokens,
                           max_tokens_per_step=args.max_tokens_per_step, seed=args.seed)
    durations, rewards = [], []
    totals = dict(setup_seconds=0., model_seconds=0., tools_seconds=0., other_seconds=0.)
    stops = {}
    completion_tokens = completed = 0
    save_seconds = 0.
    args.output.mkdir(parents=True, exist_ok=True)
    wall_started = perf_counter()
    async with AsyncExitStack() as stack:
        if args.model == 'mock':
            model = MockSqlRagModel() if hybrid else MockModel()
            if args.mock_latency_ms:
                delegate = model

                class DelayedModel:
                    metadata = delegate.metadata

                    async def complete(self, *args_, **kwargs):
                        await asyncio.sleep(args.mock_latency_ms / 1000)
                        return await delegate.complete(*args_, **kwargs)

                model = DelayedModel()
        else:
            load_dotenv()
            http = await stack.enter_async_context(httpx.AsyncClient(timeout=None, limits=httpx.Limits(
                max_connections=args.concurrency, max_keepalive_connections=args.concurrency, keepalive_expiry=300,
            )))
            model = await stack.enter_async_context(ChatCompletionsClient(
                api_key=load_api_key(), model=args.model_id or os.environ.get('MODEL_ID', ''),
                endpoint=os.environ['MODEL_API_URL'], auth_scheme=os.getenv('MODEL_AUTH_SCHEME', 'Bearer'),
                http_client=http,
            ))
        runner = await stack.enter_async_context(RolloutRunner(
            model, search, config, concurrency=args.concurrency, sql=MockSql() if hybrid else None,
            execution=args.execution))
        initialized = perf_counter() - wall_started
        sampling_started = perf_counter()
        requests = (RolloutRequest(task.model_copy(update={'task_id': f'rollout-{index:06}'}),
            reference_answer=HYBRID_REFERENCE if hybrid else None,
            reference_document_ids=None if hybrid else ['doc-001']) for index in range(args.episodes))
        async with aclosing(runner.iter_episodes(requests)) as results:
            async for result in results:
                if not args.no_save:
                    started = perf_counter()
                    await asyncio.to_thread(save_episode, result, args.output / f'{result.task_id}-{result.episode_id}.json')
                    save_seconds += perf_counter() - started
                completed += 1
                durations.append(result.duration_seconds)
                if result.reward is not None:
                    rewards.append(result.reward)
                for key in totals:
                    totals[key] += getattr(result.profile, key)
                stop = result.stop_reason.value
                stops[stop] = stops.get(stop, 0) + 1
                completion_tokens = (completion_tokens + result.usage.completion_tokens
                                     if completion_tokens is not None and result.usage.completion_tokens is not None else None)
        sampling_seconds = perf_counter() - sampling_started
    wall = perf_counter() - wall_started
    summary = {
        'model': args.model, 'scenario': args.scenario, 'episodes': completed, 'concurrency': args.concurrency,
        'execution': args.execution, 'mock_latency_ms': args.mock_latency_ms,
        'model_id': model.metadata.get('model'), 'framework_version': '0.4.0',
        'corpus_id': task.corpus_id, 'corpus_revision': task.corpus_revision,
        'python': platform.python_version(), 'platform': platform.platform(),
        'config': config.model_dump(), 'initialization_seconds': initialized,
        'wall_seconds': wall, 'sampling_seconds': sampling_seconds, 'save_seconds': save_seconds,
        'episodes_per_second': completed / wall, 'steady_episodes_per_second': completed / sampling_seconds,
        'completion_tokens_per_second': completion_tokens / wall if completion_tokens is not None else None,
        'median_episode_seconds': statistics.median(durations),
        'p95_episode_seconds': sorted(durations)[max(0, int(.95 * len(durations) + .999) - 1)],
        'mean_reward': statistics.mean(rewards) if rewards else None, 'stop_reasons': stops,
        'mean_profile': {key: value / completed for key, value in totals.items()},
        'saved_episodes': not args.no_save,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if stops == {'finished': args.episodes} else 1


if __name__ == '__main__':
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None

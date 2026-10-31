"""Локальный UI: существующий run_episode, события SSE и сохранение JSON."""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import time, perf_counter
from typing import Literal
from uuid import uuid4

import httpx
from aiohttp import web
from dotenv import load_dotenv
from pydantic import Field, ValidationError, field_validator

from search_rl_env.agent import run_episode, save_episode
from search_rl_env.agent.auth import load_api_key
from search_rl_env.agent.client import ChatCompletionsClient
from search_rl_env.agent.mocks import MockModel, MockSearch, MockSqlRagModel
from search_rl_env.environment.demo import HYBRID_HITS, HYBRID_QUESTION, hybrid_task
from search_rl_env.environment.sql import MockSql, ORDERS, ORDERS_SCHEMA
from search_rl_env.contracts import AgentTask, EpisodeConfig, EpisodeResult, SearchHit
from search_rl_env.contracts.types import Contract, EpisodeEvent
from search_rl_env.profiling import EventLog, event_logger

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_QUESTION = "Какие документы описывают перенос срока проекта?"
DEMO_HITS = [
    SearchHit(doc_id="doc-001", rank=1, snippet="Протокол встречи: срок проекта перенесён на 15 ноября из-за задержки поставки."),
    SearchHit(doc_id="doc-002", rank=2, snippet="Обновлённый план проекта: новая дата завершения — 15 ноября."),
]


class RunRequest(Contract):
    question: str = Field(min_length=1)
    mode: Literal["mock", "api"] = "mock"
    scenario: Literal["retrieval", "sql_rag"] = "retrieval"
    model_id: str = ""
    max_steps: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tokens_per_step: int = Field(default=2048, ge=1)
    request_timeout: float = Field(default=60, gt=0, allow_inf_nan=False)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Введите вопрос.")
        return value.strip()


@dataclass
class Run:
    id: str
    request: RunRequest
    path: Path
    events: list[dict] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    done: bool = False
    result: EpisodeResult | None = None
    created: float = field(default_factory=perf_counter)
    event_log: EventLog | None = None
    finished: float | None = None

    def publish(self, kind: str, data: dict) -> None:
        self.events.append({"seq": len(self.events) + 1, "type": kind, "time": time(), "data": data})
        if self.event_log is None:
            self.event_log = EventLog(self.path.with_name('events.jsonl'), run_id=self.id)
        self.event_log.write(kind, data, seq=len(self.events))
        self.changed.set()

    def progress(self, event: EpisodeEvent) -> None:
        self.publish("progress", event.model_dump(mode="json"))


@dataclass
class Workbench:
    output_dir: Path
    runs: dict[str, Run] = field(default_factory=dict)
    http: httpx.AsyncClient | None = None

    async def execute(self, run: Run) -> None:
        partial = None

        def remember(result: EpisodeResult) -> None:
            nonlocal partial
            partial = result

        run.publish("started", {"mode": run.request.mode, 'scenario': run.request.scenario,
                                'queue_seconds': perf_counter() - run.created})
        try:
            async with AsyncExitStack() as stack:
                if run.request.mode == "mock":
                    model = MockSqlRagModel() if run.request.scenario == 'sql_rag' else MockModel()
                else:
                    model = await stack.enter_async_context(ChatCompletionsClient(
                        api_key=load_api_key(),
                        endpoint=os.environ["MODEL_API_URL"],
                        model=run.request.model_id or os.getenv("MODEL_ID", ""),
                        auth_scheme=os.getenv("MODEL_AUTH_SCHEME", "Bearer"),
                        timeout_seconds=run.request.request_timeout,
                        http_client=self.http,
                    ))
                task = AgentTask(task_id="ui-" + run.id, question=run.request.question,
                                 language="ru", corpus_id="synthetic-demo", corpus_revision="v1")
                hybrid = run.request.scenario == 'sql_rag'
                if hybrid:
                    task = hybrid_task(task_id='ui-' + run.id, question=run.request.question)
                search = MockSearch(HYBRID_HITS if hybrid else DEMO_HITS,
                                    corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)
                config = EpisodeConfig(max_steps=run.request.max_steps,
                                       max_output_tokens=run.request.max_output_tokens,
                                       max_tokens_per_step=run.request.max_tokens_per_step)
                try:
                    run.result = await run_episode(task, model, search, config,
                                                   on_event=run.progress, on_interrupted=remember,
                                                   sql=MockSql() if hybrid else None)
                except asyncio.CancelledError:
                    run.result = partial
            if run.result is None:
                run.publish("cancelled", {"message": "Запуск остановлен до начала эпизода."})
            else:
                save_started = perf_counter()
                save_episode(run.result, run.path)
                run.publish('persisted', {'duration_seconds': perf_counter() - save_started})
                run.publish("complete", {
                    "result": run.result.model_dump(mode="json"),
                    "download_url": f"/api/runs/{run.id}/result",
                    "saved_to": str(run.path),
                    'events_url': f'/api/runs/{run.id}/events-log',
                    'logging_error': run.event_log.error if run.event_log else None,
                })
        except asyncio.CancelledError:
            run.publish("cancelled", {"message": "Запуск остановлен."})
        except Exception as exc:
            # Произвольный текст ошибок может включать credentials или endpoint.
            run.publish("error", {"message": f"Не удалось выполнить запуск ({type(exc).__name__})."})
        finally:
            run.done = True
            run.finished = perf_counter()
            run.changed.set()


WORKBENCH = web.AppKey("workbench", Workbench)


@web.middleware
async def local_requests(request: web.Request, handler):
    if request.url.host not in {"127.0.0.1", "localhost", "::1"}:
        raise web.HTTPForbidden(text="Доступен только локальный интерфейс.")
    origin = request.headers.get("Origin")
    if origin is not None and origin != f"{request.scheme}://{request.host}":
        raise web.HTTPForbidden(text="Запрос с другого сайта запрещён.")
    response = await handler(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def api_configuration_error(model_id: str = "") -> str | None:
    if not os.getenv("MODEL_API_URL", "").strip():
        return "Задайте MODEL_API_URL в локальном .env и перезапустите UI."
    if not (model_id or os.getenv("MODEL_ID", "")).strip():
        return "Укажите модель в интерфейсе или MODEL_ID в .env."
    try:
        load_api_key()
    except ValueError as exc:
        return str(exc)
    return None


async def index(request):
    return web.FileResponse(STATIC_DIR / "index.html")


async def settings(request):
    return web.json_response({
        "question": DEFAULT_QUESTION,
        "model_id": os.getenv("MODEL_ID", ""),
        "api_error": api_configuration_error(),
        "hits": [hit.model_dump() for hit in DEMO_HITS],
        "corpus_id": "synthetic-demo",
        "scenarios": [
            {"id": "retrieval", "question": DEFAULT_QUESTION, "hits": [hit.model_dump() for hit in DEMO_HITS]},
            {"id": "sql_rag", "question": HYBRID_QUESTION, "hits": [hit.model_dump() for hit in HYBRID_HITS],
             "sql_schema": ORDERS_SCHEMA, "columns": ['order_id', 'status', 'delay_days', 'region'],
             "rows": [list(row) for row in ORDERS]},
        ],
    })


async def start_run(request):
    if request.content_type != "application/json":
        raise web.HTTPUnsupportedMediaType()
    try:
        data = RunRequest.model_validate(await request.json())
    except (ValueError, ValidationError):
        return web.json_response({"error": "Проверьте вопрос и параметры запуска: числа должны быть положительными."}, status=400)
    if data.mode == "api" and (error := api_configuration_error(data.model_id)):
        return web.json_response({"error": error}, status=400)
    workbench = request.app[WORKBENCH]
    run_id = uuid4().hex
    run = Run(run_id, data, workbench.output_dir / run_id / "episode.json")
    workbench.runs[run_id] = run
    run.task = asyncio.create_task(workbench.execute(run), name="ui-" + run_id)

    def completed(task):
        # Отмена возможна до первого исполнения coroutine.
        if not run.done:
            run.publish("cancelled", {"message": "Запуск остановлен до начала эпизода."})
            run.done = True
            run.finished = perf_counter()
            run.changed.set()

    run.task.add_done_callback(completed)
    return web.json_response({"id": run.id}, status=202)


def get_run(request) -> Run:
    workbench = request.app[WORKBENCH]
    run_id = request.match_info['run_id']
    run = workbench.runs.get(run_id)
    # Ссылки на сохранённый результат продолжают работать после обновления UI.
    if run is None and re.fullmatch(r'[a-f0-9]{32}', run_id):
        path = workbench.output_dir / run_id / 'episode.json'
        if path.is_file():
            try:
                result = EpisodeResult.model_validate_json(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                raise web.HTTPNotFound() from None
            hybrid = (result.native_trace or {}).get('task', {}).get('data', {}).get('sql_schema') is not None
            saved_request = RunRequest(question=result.task.question,
                mode='mock' if result.model_metadata.get('adapter') == 'mock' else 'api',
                scenario='sql_rag' if hybrid else 'retrieval')
            run = Run(run_id, saved_request, path, done=True, result=result)
            run.finished = perf_counter()
            run.created = run.finished - result.duration_seconds
            run.events = [{'seq': 1, 'type': 'complete', 'time': path.stat().st_mtime, 'data': {
                'result': result.model_dump(mode='json'), 'download_url': f'/api/runs/{run_id}/result',
                'saved_to': str(path), 'events_url': f'/api/runs/{run_id}/events-log' if path.with_name('events.jsonl').exists() else None,
            }}]
            workbench.runs[run_id] = run
    if run is None:
        raise web.HTTPNotFound()
    return run


async def events(request):
    run = get_run(request)
    try:
        cursor = max(0, int(request.headers.get("Last-Event-ID", "0")))
    except ValueError:
        raise web.HTTPBadRequest() from None
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store"})
    await response.prepare(request)
    try:
        while True:
            run.changed.clear()
            for event in run.events[cursor:]:
                await response.write((f"id: {event['seq']}\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode())
                cursor = event["seq"]
            if run.done and cursor >= len(run.events):
                break
            try:
                await asyncio.wait_for(run.changed.wait(), timeout=10)
            except TimeoutError:
                await response.write(b": waiting\n\n")
    except (ConnectionResetError, ConnectionError):
        pass  # Закрытие вкладки не отменяет уже запущенный эпизод.
    return response


async def cancel_run(request):
    run = get_run(request)
    if not run.done and run.task is not None and not run.task.cancelling():
        run.task.cancel()
    return web.json_response({"done": run.done})


async def download(request):
    run = get_run(request)
    if not run.done or run.result is None or not run.path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(run.path, headers={
        "Content-Disposition": f'attachment; filename="episode-{run.id[:8]}.json"',
    })


async def download_events(request):
    run = get_run(request)
    path = run.path.with_name('events.jsonl')
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={
        'Content-Type': 'application/x-ndjson',
        'Content-Disposition': f'attachment; filename="events-{run.id[:8]}.jsonl"',
    })


async def run_statuses(request):
    runs = request.app[WORKBENCH].runs.values()
    statuses = []
    for run in runs:
        last = next((event['data'] for event in reversed(run.events) if event['type'] == 'progress'), {})
        statuses.append({'id': run.id, 'done': run.done, 'mode': run.request.mode, 'scenario': run.request.scenario,
         'elapsed_seconds': (run.finished or perf_counter()) - run.created,
         'episode_seconds': run.result.duration_seconds if run.result else None,
         'last_event': run.events[-1]['type'] if run.events else None,
         'phase': last.get('details', {}).get('phase') or last.get('kind'),
         'step': last.get('step'), 'episode_id': last.get('episode_id')})
    return web.json_response({'runs': statuses})


async def cleanup(app):
    tasks = [run.task for run in app[WORKBENCH].runs.values() if run.task is not None and not run.task.done()]
    for task in tasks:
        if not task.cancelling():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def http_pool(app):
    workbench = app[WORKBENCH]
    # Полный deadline остаётся на каждом вызове модели; пул принадлежит UI,
    # а обёртки ChatCompletionsClient не закрывают чужой клиент.
    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(
        max_connections=20, max_keepalive_connections=20, keepalive_expiry=300,
    )) as client:
        workbench.http = client
        yield
        workbench.http = None


def create_app(*, output_dir: str | Path = "runs/ui") -> web.Application:
    app = web.Application(middlewares=[local_requests])
    app[WORKBENCH] = Workbench(Path(output_dir))
    app.add_routes([
        web.get("/", index),
        web.get("/api/settings", settings),
        web.post("/api/runs", start_run),
        web.get('/api/runs', run_statuses),
        web.get("/api/runs/{run_id}/events", events),
        web.post("/api/runs/{run_id}/cancel", cancel_run),
        web.get("/api/runs/{run_id}/result", download),
        web.get('/api/runs/{run_id}/events-log', download_events),
        web.static("/assets", STATIC_DIR),
    ])
    app.on_shutdown.append(cleanup)
    app.cleanup_ctx.append(http_pool)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ui"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    load_dotenv(args.env_file)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    event_logger.addHandler(handler)
    event_logger.setLevel(logging.INFO)
    event_logger.propagate = False
    print("Search Lab · локальная проверка харнесса", flush=True)
    try:
        web.run_app(create_app(output_dir=args.output_dir), host="127.0.0.1", port=args.port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.exit(2, f"Порт {args.port} занят. Укажите другой: --port {args.port + 1}\n")
        raise


if __name__ == "__main__":
    main()

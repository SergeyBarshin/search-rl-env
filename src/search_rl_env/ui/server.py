"""Локальный UI: существующий run_episode, события SSE и сохранение JSON."""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal
from uuid import uuid4

from aiohttp import web
from dotenv import load_dotenv
from pydantic import Field, ValidationError, field_validator

from search_rl_env.agent import run_episode, save_episode
from search_rl_env.agent.auth import load_api_key
from search_rl_env.agent.client import ChatCompletionsClient
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, EpisodeResult, SearchHit
from search_rl_env.contracts.types import Contract, EpisodeEvent

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_QUESTION = "Какие документы описывают перенос срока проекта?"
DEMO_HITS = [
    SearchHit(doc_id="doc-001", rank=1, snippet="Протокол встречи: срок проекта перенесён на 15 ноября из-за задержки поставки."),
    SearchHit(doc_id="doc-002", rank=2, snippet="Обновлённый план проекта: новая дата завершения — 15 ноября."),
]


class RunRequest(Contract):
    question: str = Field(min_length=1)
    mode: Literal["mock", "api"] = "mock"
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

    def publish(self, kind: str, data: dict) -> None:
        self.events.append({"seq": len(self.events) + 1, "type": kind, "time": time(), "data": data})
        self.changed.set()

    def progress(self, event: EpisodeEvent) -> None:
        self.publish("progress", event.model_dump(mode="json"))


@dataclass
class Workbench:
    output_dir: Path
    runs: dict[str, Run] = field(default_factory=dict)

    async def execute(self, run: Run) -> None:
        partial = None

        def remember(result: EpisodeResult) -> None:
            nonlocal partial
            partial = result

        run.publish("started", {"mode": run.request.mode})
        try:
            async with AsyncExitStack() as stack:
                if run.request.mode == "mock":
                    model = MockModel()
                else:
                    model = await stack.enter_async_context(ChatCompletionsClient(
                        api_key=load_api_key(),
                        endpoint=os.environ["MODEL_API_URL"],
                        model=run.request.model_id or os.getenv("MODEL_ID", ""),
                        auth_scheme=os.getenv("MODEL_AUTH_SCHEME", "Bearer"),
                        timeout_seconds=run.request.request_timeout,
                    ))
                task = AgentTask(task_id="ui-" + run.id, question=run.request.question,
                                 language="ru", corpus_id="synthetic-demo", corpus_revision="v1")
                search = MockSearch(DEMO_HITS, corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)
                config = EpisodeConfig(max_steps=run.request.max_steps,
                                       max_output_tokens=run.request.max_output_tokens,
                                       max_tokens_per_step=run.request.max_tokens_per_step)
                try:
                    run.result = await run_episode(task, model, search, config,
                                                   on_event=run.progress, on_interrupted=remember)
                except asyncio.CancelledError:
                    run.result = partial
            if run.result is None:
                run.publish("cancelled", {"message": "Запуск остановлен до начала эпизода."})
            else:
                save_episode(run.result, run.path)
                run.publish("complete", {
                    "result": run.result.model_dump(mode="json"),
                    "download_url": f"/api/runs/{run.id}/result",
                    "saved_to": str(run.path),
                })
        except asyncio.CancelledError:
            run.publish("cancelled", {"message": "Запуск остановлен."})
        except Exception as exc:
            # Произвольный текст ошибок может включать credentials или endpoint.
            run.publish("error", {"message": f"Не удалось выполнить запуск ({type(exc).__name__})."})
        finally:
            run.done = True
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
            run.changed.set()

    run.task.add_done_callback(completed)
    return web.json_response({"id": run.id}, status=202)


def get_run(request) -> Run:
    run = request.app[WORKBENCH].runs.get(request.match_info["run_id"])
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


async def cleanup(app):
    tasks = [run.task for run in app[WORKBENCH].runs.values() if run.task is not None and not run.task.done()]
    for task in tasks:
        if not task.cancelling():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def create_app(*, output_dir: str | Path = "runs/ui") -> web.Application:
    app = web.Application(middlewares=[local_requests])
    app[WORKBENCH] = Workbench(Path(output_dir))
    app.add_routes([
        web.get("/", index),
        web.get("/api/settings", settings),
        web.post("/api/runs", start_run),
        web.get("/api/runs/{run_id}/events", events),
        web.post("/api/runs/{run_id}/cancel", cancel_run),
        web.get("/api/runs/{run_id}/result", download),
        web.static("/assets", STATIC_DIR),
    ])
    app.on_shutdown.append(cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ui"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    load_dotenv(args.env_file)
    print("Search Lab · локальная проверка харнесса", flush=True)
    try:
        web.run_app(create_app(output_dir=args.output_dir), host="127.0.0.1", port=args.port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.exit(2, f"Порт {args.port} занят. Укажите другой: --port {args.port + 1}\n")
        raise


if __name__ == "__main__":
    main()

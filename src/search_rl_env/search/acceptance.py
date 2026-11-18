"""Бесплатная приёмка SQL/FAISS-сред через публичный RolloutRunner."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from search_rl_env.agent import save_episode
from search_rl_env.contracts import AgentTask, EpisodeConfig
from search_rl_env.contracts.types import ModelAuthenticationError, ModelError, ModelResponse, ModelTimeoutError, Usage
from .environments import EnvironmentFactory
from .local import fingerprint, joined_thread


class AcceptanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    datasets: list[str] = Field(min_length=1)
    tasks_path: str
    episodes: int = Field(default=80, ge=1)
    warmup_episodes: int = Field(default=2, ge=0)
    concurrency: list[int] = Field(default_factory=lambda: [1, 40], min_length=1)
    mock_latency_seconds: float = Field(default=0.005, ge=0, allow_inf_nan=False)
    gate_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    sql_timeout_seconds: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    max_vm_steps: int = Field(default=100_000, ge=1)
    episode: EpisodeConfig = Field(default_factory=lambda: EpisodeConfig(max_steps=5, seed=42))

    @model_validator(mode="after")
    def bounded(self):
        if any(isinstance(n, bool) or n < 1 for n in self.concurrency) or len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("Параллельности должны быть уникальными положительными целыми числами.")
        if self.episode.max_steps is None:
            raise ValueError("Для приёмки нужен явный max_steps.")
        return self


class TaskBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_id: str
    task: AgentTask
    reference_path: str


class PrivateReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: list[str]
    document_ids: list[str]


class StartGate:
    """Однократный барьер первой волны; ждёт реальные обращения модели."""

    def __init__(self, expected, timeout):
        self.expected, self.timeout, self.arrivals = expected, timeout, 0
        self._ready = asyncio.Event()

    async def arrive(self):
        if self._ready.is_set():
            return
        self.arrivals += 1
        if self.arrivals >= self.expected:
            self._ready.set()
        async with asyncio.timeout(self.timeout):
            await self._ready.wait()


class ProbeModel:
    """Сценарный агент для механики; ответы берёт из результатов инструментов."""

    def __init__(self, gate, latency):
        self.gate, self.latency = gate, latency
        self.metadata = {"model": "sql-faiss-probe-v1", "adapter": "synthetic",
                         "latency_seconds": latency}

    async def complete(self, messages, **kwargs):
        observations = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if not observations:
            await self.gate.arrive()
        await asyncio.sleep(self.latency)
        if not observations:
            name, arguments = "sql", {"query": "SELECT value, keyword FROM facts WHERE fact_id='target'"}
        elif "rows" in observations[-1]:
            name, arguments = "search", {"query": str(observations[-1]["rows"][0][1])}
        else:
            hits = observations[-1].get("hits", [])
            name, arguments = "finish", {"answer": [str(observations[0]["rows"][0][0])],
                                          "document_ids": [hits[0]["doc_id"]] if hits else []}
        return ModelResponse(message={"role": "assistant", "content": None, "tool_calls": [{
            "id": f"probe-{len(observations)}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]},
            usage=Usage(prompt_tokens=0, completion_tokens=1, total_tokens=1, cost=0.0),
            finish_reason="tool_calls", model=self.metadata["model"])


class ReservedBudgetModel:
    """Резерв до dispatch; неизвестная стоимость/ошибка блокирует новые запросы.

    Размер резерва задаёт оператор по верхней оценке запроса. Это предохранитель
    клиента, не жёсткий лимит биллинга провайдера.
    """

    def __init__(self, delegate, budget_usd, request_reserve_usd, *, continue_after_timeout=False,
                 retry_auth_once=False, recoverable_errors: tuple[type[ModelError], ...] = ()):
        if (not all(math.isfinite(n) and n > 0 for n in (budget_usd, request_reserve_usd)) or
                request_reserve_usd > budget_usd):
            raise ValueError("Нужны конечный бюджет и положительный резерв на запрос <= бюджета.")
        self.delegate, self.budget, self.reserve = delegate, budget_usd, request_reserve_usd
        self.spent, self.reserved, self.blocked = 0.0, 0.0, False
        self._recoverable_errors = (*recoverable_errors, ModelTimeoutError) if continue_after_timeout else recoverable_errors
        self._retry_auth_once = retry_auth_once
        self.auth_retries = 0
        self._lock = asyncio.Lock()

    @property
    def metadata(self):
        return {**self.delegate.metadata, "budget_usd": self.budget, "request_reserve_usd": self.reserve}

    async def complete(self, *args, **kwargs):
        try:
            return await self._complete_once(*args, **kwargs)
        except ModelAuthenticationError:
            if not self._retry_auth_once:
                raise
            self.auth_retries += 1
            await asyncio.sleep(1)
            return await self._complete_once(*args, **kwargs)

    async def _complete_once(self, *args, **kwargs):
        async with self._lock:
            if self.blocked or self.spent + self.reserved + self.reserve > self.budget + 1e-12:
                raise ModelError("Бюджет модели исчерпан или стоимость предыдущего запроса неизвестна.")
            self.reserved += self.reserve
        try:
            response = await self.delegate.complete(*args, **kwargs)
        except ModelAuthenticationError:
            if self._retry_auth_once:
                async with self._lock:
                    self.reserved = max(0.0, self.reserved - self.reserve)
            else:
                self.blocked = True
            raise
        except self._recoverable_errors:
            raise
        except BaseException:
            self.blocked = True
            raise
        async with self._lock:
            self.reserved = max(0.0, self.reserved - self.reserve)
            if response.usage.cost is None:
                self.reserved += self.reserve
                self.blocked = True
                raise ModelError("Провайдер не сообщил стоимость; следующие запросы остановлены.")
            self.spent += response.usage.cost
            if response.usage.cost > self.reserve + 1e-12:
                self.blocked = True
                raise ModelError("Стоимость запроса превысила резерв; следующие запросы остановлены.")
        return response


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "p50_seconds": None, "p95_seconds": None}
    return {"count": len(values), "p50_seconds": values[math.ceil(len(values) * .5) - 1],
            "p95_seconds": values[math.ceil(len(values) * .95) - 1]}


def code_revision():
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=root)
    except (OSError, subprocess.CalledProcessError):
        commit, diff = os.environ.get("SEARCH_RL_SOURCE_COMMIT"), b""
    sources = []
    for directory in ("src", "services", "examples", "configs"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".go", ".mod", ".sum", ".json", ".jsonl", ".sql"}:
                sources.append([str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()])
    return {"base_commit": commit, "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_tree_sha256": fingerprint(sources), "upstream_backend_commit": "71116f200e9990b820a0141b48fc02d5619697f1"}


@asynccontextmanager
async def memory_sample():
    import psutil

    process = psutil.Process()
    state = {"peak_rss_bytes": 0, "samples": 0, "includes_children": True}
    done = asyncio.Event()

    async def sample():
        while True:
            rss = 0
            children = []
            if state["includes_children"]:
                try:
                    children = process.children(recursive=True)
                except (PermissionError, psutil.AccessDenied):
                    state["includes_children"] = False
            for item in [process, *children]:
                try:
                    rss += item.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            state["peak_rss_bytes"] = max(state["peak_rss_bytes"], rss)
            state["samples"] += 1
            if done.is_set():
                return
            try:
                await asyncio.wait_for(done.wait(), .02)
            except TimeoutError:
                pass

    sampler = asyncio.create_task(sample())
    try:
        yield state
    finally:
        done.set()
        await sampler


async def run_batch(factory, rows, references, config, concurrency, output, entries, model=None):
    from search_rl_env.agent import RolloutRunner

    gate = StartGate(min(concurrency, config.episodes), config.gate_timeout_seconds)
    model = model or ProbeModel(gate, config.mock_latency_seconds)
    source = iter(range(config.episodes))
    durations, sql_times, search_times, model_times, overhead, creation, teardown = [], [], [], [], [], [], []
    errors, finished, active, peak = [], 0, 0, 0

    def persist(result, environment, phase="measured"):
        folder = output / f"c{concurrency}" / phase
        trace = folder / (result.episode_id + ".episode.json")
        save_episode(result, trace)
        binding = {"version": 1, "task_id": result.task_id, "episode_id": result.episode_id,
                   "attempt_id": result.episode_id, "environment": environment.manifest,
                   "episode": str(trace.relative_to(output)), "status": result.stop_reason.value, "phase": phase}
        write_json(folder / (result.episode_id + ".binding.json"), binding)
        entries.append(binding)

    def progress(event, phase="measured"):
        if event.kind not in {"episode_started", "model_started", "model_progress", "tool_started", "tool_finished", "episode_finished"}:
            return
        line = json.dumps({"kind": event.kind, "episode_id": event.episode_id, "step": event.step,
                           "phase": phase, "concurrency": concurrency,
                           "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                           "elapsed_seconds": event.elapsed_seconds}, ensure_ascii=False)
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    warmup_started = perf_counter()
    warmup_model = ProbeModel(StartGate(1, config.gate_timeout_seconds), config.mock_latency_seconds) if isinstance(model, ProbeModel) else model
    if config.warmup_episodes:
        async with RolloutRunner(warmup_model, config=config.episode, concurrency=1) as runner:
            for index in range(config.warmup_episodes):
                row, reference = rows[index % len(rows)], references[index % len(rows)]
                async with factory.open(row.dataset_id) as environment:
                    environment.validate_task(row.task)
                    task = row.task.model_copy(update={"task_id": f"warmup-{row.task.task_id}-c{concurrency}-{index}",
                        "corpus_revision": environment.manifest["search"]["corpus_revision"]})
                    result = await runner.run(task, search=environment.search, sql=environment.sql,
                        reference_answer=reference.answer, reference_document_ids=reference.document_ids,
                        on_event=lambda event: progress(event, "warmup"),
                        on_interrupted=lambda partial: persist(partial, environment, "warmup"))
                    await joined_thread(persist, result, environment, "warmup")
                    if (result.stop_reason.value != "finished" or result.answer != reference.answer or
                            result.selected_document_ids != reference.document_ids or
                            result.sql_calls < 1 or result.search_calls < 1):
                        raise RuntimeError("Прогрев не прошёл проверку; измерение не запущено.")
    warmup_seconds = perf_counter() - warmup_started
    factory.peak_active = 0
    batch_started = perf_counter()
    async with memory_sample() as memory:
        async with RolloutRunner(model, config=config.episode, concurrency=concurrency) as runner:
            async def worker():
                nonlocal finished, active, peak
                for index in source:
                    row = rows[index % len(rows)]
                    reference = references[index % len(rows)]
                    async with factory.open(row.dataset_id) as environment:
                        task = row.task.model_copy(update={"task_id": f"{row.task.task_id}-c{concurrency}-{index:06d}",
                            "corpus_revision": environment.manifest["search"]["corpus_revision"]})
                        environment.validate_task(row.task)
                        active += 1
                        peak = max(peak, active)
                        try:
                            result = await runner.run(task, search=environment.search, sql=environment.sql,
                                reference_answer=reference.answer, reference_document_ids=reference.document_ids,
                                on_event=progress,
                                on_interrupted=lambda partial: persist(partial, environment))
                        finally:
                            active -= 1
                        await joined_thread(persist, result, environment)
                        correct = (result.stop_reason.value == "finished" and result.answer == reference.answer and
                                   result.selected_document_ids == reference.document_ids and
                                   result.sql_calls >= 1 and result.search_calls >= 1)
                        if correct:
                            finished += 1
                        else:
                            errors.append({"task_id": task.task_id, "stop_reason": result.stop_reason.value,
                                           "error_code": result.error.code if result.error else "reference_mismatch"})
                        durations.append(result.duration_seconds)
                        actions = [action for step in result.trajectory for action in step.actions]
                        sql_times.extend(action.duration_seconds for action in actions if action.name == "sql")
                        search_times.extend(action.duration_seconds for action in actions if action.name == "search")
                        if result.profile:
                            model_times.append(result.profile.model_seconds)
                            overhead.append(max(0, result.profile.wall_seconds - result.profile.model_seconds - result.profile.tools_seconds))
                        creation.append(environment.manifest["creation_seconds"])
                        closing = perf_counter()
                    teardown.append(perf_counter() - closing)
                    print(json.dumps({"kind": "episode_finished", "task_id": task.task_id,
                                      "episode_id": result.episode_id, "concurrency": concurrency,
                                      "completed": len(durations), "correct": correct}), flush=True)
            async with asyncio.TaskGroup() as group:
                for _ in range(min(concurrency, config.episodes)):
                    group.create_task(worker())
    wall = perf_counter() - batch_started
    expected_peak = min(concurrency, config.episodes)
    return {"concurrency": concurrency, "episodes": len(durations), "correct": finished, "errors": errors,
            "warmup_episodes": config.warmup_episodes, "warmup_seconds": warmup_seconds,
            "peak_active_episodes": peak, "peak_active_environments": factory.peak_active,
            "first_wave_model_arrivals": gate.arrivals if isinstance(model, ProbeModel) else None,
            "wall_seconds_including_setup_persistence_teardown": wall,
            "completed_episodes_per_minute": len(durations) / wall * 60,
            "episode": distribution(durations), "sql": distribution(sql_times), "search": distribution(search_times),
            "model": distribution(model_times), "harness_overhead": distribution(overhead),
            "environment_creation": distribution(creation), "environment_teardown": distribution(teardown),
            "memory": memory, "resources_released": factory.active_count == 0,
            "passed": finished == config.episodes and peak == expected_peak and
                      factory.peak_active == expected_peak and factory.active_count == 0}


async def run_acceptance(config_path, output, *, model=None):
    config_path, output = Path(config_path).resolve(), Path(output)
    config = AcceptanceConfig.model_validate_json(config_path.read_bytes())
    task_path = (config_path.parent / config.tasks_path).resolve()
    rows = [TaskBinding.model_validate(row) for row in json.loads(task_path.read_text(encoding="utf-8"))]
    if not rows:
        raise ValueError("Набор задач пуст.")
    references = [PrivateReference.model_validate_json((task_path.parent / row.reference_path).read_bytes()) for row in rows]
    output.mkdir(parents=True, exist_ok=False)
    report = {"version": 1, "run_id": uuid4().hex, "mode": "synthetic_environment_acceptance",
              "created_at": datetime.now(timezone.utc).isoformat(), "config": config.model_dump(mode="json"),
              "code": code_revision(), "python": platform.python_version(), "platform": platform.platform(),
              "cpu_count": os.cpu_count(), "cpu": platform.processor(),
              "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "status": "running", "batches": [], "episodes": [],
              "limitations": ["Техническая проверка на синтетике; не benchmark score.",
                              "Hashing encoder проверяет механизм FAISS, не качество semantic retrieval.",
                              "Численный SLO скорости пока не согласован."]}
    write_json(output / "manifest.json", report)
    factory = EnvironmentFactory([config_path.parent / path for path in config.datasets],
                                 sql_timeout_seconds=config.sql_timeout_seconds, max_vm_steps=config.max_vm_steps)
    try:
        async with factory:
            report["datasets"] = factory.manifests
            report["dataset_artifacts"] = await factory.export(output / "datasets")
            report["preparation_seconds"] = factory.preparation_seconds
            for row in rows:
                if row.dataset_id not in factory.manifests:
                    raise ValueError("Задача ссылается на неизвестный dataset_id.")
            for concurrency in config.concurrency:
                report["batches"].append(await run_batch(factory, rows, references, config, concurrency,
                                                       output, report["episodes"], model=model))
                write_json(output / "manifest.json", report)
        report["status"] = "passed" if all(batch["passed"] for batch in report["batches"]) else "failed"
    except asyncio.CancelledError:
        report["status"] = "cancelled"
        raise
    except BaseException as error:
        report["status"], report["error_type"] = "failed", type(error).__name__
        raise
    finally:
        report["resources_released"] = factory.active_count == 0 and factory.closed
        if isinstance(model, ReservedBudgetModel):
            report["budget"] = {"spent_usd": model.spent, "reserved_or_unknown_usd": model.reserved,
                                "blocked": model.blocked, "budget_usd": model.budget}
        write_json(output / "manifest.json", report)
    return report

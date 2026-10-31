"""Профили запросов и эпизодов; журнал не содержит HTTP-заголовки или тела."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

from search_rl_env.contracts.types import EpisodeEvent, EpisodeProfile, HttpPhase, HttpProfile

HttpObserver = Callable[[HttpProfile, str, str], None]
_http_observer: ContextVar[HttpObserver | None] = ContextVar("http_profile_observer", default=None)
event_logger = logging.getLogger('search_rl_env.profile')


@contextmanager
def observe_http(observer: HttpObserver):
    token = _http_observer.set(observer)
    try:
        yield
    finally:
        _http_observer.reset(token)


class HttpRecorder:
    def __init__(self):
        self.started = perf_counter()
        self.data = HttpProfile(request_id=uuid4().hex, duration_seconds=0)
        self.observer = _http_observer.get()
        self.pending: dict[str, float] = {}
        self.trace_seen = False
        self.connect_seen = False

    def snapshot(self) -> HttpProfile:
        now = perf_counter()
        data = self.data.model_copy(deep=True)
        if data.status == "running":
            data.duration_seconds = now - self.started
        for name, start in self.pending.items():
            data.phases.append(HttpPhase(name=name, offset_seconds=start - self.started,
                                         duration_seconds=now - start, status="running"))
        data.phases.sort(key=lambda phase: phase.offset_seconds)
        return data

    def notify(self, phase: str, status: str) -> None:
        if self.observer is not None:
            self.observer(self.snapshot(), phase, status)

    async def trace(self, event: str, info: dict) -> None:
        # В info есть URL, заголовки, сокеты и исключения. Сохраняем только
        # имя фазы, длительность и класс ошибки, а не произвольные значения.
        phase, _, status = event.rpartition(".")
        if not re.fullmatch(r"[a-z0-9_.]+", phase) or status not in {"started", "complete", "failed"}:
            return
        self.trace_seen = True
        if self.data.time_to_transport_seconds is None:
            self.data.time_to_transport_seconds = perf_counter() - self.started
        self.connect_seen |= phase.endswith(("connect_tcp", "connect_unix_socket", "start_tls"))
        now = perf_counter()
        if status == "started":
            self.pending[phase] = now
        else:
            start = self.pending.pop(phase, None)
            if start is not None:
                error = info.get("exception")
                self.data.phases.append(HttpPhase(
                    name=phase, offset_seconds=start - self.started, duration_seconds=now - start,
                    status="cancelled" if isinstance(error, BaseException) and type(error).__name__ == "CancelledError" else status,
                    error_type=type(error).__name__ if error is not None else None,
                ))
        self.notify(phase, status)

    def headers(self, response) -> None:
        self.data.time_to_headers_seconds = perf_counter() - self.started
        self.data.status_code = response.status_code
        self.data.http_version = response.http_version
        self.data.connection_reused = not self.connect_seen if self.trace_seen else None
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        if request_id and re.fullmatch(r"[A-Za-z0-9._:/=-]{1,160}", request_id):
            self.data.provider_request_id = request_id
        allowed = {"queue", "queue_time", "inference", "processing", "prefill", "decode", "ttft", "total", "upstream"}
        for metric in response.headers.get("server-timing", "").split(","):
            parts = metric.strip().split(";")
            name = parts[0].lower()
            if name not in allowed:
                continue
            for parameter in parts[1:]:
                match = re.fullmatch(r"\s*dur=([0-9.eE+-]+)\s*", parameter)
                if match:
                    try:
                        duration = float(match[1]) / 1000
                    except ValueError:
                        continue
                    if math.isfinite(duration) and duration >= 0:
                        self.data.provider_timing_seconds[name] = duration
        self.notify("response_headers", "complete")

    def body_chunk(self, length: int) -> None:
        if not length:
            return
        self.data.response_bytes += length
        if self.data.time_to_first_body_byte_seconds is None:
            self.data.time_to_first_body_byte_seconds = perf_counter() - self.started
            self.notify("first_body_byte", "complete")

    def finish(self, *, error_code: str | None = None, cancelled: bool = False) -> HttpProfile:
        now = perf_counter()
        status = "cancelled" if cancelled else "failed" if error_code else "complete"
        for name, start in self.pending.items():
            self.data.phases.append(HttpPhase(name=name, offset_seconds=start - self.started,
                                              duration_seconds=now - start, status=status))
        self.pending.clear()
        self.data.status = status
        self.data.error_code = error_code
        self.data.duration_seconds = now - self.started
        self.notify("request", status)
        return self.snapshot()


class EpisodeRecorder:
    def __init__(self, on_event=None):
        self.started = perf_counter()
        self.started_at = datetime.now(timezone.utc)
        self.episode_id = uuid4().hex
        self.on_event = on_event
        self.seen_phases: set[tuple[str, str]] = set()

    def emit(self, event: EpisodeEvent) -> None:
        if self.on_event is not None:
            self.on_event(event.model_copy(update={
                "episode_id": self.episode_id,
                "elapsed_seconds": perf_counter() - self.started,
                "timestamp": datetime.now(timezone.utc),
            }))

    def observe_trace(self, trace) -> None:
        self.episode_id = trace.id
        for name in ("boot", "setup", "agent", "finalize", "scoring"):
            span = getattr(trace.timing, name)
            for status, moment in (("started", span.start), ("complete", span.end)):
                key = (name, status)
                if moment and key not in self.seen_phases:
                    self.seen_phases.add(key)
                    self.emit(EpisodeEvent(kind="episode_phase", step=trace.num_turns,
                        details={"phase": name, "status": status,
                                 "duration_seconds": max(0, span.end - span.start) if status == "complete" else None}))

    def summary(self, trace, records, actions) -> EpisodeProfile:
        framework = {}
        if trace is not None:
            for name in ("boot", "setup", "agent", "finalize", "scoring"):
                span = getattr(trace.timing, name)
                if span.start:
                    framework[name] = max(0, (span.end or time()) - span.start)
        wall = perf_counter() - self.started
        model = sum(record["duration"] for record in records)
        tools = sum(action.duration_seconds for action in actions)
        setup = framework.get("boot", 0) + framework.get("setup", 0)
        finalize = framework.get("finalize", 0)
        scoring = framework.get("scoring", 0)
        return EpisodeProfile(wall_seconds=wall, setup_seconds=setup, model_seconds=model,
            tools_seconds=tools, finalize_seconds=finalize, scoring_seconds=scoring,
            other_seconds=max(0, wall - model - tools - setup - finalize - scoring), framework_seconds=framework)


def safe_event_record(kind: str, data: dict, *, run_id: str | None = None) -> dict:
    """Минимальный JSONL-журнал; полные вопросы и ответы остаются в episode.json."""
    record = {"schema_version": "1", "timestamp": datetime.now(timezone.utc).isoformat(), "event": kind}
    if run_id:
        record["run_id"] = run_id
    if kind == "progress":
        record['timestamp'] = data.get('timestamp') or record['timestamp']
        record.update({key: data.get(key) for key in ("kind", "step", "episode_id", "elapsed_seconds")})
        details = data.get("details", {})
        allowed = {"name", "tool_call_id", "duration_seconds", "phase", "status", "max_output_tokens", "http_profile", "profile",
                   'model', 'adapter', 'response_id', 'framework_version', 'config', 'corpus_id', 'corpus_revision'}
        record["details"] = {key: value for key, value in details.items() if key in allowed}
        if "usage" in details:
            record["details"]["usage"] = {key: details["usage"].get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")}
            for section, name, output in (('prompt_tokens_details', 'cached_tokens', 'cached_input_tokens'),
                                          ('completion_tokens_details', 'reasoning_tokens', 'reasoning_tokens')):
                metadata = details['usage'].get(section)
                value = metadata.get(name) if isinstance(metadata, dict) else None
                record['details']['usage'][output] = value if isinstance(value, int) and value >= 0 else None
        if "result" in details:
            result = details["result"]
            record["details"]["ok"] = result.get("ok")
            record["details"]["error_code"] = (result.get("error") or {}).get("code")
    elif kind == "complete":
        result = data["result"]
        record.update(episode_id=result["episode_id"], stop_reason=result["stop_reason"], profile=result.get("profile"))
    elif kind in {"started", "persisted"}:
        record.update({key: data[key] for key in ('queue_seconds', 'duration_seconds', 'scenario', 'mode') if key in data})
    return record


class EventLog:
    def __init__(self, path: str | Path, *, run_id: str | None = None):
        self.path = Path(path)
        self.run_id = run_id
        self.error: str | None = None

    def write(self, kind: str, data: dict, *, seq: int | None = None):
        record = safe_event_record(kind, data, run_id=self.run_id)
        if seq is not None:
            record['seq'] = seq
        line = json.dumps(record, ensure_ascii=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'a', encoding='utf-8') as stream:
                stream.write(line + '\n')
        except OSError as exc:
            self.error = type(exc).__name__
        event_logger.info(line)


def format_profile(result) -> str:
    profile = result.get("profile") or {}
    native = (result.get('native_trace') or {}).get('timing') or {}
    if not profile and all(native.get(name, {}).get('start') and native[name].get('end')
                           for name in ('boot', 'setup', 'finalize', 'scoring')):
        duration = lambda name: max(0, native[name]['end'] - native[name]['start'])
        profile = {'setup_seconds': duration('boot') + duration('setup'),
                   'model_seconds': sum(step['model_duration_seconds'] for step in result['trajectory']),
                   'tools_seconds': sum(action['duration_seconds'] for step in result['trajectory'] for action in step['actions']),
                   'finalize_seconds': duration('finalize'), 'scoring_seconds': duration('scoring')}
        profile['other_seconds'] = max(0, result['duration_seconds'] - sum(profile.values()))
    lines = [f"Эпизод {result['episode_id']} · {result['stop_reason']} · всего {result['duration_seconds']:.3f} с"]
    if profile:
        for key, label in (("setup_seconds", "Запуск среды"), ("model_seconds", "Запросы модели"),
                           ("tools_seconds", "Инструменты"), ("finalize_seconds", "Финализация"),
                           ("scoring_seconds", "Оценка"), ("other_seconds", "Обвязка и завершение")):
            lines.append(f"  {label}: {profile[key]:.3f} с")
    for step in result["trajectory"]:
        http = step.get("http_profile")
        lines.append(f"Шаг {step['step']}: модель {step['model_duration_seconds']:.3f} с")
        if http:
            lines.append(f"  HTTP {http['status_code']} · request_id={http['request_id']} · reused={http['connection_reused']}")
            for key, label in (("time_to_headers_seconds", "До заголовков"),
                               ("time_to_first_body_byte_seconds", "До первого байта тела (не TTFT)"),
                               ("body_read_seconds", "Чтение тела"), ("decode_seconds", "Разбор JSON")):
                value = http.get(key)
                lines.append(f"  {label}: {value:.3f} с" if value is not None else f"  {label}: нет данных")
            for phase in http["phases"]:
                lines.append(f"    {phase['name']}: {phase['duration_seconds']:.3f} с [{phase['status']}]")
        else:
            lines.append("  Детальных HTTP-замеров нет.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Разобрать профиль сохранённого эпизода без повторного инференса.")
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    print(format_profile(json.loads(args.episode.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()

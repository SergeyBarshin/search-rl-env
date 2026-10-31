"""Async-цикл, независимый от провайдера модели и реализации поиска."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from search_rl_env.contracts.types import (
    AgentTask,
    ChatModel,
    EpisodeConfig,
    EpisodeError,
    EpisodeResult,
    EpisodeStep,
    ModelError,
    SearchBackend,
    StopReason,
    ToolExecution,
    Usage,
)

from .prompts import budget_message, initial_history
from .tools import ToolCall, ToolState, execute_tool, invalid_action, tool_schemas


async def run_episode(
    task: AgentTask,
    model: ChatModel,
    search: SearchBackend,
    config: EpisodeConfig,
) -> EpisodeResult:
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    state = ToolState()
    history = initial_history(task)
    trajectory: list[EpisodeStep] = []
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0)
    call_ids: set[str] = set()

    while state.stop_reason is None:
        if len(trajectory) >= config.max_steps:
            state.stop_reason = StopReason.MAX_STEPS
            break
        # Отсутствующий usage завершает эпизод ниже, поэтому здесь счётчик известен.
        remaining_tokens = config.max_output_tokens - (usage.completion_tokens or 0)
        if remaining_tokens <= 0:
            state.stop_reason = StopReason.MAX_OUTPUT_TOKENS
            break
        history.append(budget_message(
            config,
            steps=len(trajectory),
            searches=state.searches,
            output_tokens=usage.completion_tokens or 0,
        ))
        step = EpisodeStep(
            step=len(trajectory) + 1,
            request_max_output_tokens=min(config.max_tokens_per_step, remaining_tokens),
        )
        trajectory.append(step)
        model_started = perf_counter()
        try:
            response = await model.complete(
                deepcopy(history),
                tools=tool_schemas(),
                max_output_tokens=step.request_max_output_tokens,
                seed=config.seed,
            )
        except Exception as exc:
            state.stop_reason = StopReason.INFRASTRUCTURE_ERROR
            message = str(exc) if isinstance(exc, ModelError) else f"Ошибка модели: {type(exc).__name__}."
            step.error = EpisodeError(code="model_error", message=message)
            state.error = step.error
            usage = usage.add(Usage())
            break
        finally:
            step.model_duration_seconds = perf_counter() - model_started

        step.usage = response.usage
        step.response_id = response.response_id
        step.model = response.model
        step.provider = response.provider
        step.finish_reason = response.finish_reason
        usage = usage.add(response.usage)
        history.append(deepcopy(response.message))

        if response.usage.completion_tokens is None:
            state.stop_reason = StopReason.INFRASTRUCTURE_ERROR
            state.error = EpisodeError(code="missing_usage", message="Нет completion_tokens; бюджет неизвестен.")
        elif response.finish_reason == "length" or response.usage.completion_tokens > step.request_max_output_tokens:
            state.stop_reason = StopReason.MAX_OUTPUT_TOKENS
        elif response.finish_reason == "error":
            state.stop_reason = StopReason.INFRASTRUCTURE_ERROR
            state.error = EpisodeError(code="provider_error", message="Провайдер вернул finish_reason=error.")

        try:
            if response.message.get("role") != "assistant":
                raise ValueError("Expected assistant message")
            calls = TypeAdapter(list[ToolCall]).validate_python(
                response.message.get("tool_calls") or [], strict=True
            )
            ids = [call.id for call in calls]
            if len(ids) != len(set(ids)) or call_ids.intersection(ids):
                raise ValueError("Repeated tool_call_id")
        except (ValidationError, ValueError):
            if state.stop_reason is None:
                state.stop_reason = StopReason.INVALID_ACTION
                state.error = EpisodeError(code="invalid_tool_calls", message="Нарушен формат tool_calls или id вызова.")
                state.invalid_actions += 1
            step.error = state.error
            break

        call_ids.update(ids)
        if not calls and state.stop_reason is None:
            error = invalid_action(state, config, "missing_tool_call", "Для действия нужны search или finish.")
            step.error = EpisodeError(**error["error"])
            history.append({"role": "user", "content": json.dumps(error, ensure_ascii=False)})

        for call in calls:
            action_started = perf_counter()
            result = await execute_tool(call, task=task, search=search, config=config, state=state)
            step.actions.append(ToolExecution(
                tool_call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
                result=result,
                duration_seconds=perf_counter() - action_started,
            ))
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        if state.error is not None:
            step.error = state.error

    return EpisodeResult(
        task_id=task.task_id,
        episode_id=str(uuid4()),
        task=task,
        config=config,
        model_metadata=deepcopy(model.metadata),
        selected_document_ids=state.selected_ids,
        stop_reason=state.stop_reason,
        truncated=state.stop_reason in {
            StopReason.MAX_STEPS, StopReason.MAX_SEARCHES, StopReason.MAX_OUTPUT_TOKENS,
        },
        history=history,
        trajectory=trajectory,
        usage=usage,
        search_calls=state.searches,
        invalid_actions=state.invalid_actions,
        started_at=started_at,
        duration_seconds=perf_counter() - started,
        error=state.error,
    )


def save_episode(result: EpisodeResult, path: str | Path) -> Path:
    """Атомарно записать один результат; runner сам выбирает путь и возобновление."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = stream.name
            stream.write(result.model_dump_json(indent=2) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return path

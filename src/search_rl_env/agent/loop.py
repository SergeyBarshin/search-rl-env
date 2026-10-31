"""Тонкий запуск Verifiers v1 и проекция нативной Trace в EpisodeResult."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import time

import verifiers.v1 as vf
from verifiers.v1.configs.agent import AgentConfig, TimeoutConfig

from search_rl_env.contracts.types import (
    AgentTask, ChatModel, EpisodeConfig, EpisodeError, EpisodeEvent, EpisodeResult,
    EpisodeStep, SearchBackend, StopReason, ToolExecution, Usage,
)
from search_rl_env.environment.state import SearchState
from search_rl_env.environment.taskset import SearchTask, SearchTaskConfig, SearchTaskData
from search_rl_env.environment.tools import SearchToolsetConfig

from .bridge import BackendBridge
from .harness import SearchHarnessConfig
from .prompts import SYSTEM_PROMPT


def episode_from_trace(task, model, config, trace, bridge, *, interrupted=False):
    state = trace.state.model_copy(deep=True) if trace is not None else SearchState()
    error = state.error or bridge.error
    reason = state.stop_reason
    if interrupted:
        reason = StopReason.INFRASTRUCTURE_ERROR
        error = EpisodeError(code="interrupted", message="Эпизод отменён во время выполнения.")
        state.stop_reason = reason
        state.error = error
        recorded = {action.tool_call_id for action in state.actions}
        for call in state.pending_calls:
            if call['id'] not in recorded:
                state.actions.append(ToolExecution(
                    tool_call_id=call['id'], name=call['function']['name'],
                    arguments=call['function']['arguments'],
                    result={'ok': False, 'error': error.model_dump()}, duration_seconds=0,
                ))
    elif bridge.error:
        reason = StopReason.INFRASTRUCTURE_ERROR
    elif trace is not None and not trace.ok:
        reason = StopReason.INFRASTRUCTURE_ERROR
    elif reason is None:
        reason = {"max_turns": StopReason.MAX_STEPS,
                  "max_output_tokens": StopReason.MAX_OUTPUT_TOKENS}.get(
                      trace.stop_condition if trace else None, StopReason.INFRASTRUCTURE_ERROR)
    if reason == StopReason.INFRASTRUCTURE_ERROR and error is None:
        error = EpisodeError(code="framework_error", message="Ошибка выполнения среды Verifiers.")
    actions = {action.tool_call_id: action for action in state.actions}
    history = []
    trajectory = []
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0)
    for index, record in enumerate(bridge.records, 1):
        response = record["response"]
        current_usage = response.usage if response else Usage()
        usage = usage.add(current_usage)
        step = EpisodeStep(step=index, request_max_output_tokens=record["max_tokens"],
            model_duration_seconds=record["duration"], usage=current_usage,
            model=response.model if response else None,
            provider=response.provider if response else None,
            response_id=response.response_id if response else None,
            finish_reason=response.finish_reason if response else None)
        history = deepcopy(record["messages"])
        if response:
            history.append(deepcopy(response.message))
            for call in response.message.get("tool_calls") or []:
                if call.get("id") in actions:
                    step.actions.append(actions[call["id"]])
        trajectory.append(step)
    observed = {message.get("tool_call_id") for message in history if message.get("role") == "tool"}
    for action in state.actions:
        if action.tool_call_id not in observed:
            history.append({"role": "tool", "tool_call_id": action.tool_call_id,
                            "content": json.dumps(action.result, ensure_ascii=False)})
    if trajectory and error:
        trajectory[-1].error = error
    native = trace.to_record() if trace is not None else None
    if native is not None and interrupted:
        native["info"]["interrupted"] = True
        native["info"]["episode"] = state.model_dump(mode='json', exclude={'artifacts'})
    has_reference = bool(trace and trace.info.get("has_reference"))
    return EpisodeResult(
        task_id=task.task_id, episode_id=trace.id if trace else bridge.token, task=task, config=config,
        model_metadata={**deepcopy(model.metadata), "framework": "verifiers", "framework_version": "0.4.0"},
        selected_document_ids=list(state.selected_ids) if reason == StopReason.FINISHED else [],
        stop_reason=reason,
        truncated=reason in {StopReason.MAX_STEPS, StopReason.MAX_OUTPUT_TOKENS},
        history=history, trajectory=trajectory, usage=usage, search_calls=max(state.searches, bridge.search_calls),
        invalid_actions=state.invalid_actions,
        started_at=datetime.fromtimestamp(trace.timing.start, timezone.utc) if trace else datetime.now(timezone.utc),
        duration_seconds=max(0, time() - trace.timing.start) if trace else 0,
        error=error, reward=trace.reward if trace and trace.ok and has_reference else None,
        native_trace=native,
    )


async def run_episode(
    task: AgentTask,
    model: ChatModel,
    search: SearchBackend,
    config: EpisodeConfig,
    *,
    on_event: Callable[[EpisodeEvent], None] | None = None,
    on_interrupted: Callable[[EpisodeResult], None] | None = None,
    reference_document_ids: list[str] | None = None,
) -> EpisodeResult:
    """Запустить vf.Agent с инструментами vf.Toolset и получить нативную траекторию."""
    async with BackendBridge(task, model, search, config, on_event) as bridge:
        search_task = SearchTask(
            SearchTaskData(id=task.task_id, agent_task=task, prompt=task.model_dump_json(),
                system_prompt=SYSTEM_PROMPT,
                reference_document_ids=reference_document_ids),
            SearchTaskConfig(tools=SearchToolsetConfig(
                backend_url=bridge.url, search_limit=config.search_limit,
                max_snippet_chars=config.max_snippet_chars,
            )),
            runtime_env={"SEARCH_RL_BRIDGE_TOKEN": bridge.token},
            on_event=on_event,
        )
        agent = vf.Agent(AgentConfig(
            model=model.metadata.get("model") or "model",
            harness=SearchHarnessConfig(),
            runtime=vf.SubprocessConfig(),
            client=vf.EvalClientConfig(base_url=bridge.url + "/v1", api_key_var=bridge.key_var),
            sampling=vf.SamplingConfig(temperature=0.0, max_tokens=config.max_tokens_per_step),
            max_turns=config.max_steps, max_output_tokens=config.max_output_tokens,
            timeout=TimeoutConfig(setup=60, finalize=10, scoring=10),
            network_notice=False,
        ))
        active = None

        def observe(trace):
            nonlocal active
            active = trace

        try:
            trace = await agent.run(search_task, on_trace=observe)
        except asyncio.CancelledError:
            if on_interrupted:
                on_interrupted(episode_from_trace(task, model, config, active, bridge, interrupted=True))
            raise
        return episode_from_trace(task, model, config, trace, bridge)

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

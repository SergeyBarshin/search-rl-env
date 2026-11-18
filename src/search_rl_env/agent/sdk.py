"""Адаптер OpenAI Agents SDK для публичного протокола SQL, search и finish."""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from time import perf_counter, time

import httpx
from agents import (
    Agent, FunctionTool, MaxTurnsExceeded, OpenAIChatCompletionsModel, RunConfig, Runner, ToolsToFinalOutputResult,
)
from pydantic import ValidationError

from search_rl_env.contracts.types import (
    EpisodeError, EpisodeEvent, EpisodeResult, EpisodeStep, ModelError, StopReason, ToolExecution, Usage,
)
from search_rl_env.environment.rewards import score_episode
from search_rl_env.environment.state import SearchState
from search_rl_env.profiling import EpisodeRecorder, observe_http

from .prompts import HYBRID_SYSTEM_PROMPT, SYSTEM_PROMPT
from .rounds import BOUNDED_RETRIEVAL_PROMPT, ROUND_PROMPT, EvidenceRanking, RetrievalRounds
from .tools import ToolCall, execute_tool, invalid_action, tool_error, tool_schemas

SDK_VERSION = version("openai-agents")
BATCH_TOOL = "execute_episode_actions"
FINISH_REMINDER = (
    "No final answer has been submitted. Use the finish tool to submit "
    "your supported answer and retrieved document_ids (an empty list is allowed). "
    "State missing evidence explicitly. If another tool call is needed, "
    "you may make it before finish. Do not invent tool results."
)


class SdkEpisode:
    def __init__(self, task, model, search, config, sql, on_event):
        self.task, self.model, self.search, self.config, self.sql = task, model, search, config, sql
        self.recorder = EpisodeRecorder(on_event)
        self.state = SearchState()
        self.rounds = RetrievalRounds(config.retrieval_rounds) if config.retrieval_rounds else None
        self.ranking = EvidenceRanking() if config.retrieval_rounds and config.retrieval_rounds.preserve_ranking else None
        default_prompt = HYBRID_SYSTEM_PROMPT if sql is not None else ROUND_PROMPT if self.rounds else SYSTEM_PROMPT
        if self.ranking is not None:
            default_prompt = BOUNDED_RETRIEVAL_PROMPT
        self.prompt = config.system_prompt or default_prompt
        self.prompt_revision = sha256(self.prompt.encode()).hexdigest()
        public_task = task.model_dump()
        if sql is not None:
            public_task["sql_schema"] = sql.schema
        if self.ranking is not None:
            public_task["search_limit"] = config.search_limit
        self.user_prompt = json.dumps(public_task, ensure_ascii=False)
        self.history = [{"role": "system", "content": self.prompt}, {"role": "user", "content": self.user_prompt}]
        self.context = list(self.history)
        self.schemas = tool_schemas(sql is not None, retrieval_rounds=self.rounds is not None)
        self.tool_names = {schema["function"]["name"] for schema in self.schemas}
        self.offered = self.tool_names
        self.trajectory: list[EpisodeStep] = []
        self.model_turns = 0
        self.pending: list[ToolCall] = []
        self.action_started = None
        self.tool_task = None
        self.finish_retries = 0
        self.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0)

    def emit(self, kind, **details):
        self.recorder.emit(EpisodeEvent(kind=kind, step=self.model_turns, details=deepcopy(details)))

    def stop(self, reason, code=None, message=None):
        self.state.stop_reason = reason
        if code is not None:
            self.state.error = EpisodeError(code=code, message=message)

    def check_output_budget(self):
        cap = self.config.max_output_tokens
        if self.state.stop_reason is None and cap is not None and self.usage.completion_tokens >= cap:
            self.stop(StopReason.MAX_OUTPUT_TOKENS)

    def remember(self, message):
        self.history.append(message)
        self.context.append(message)

    async def complete(self):
        self.model_turns += 1
        step = EpisodeStep(step=self.model_turns, request_max_output_tokens=self.config.max_tokens_per_step)
        schemas = self.schemas
        final_turn = step.step == self.config.max_steps
        if self.rounds is not None:
            self.rounds.turns += 1
            step.retrieval_round = self.rounds.number
            self.offered = self.rounds.tools_for_turn(self.context, final_turn=final_turn)
            if self.ranking is not None and self.offered == {"finish"}:
                self.prepare_finalization()
            schemas = [schema for schema in self.schemas if schema["function"]["name"] in self.offered]
        elif self.config.finish_on_last_step and final_turn:
            self.remember({"role": "user", "content": (
                "This is the final turn. Call finish with the best answer supported by the observations. "
                "If some evidence is missing, state that explicitly in the answer."
            )})
            schemas = [schema for schema in self.schemas if schema["function"]["name"] == "finish"]
        self.trajectory.append(step)
        self.emit("model_started", max_output_tokens=step.request_max_output_tokens)
        started = perf_counter()

        def progress(profile, phase, status):
            step.http_profile = profile
            self.emit("model_progress", phase=phase, status=status, http_profile=profile.model_dump(mode="json"))

        try:
            # Каноническая wire-история включает поля провайдера, которые SDK
            # может не представить в унифицированных Responses items.
            with observe_http(progress):
                response = await self.model.complete(
                    deepcopy(self.context), tools=deepcopy(schemas),
                    max_output_tokens=step.request_max_output_tokens, seed=self.config.seed,
                )
        except asyncio.CancelledError as error:
            step.http_profile = getattr(error, "http_profile", step.http_profile)
            self.usage = self.usage.add(step.usage)
            raise
        except Exception as error:
            step.http_profile = getattr(error, "http_profile", step.http_profile)
            self.usage = self.usage.add(step.usage)
            code, message = (error.code, str(error)) if isinstance(error, ModelError) else (
                "model_error", f"Ошибка модели: {type(error).__name__}.")
            self.stop(StopReason.INFRASTRUCTURE_ERROR, code, message)
            return self.sdk_response()
        finally:
            step.model_duration_seconds = perf_counter() - started

        step.usage = response.usage
        step.http_profile = response.http_profile or step.http_profile
        step.model, step.provider = response.model, response.provider
        step.response_id, step.finish_reason = response.response_id, response.finish_reason
        self.usage = self.usage.add(step.usage)
        self.remember(deepcopy(response.message))
        self.emit("model_finished", model=step.model, usage=step.usage.model_dump(),
                  duration_seconds=step.model_duration_seconds, response_id=step.response_id,
                  http_profile=step.http_profile.model_dump(mode="json") if step.http_profile else None)

        if step.finish_reason == "length":
            self.stop(StopReason.MAX_OUTPUT_TOKENS)
        elif step.finish_reason == "error":
            self.stop(StopReason.INFRASTRUCTURE_ERROR, "provider_error", "Провайдер вернул ошибку генерации.")
        elif step.usage.prompt_tokens is None or step.usage.completion_tokens is None:
            self.stop(StopReason.INFRASTRUCTURE_ERROR, "missing_usage", "Нет токенной статистики ответа модели.")
        else:
            try:
                self.pending = [ToolCall.model_validate(call) for call in response.message.get("tool_calls") or []]
            except (TypeError, ValidationError):
                self.stop(StopReason.INFRASTRUCTURE_ERROR, "model_error", "Неверный формат вызовов инструментов.")
        if self.pending:
            self.finish_retries = 0
        return self.sdk_response()

    def sdk_response(self):
        step = self.trajectory[-1]
        message = {"role": "assistant", "content": "Turn completed."}
        finish_reason = "stop"
        if self.pending:
            # SDK может выполнять инструменты одного ответа конкурентно.
            # Для нескольких действий SDK вызывает один диспетчер, который
            # сохраняет порядок, исходные id и наблюдение для каждого действия.
            first = self.pending[0]
            name = first.function.name if len(self.pending) == 1 and first.function.name in self.tool_names else BATCH_TOOL
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{step.step}", "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }]}
            finish_reason = "tool_calls"
        prompt_tokens = step.usage.prompt_tokens or 0
        completion_tokens = step.usage.completion_tokens or 0
        return {
            "id": f"chatcmpl-{self.recorder.episode_id}-{step.step}", "object": "chat.completion",
            "created": int(time()), "model": step.model or "model",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }

    def record_action(self, call, result, duration, *, model_tool=True):
        action = ToolExecution(tool_call_id=call.id, name=call.function.name, arguments=call.function.arguments,
                               result=result, duration_seconds=duration)
        self.trajectory[-1].actions.append(action)
        if model_tool:
            self.remember({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
        self.emit("tool_finished", **action.model_dump())
        return result

    async def execute(self, call):
        visible_ids = None
        if self.rounds is not None and self.state.stop_reason is None:
            name = call.function.name
            if self.rounds.checkpoint is not None:
                return tool_error("round_advanced", "The next round starts after this tool batch.")
            if name not in self.offered:
                return invalid_action(self.state, "unavailable_tool", "This tool is unavailable on the current turn.")
            if name == "search" and not self.rounds.can_search(self.context):
                return invalid_action(self.state, "search_locked", "Save this round or submit the final ranking.")
            if name == "save_and_advance":
                return self.rounds.save(call.function.arguments, self.state)
            visible_ids = set(self.rounds.visible)
        result = await execute_tool(call, task=self.task, search=self.search, config=self.config,
                                    state=self.state, sql=self.sql, visible_ids=visible_ids,
                                    on_search_hits=self.ranking.remember if self.ranking is not None else None)
        if self.rounds is not None and call.function.name == "search" and result["ok"]:
            self.rounds.remember_hits(result)
        return result

    async def invoke_tools(self, _context, _arguments):
        self.tool_task = asyncio.current_task()
        try:
            results = []
            for call in self.pending:
                self.action_started = perf_counter()
                self.emit("tool_started", tool_call_id=call.id, name=call.function.name, arguments=call.function.arguments)
                result = await self.execute(call)
                results.append(self.record_action(call, result, perf_counter() - self.action_started))
                self.action_started = None
            self.pending.clear()
            return json.dumps(results, ensure_ascii=False)
        finally:
            self.tool_task = None

    def after_tools(self, _context, _results):
        self.check_output_budget()
        ended = self.state.stop_reason is not None or (self.rounds is not None and self.rounds.checkpoint is not None)
        return ToolsToFinalOutputResult(is_final_output=ended, final_output="Turn completed." if ended else None)

    async def run(self, agent, run_config):
        if self.ranking is not None:
            await self.bootstrap_search()
        while self.state.stop_reason is None:
            remaining = None if self.config.max_steps is None else self.config.max_steps - self.model_turns
            if self.rounds is not None:
                round_remaining = self.rounds.config.max_turns - self.rounds.turns
                remaining = round_remaining if remaining is None else min(round_remaining, remaining)
            if remaining == 0:
                self.stop(StopReason.MAX_STEPS)
                break
            try:
                await Runner.run(agent, self.user_prompt, max_turns=remaining, run_config=run_config)
            except MaxTurnsExceeded:
                self.stop(StopReason.MAX_STEPS)
            self.check_output_budget()
            if self.state.stop_reason is not None:
                break
            if self.rounds is not None and self.rounds.checkpoint is not None:
                self.context = self.history[:2]
                self.remember(self.rounds.advance())
                self.finish_retries = 0
                self.emit("retrieval_round_started", retrieval_round=self.rounds.number)
                continue
            if self.finish_retries == self.config.max_finish_retries:
                self.stop(StopReason.INVALID_ACTION, "missing_finish", "Модель завершила ответ без finish.")
                break
            self.finish_retries += 1
            self.remember({"role": "user", "content": FINISH_REMINDER})

    async def bootstrap_search(self):
        self.trajectory.append(EpisodeStep(step=0, retrieval_round=1, request_max_output_tokens=None))
        call = ToolCall.model_validate({
            "id": "initial_search", "type": "function",
            "function": {"name": "search", "arguments": json.dumps({"query": self.task.question}, ensure_ascii=False)},
        })
        started = perf_counter()
        self.emit("tool_started", tool_call_id=call.id, name="search", arguments=call.function.arguments)
        result = await self.execute(call)
        self.record_action(call, result, perf_counter() - started, model_tool=False)
        if result["ok"]:
            self.remember({"role": "user", "content": json.dumps({"initial_search": result["hits"]}, ensure_ascii=False)})

    def prepare_finalization(self):
        ids = self.ranking.top_ids(limit=20)
        snippet_limit = self.rounds.config.final_snippet_chars
        if self.config.max_snippet_chars is not None:
            snippet_limit = min(self.config.max_snippet_chars, snippet_limit or self.config.max_snippet_chars)
        self.rounds.visible = {identifier: self.ranking.snippets[identifier][:snippet_limit] for identifier in ids}
        self.context = self.history[:2]
        self.remember({"role": "user", "content": json.dumps({
            "instruction": "Call finish now. Rank up to ten of the supplied evidence IDs by support for the question.",
            "evidence": [{"doc_id": identifier, "snippet": snippet} for identifier, snippet in self.rounds.visible.items()],
        }, ensure_ascii=False)})

    def incomplete_actions(self):
        if not self.pending:
            return
        completed = len(self.trajectory[-1].actions)
        for index, call in enumerate(self.pending[completed:]):
            elapsed = perf_counter() - self.action_started if index == 0 and self.action_started is not None else 0
            self.record_action(call, {"ok": False, "error": self.state.error.model_dump()}, elapsed)
        self.pending.clear()

    def result(self, profile):
        reason = self.state.stop_reason
        if self.trajectory:
            self.trajectory[-1].error = self.state.error
        finished = reason == StopReason.FINISHED
        selected = self.state.selected_ids if finished else []
        ranking_source = "model" if finished and self.ranking is not None else None
        if self.ranking is not None and reason != StopReason.INFRASTRUCTURE_ERROR:
            ranking = list(dict.fromkeys([*selected, *self.ranking.top_ids()]))[:10]
            if ranking != selected:
                ranking_source = "model_plus_retrieval" if selected else "retrieval_fallback"
            selected = ranking
        return EpisodeResult(
            schema_version="4", task_id=self.task.task_id, episode_id=self.recorder.episode_id, task=self.task,
            config=self.config, model_metadata={**deepcopy(self.model.metadata), "framework": "openai-agents",
                                                "framework_version": SDK_VERSION},
            selected_document_ids=selected, ranking_source=ranking_source, stop_reason=reason,
            truncated=reason in {StopReason.MAX_STEPS, StopReason.MAX_OUTPUT_TOKENS},
            history=self.history, trajectory=self.trajectory, usage=self.usage, search_calls=self.state.searches,
            invalid_actions=self.state.invalid_actions, started_at=self.recorder.started_at,
            duration_seconds=profile.wall_seconds, error=self.state.error, profile=profile,
            answer=self.state.answer if finished else None, sql_calls=self.state.sql_calls,
            sql_enabled=self.sql is not None, prompt_revision=self.prompt_revision,
        )


_active_episode: ContextVar[SdkEpisode] = ContextVar("active_sdk_episode")


async def dispatch_completion(_request):
    episode = _active_episode.get()
    return httpx.Response(200, json=await episode.complete())


async def run_sdk_episode(task, model, search, config, *, client, sql=None, on_event=None, on_interrupted=None,
                          reference_document_ids=None, reference_answer=None):
    if sql is not None and config.retrieval_rounds is not None:
        raise ValueError("Раунды retrieval предназначены для текстового поиска.")
    if sql is None and reference_document_ids == []:
        raise ValueError("Для Recall нужно непустое множество эталонных документов.")
    episode = SdkEpisode(task, model, search, config, sql, on_event)
    episode.emit("episode_started", model=model.metadata.get("model"), adapter=model.metadata.get("adapter"),
                 framework_version=SDK_VERSION, config=config.model_dump(mode="json"),
                 corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)
    episode.emit("episode_phase", phase="setup", status="started")
    tools = [FunctionTool(name=spec["name"], description=spec["description"], params_json_schema=spec["parameters"],
                          on_invoke_tool=episode.invoke_tools, strict_json_schema=False)
             for spec in (schema["function"] for schema in episode.schemas)]
    tools.append(FunctionTool(name=BATCH_TOOL, description="Dispatch an ordered tool batch.",
                              params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
                              on_invoke_tool=episode.invoke_tools))
    agent = Agent(name="search_rl_env", instructions=episode.prompt, tools=tools, tool_use_behavior=episode.after_tools,
                  model=OpenAIChatCompletionsModel(model=model.metadata.get("model") or "model", openai_client=client))
    run_config = RunConfig(tracing_disabled=True, trace_include_sensitive_data=False)
    setup = perf_counter() - episode.recorder.started
    episode.emit("episode_phase", phase="setup", status="complete", duration_seconds=setup)
    token = _active_episode.set(episode)
    cancelled = False
    try:
        async with asyncio.timeout(config.rollout_timeout_seconds):
            await episode.run(agent, run_config)
    except asyncio.CancelledError:
        cancelled = True
        episode.stop(StopReason.INFRASTRUCTURE_ERROR, "interrupted", "Эпизод отменён во время выполнения.")
        raise
    except TimeoutError:
        episode.stop(StopReason.INFRASTRUCTURE_ERROR, "rollout_timeout", "Истекло время выполнения эпизода.")
    except Exception as error:
        episode.stop(StopReason.INFRASTRUCTURE_ERROR, "framework_error", f"Ошибка Agents SDK: {type(error).__name__}.")
    finally:
        _active_episode.reset(token)
        if episode.tool_task is not None:
            # SDK может отменить дочернюю задачу до завершения очистки бэкенда.
            tool_task = episode.tool_task
            if not tool_task.cancelling():
                tool_task.cancel()
            await asyncio.gather(tool_task, return_exceptions=True)
        if episode.state.stop_reason is not None:
            episode.incomplete_actions()
            profile = episode.recorder.summary(episode.trajectory, setup_seconds=setup)
            result = episode.result(profile)
            episode.emit("episode_finished", profile=profile.model_dump(mode="json"),
                         status="cancelled" if cancelled else result.stop_reason.value)
            if cancelled and on_interrupted is not None:
                on_interrupted(result)
    result.reward = score_episode(result, reference_document_ids=reference_document_ids, reference_answer=reference_answer)
    return result

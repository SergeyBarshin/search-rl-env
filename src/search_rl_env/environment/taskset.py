"""Задачи Verifiers v1: приватные эталоны, инструменты и терминальная оценка."""

import json
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field, field_validator

from search_rl_env.agent.prompts import SYSTEM_PROMPT
from search_rl_env.contracts.types import AgentTask, EpisodeError, EpisodeEvent, SearchHit, StopReason, ToolExecution
from .rewards import recall_at_k
from .state import SearchState
from .tools import SearchToolset, SearchToolsetConfig


class SearchTaskData(vf.TaskData):
    agent_task: AgentTask
    reference_document_ids: list[str] | None = None

    @field_validator('reference_document_ids')
    @classmethod
    def nonempty_reference(cls, value):
        if value is not None and not value:
            raise ValueError('Эталонное множество не может быть пустым.')
        return value


class SearchTaskConfig(vf.TaskConfig):
    tools: SearchToolsetConfig = Field(default_factory=SearchToolsetConfig)


class SearchTask(vf.Task[SearchTaskData, SearchState, SearchTaskConfig]):
    def __init__(self, data, config=None, *, runtime_env=None, on_event=None):
        super().__init__(data, config)
        self._runtime_env = runtime_env or {}
        self._on_event = on_event
        self._emitted = {}

    def runtime_env(self):
        return self._runtime_env

    @classmethod
    def toolsets(cls, config):
        return [SearchToolset(config.tools)]

    def hooks(self, attr):
        if attr == 'reward' and self.data.reference_document_ids is None:
            return []
        return super().hooks(attr)

    @vf.stop(priority=100)
    async def task_stopped(self, trace: vf.Trace) -> bool:
        self._completed_actions(trace)
        if trace.state.stop_reason == StopReason.INFRASTRUCTURE_ERROR:
            raise vf.TaskError('Сбой поискового инструмента.')
        return trace.state.stop_reason is not None

    @vf.stop(priority=100)
    async def observe_response(self, response: vf.Response, trace: vf.Trace) -> bool:
        state = trace.state
        if response.finish_reason == 'length':
            state.stop_reason = StopReason.MAX_OUTPUT_TOKENS
            return True
        calls = response.message.tool_calls or []
        state.pending_calls = [{'id': call.id, 'type': 'function', 'function': {
            'name': call.name, 'arguments': call.arguments}} for call in calls]
        for call in calls:
            if self._on_event:
                self._on_event(EpisodeEvent(kind='tool_started', step=trace.num_turns + 1,
                    details={'name': call.name, 'arguments': call.arguments}))
        return False

    def _completed_actions(self, trace):
        # MCP возвращает ошибки схемы/неизвестного инструмента до вызова @vf.tool.
        # Их тоже сохраняем, но не добавляем собственной политики остановки.
        calls = {call.id: call for node in trace.nodes
                 for call in getattr(node.message, 'tool_calls', None) or []}
        recorded = {action.tool_call_id for action in trace.state.actions}
        for node in trace.nodes:
            message = node.message
            if message.role != 'tool' or message.tool_call_id in recorded:
                continue
            call = calls.get(message.tool_call_id)
            if call is None:
                continue
            trace.state.actions.append(ToolExecution(
                tool_call_id=call.id, name=call.name, arguments=call.arguments,
                result={'ok': False, 'error': {'code': 'tool_error', 'message': str(message.content)}},
                duration_seconds=0,
            ))
            trace.state.invalid_actions += 1
            recorded.add(call.id)
        emitted = self._emitted.setdefault(trace.id, set())
        if self._on_event:
            for action in trace.state.actions:
                if action.tool_call_id not in emitted:
                    self._on_event(EpisodeEvent(kind='tool_finished', step=trace.num_turns, details=action.model_dump()))
                    emitted.add(action.tool_call_id)

    async def finalize(self, trace, runtime):
        self._completed_actions(trace)
        state = trace.state
        if state.stop_reason is None:
            state.stop_reason = {
                'max_turns': StopReason.MAX_STEPS,
                'max_output_tokens': StopReason.MAX_OUTPUT_TOKENS,
            }.get(trace.stop_condition, StopReason.INFRASTRUCTURE_ERROR if trace.errors else StopReason.INVALID_ACTION)
        if state.stop_reason != StopReason.FINISHED:
            state.selected_ids = []
            if state.stop_reason == StopReason.INVALID_ACTION:
                state.error = EpisodeError(code='missing_finish', message='Модель завершила ответ без finish.')
        elif not trace.errors:
            # finish на последнем разрешённом ответе остаётся добровольным завершением.
            trace.stop('task_finished', override=True)
        trace.info['episode'] = state.model_dump(mode='json', exclude={'artifacts'})
        trace.info['has_reference'] = self.data.reference_document_ids is not None

    @vf.metric
    async def search_calls(self, trace: vf.Trace) -> float:
        return float(trace.state.searches)

    @vf.reward
    async def retrieval_recall(self, trace: vf.Trace) -> float:
        return recall_at_k(trace.state.selected_ids, self.data.reference_document_ids)


class SearchTasksetConfig(vf.TasksetConfig):
    tasks_path: Path | None = None
    hits_path: Path | None = None
    require_references: bool = False
    task: SearchTaskConfig = Field(default_factory=SearchTaskConfig)


class SearchTaskset(vf.Taskset[SearchTask, SearchTasksetConfig]):
    def load(self):
        if self.config.tasks_path:
            records = json.loads(self.config.tasks_path.read_text(encoding='utf-8'))
            if isinstance(records, dict):
                records = [records]
        else:
            records = [{'task_id': 'demo-001', 'question': 'Какие документы описывают перенос срока проекта?',
                        'language': 'ru', 'corpus_id': 'synthetic-demo', 'corpus_revision': 'v1',
                        'reference_document_ids': ['doc-001', 'doc-002']}]
        task_config = self.config.task.model_copy(deep=True)
        if self.config.hits_path:
            task_config.tools.hits = [SearchHit.model_validate(hit) for hit in json.loads(self.config.hits_path.read_text(encoding='utf-8'))]
        elif not task_config.tools.hits and task_config.tools.backend_url is None:
            task_config.tools.hits = [
                SearchHit(doc_id='doc-001', snippet='Срок проекта перенесён на 15 ноября.', rank=1),
                SearchHit(doc_id='doc-002', snippet='Обновлённый план проекта: завершение 15 ноября.', rank=2),
            ]
        for record in records:
            record = dict(record)
            reference = record.pop('reference_document_ids', None)
            if self.config.require_references and reference is None:
                raise ValueError('Для обучения нужны reference_document_ids.')
            agent_task = AgentTask.model_validate(record)
            yield SearchTask(SearchTaskData(
                id=agent_task.task_id, agent_task=agent_task, prompt=agent_task.model_dump_json(),
                system_prompt=SYSTEM_PROMPT, reference_document_ids=reference,
            ), task_config.model_copy(deep=True))

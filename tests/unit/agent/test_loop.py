import asyncio
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from search_rl_env.agent import run_episode, save_episode
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, EpisodeResult, SearchHit
from search_rl_env.contracts.types import ModelResponse, StopReason, Usage


@pytest.fixture
def task():
    return AgentTask(task_id="test-1", question="Когда перенесли срок?", language="ru", corpus_id="demo", corpus_revision="v1")


@pytest.fixture
def search(task):
    return MockSearch([
        SearchHit(doc_id="doc-1", snippet="Срок перенесён на ноябрь.", rank=1),
        SearchHit(doc_id="doc-2", snippet="Обновлённый план.", rank=2),
    ], corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)


def action(name, arguments, *, call_id="call-1", tokens=1):
    return ModelResponse(
        message={"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]},
        usage=Usage(prompt_tokens=5, completion_tokens=tokens, total_tokens=5 + tokens, cost=0.0),
        finish_reason="tool_calls", model="scripted-test-model",
    )


class ScriptedModel:
    metadata = {"adapter": "scripted"}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def complete(self, messages, **kwargs):
        self.requests.append({"messages": deepcopy(messages), **kwargs})
        return next(self.responses)


async def test_complete_episode_and_native_trace_roundtrip(task, search, tmp_path):
    result = await run_episode(task, MockModel(), search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert result.selected_document_ids == ["doc-1", "doc-2"]
    assert not result.truncated
    assert result.search_calls == 1
    assert result.usage.completion_tokens == 2
    assert [step.actions[0].name for step in result.trajectory] == ["search", "finish"]
    assert search.calls[0] == {"query": task.question, "corpus_id": "demo", "corpus_revision": "v1", "limit": 10}
    assert [m["tool_call_id"] for m in result.history if m["role"] == "tool"] == ["mock-call-1", "mock-call-2"]
    native = result.native_trace
    assert native["verifiers"]["version"] == "0.4.0"
    assert len(native["calls"]) == 2
    assert native["stop_condition"] == "task_finished"
    assert all(node["sampled"] == (node["message"]["role"] == "assistant") for node in native["nodes"])
    assert result.reward is None
    path = save_episode(result, tmp_path / "nested/episode.json")
    assert EpisodeResult.model_validate_json(path.read_text()) == result
    assert list(path.parent.iterdir()) == [path]


def test_task_rejects_reference_fields(task):
    with pytest.raises(ValidationError):
        AgentTask.model_validate({**task.model_dump(), "gold": {"relevant_doc_ids": ["doc-1"]}})


async def test_model_decides_searches_and_order_without_default_caps(task, search):
    model = ScriptedModel([
        *(action("search", {"query": f"уточнение {i}"}, call_id=f"s-{i}") for i in range(7)),
        action("finish", {"document_ids": ["doc-2", "doc-1", "doc-2"]}, call_id="finish"),
    ])
    result = await run_episode(task, model, search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert result.search_calls == 7
    assert result.selected_document_ids == ["doc-2", "doc-1"]
    assert result.config.max_steps is result.config.max_output_tokens is None
    assert "remaining_budget" not in json.dumps(model.requests, ensure_ascii=False)


async def test_step_limit_does_not_select_all_found_documents(task, search):
    result = await run_episode(task, MockModel(), search, EpisodeConfig(max_steps=1))
    assert result.stop_reason == StopReason.MAX_STEPS
    assert result.truncated
    assert result.selected_document_ids == []
    assert len(result.trajectory) == 1


async def test_native_output_cap_is_checked_between_turns(task, search):
    model = ScriptedModel([
        action("search", {"query": "срок"}, tokens=2),
        action("finish", {"document_ids": ["doc-1"]}, call_id="call-2", tokens=2),
    ])
    result = await run_episode(task, model, search, EpisodeConfig(max_output_tokens=3, max_tokens_per_step=10))
    assert [request["max_output_tokens"] for request in model.requests] == [10, 10]
    assert result.stop_reason == StopReason.FINISHED
    assert result.usage.completion_tokens == 4


async def test_output_budget_stops_before_next_model_request(task, search):
    model = ScriptedModel([action("search", {"query": "срок"}, tokens=3)])
    result = await run_episode(task, model, search, EpisodeConfig(max_output_tokens=3))
    assert result.stop_reason == StopReason.MAX_OUTPUT_TOKENS
    assert result.truncated
    assert len(model.requests) == 1
    assert result.selected_document_ids == []


async def test_truncated_generation_does_not_execute_tools(task, search):
    response = action("search", {"query": "срок"})
    response.finish_reason = "length"
    result = await run_episode(task, ScriptedModel([response]), search, EpisodeConfig())
    assert result.stop_reason == StopReason.MAX_OUTPUT_TOKENS
    assert not search.calls


@pytest.mark.parametrize(("name", "arguments"), [
    ("read", {"doc_id": "doc-1"}),
    ("search", {"query": 123}),
    ("search", {"query": "  "}),
    ("search", {"query": "срок", "corpus_id": "other"}),
    ("finish", {"document_ids": "doc-1"}),
    ("finish", {"document_ids": ["invented"]}),
])
async def test_invalid_action_is_returned_and_can_be_repaired(task, search, name, arguments):
    model = ScriptedModel([
        action(name, arguments, call_id="bad"),
        action("finish", {"document_ids": []}, call_id="fixed"),
    ])
    result = await run_episode(task, model, search, EpisodeConfig(max_steps=3))
    assert result.stop_reason == StopReason.FINISHED
    assert result.error is None
    assert result.invalid_actions == 1
    assert not search.calls
    tool = next(m for m in model.requests[-1]["messages"] if m.get("tool_call_id") == "bad")
    assert tool["content"]
    assert result.trajectory[0].actions[0].result["ok"] is False


async def test_malformed_json_is_returned_to_model(task, search):
    response = action("search", {})
    response.message["tool_calls"][0]["function"]["arguments"] = "{broken"
    model = ScriptedModel([response, action("finish", {"document_ids": []}, call_id="fixed")])
    result = await run_episode(task, model, search, EpisodeConfig(max_steps=3))
    assert result.stop_reason == StopReason.FINISHED
    assert result.invalid_actions == 1
    assert not search.calls


async def test_more_than_two_errors_do_not_force_stop(task, search):
    model = ScriptedModel([
        *(action("finish", {"document_ids": ["invented"]}, call_id=f"bad-{i}") for i in range(3)),
        action("search", {"query": "срок"}, call_id="search"),
        action("finish", {"document_ids": ["doc-1"]}, call_id="finish"),
    ])
    result = await run_episode(task, model, search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert result.invalid_actions == 3
    assert result.selected_document_ids == ["doc-1"]


async def test_search_deduplication_and_observation_limits(task):
    class UnboundedSearch:
        async def search(self, *args, **kwargs):
            return [
                SearchHit(doc_id="doc-1", snippet="abcdef", rank=1),
                SearchHit(doc_id="doc-1", snippet="duplicate", rank=2),
                SearchHit(doc_id="doc-2", snippet="other", rank=3),
            ]

    model = ScriptedModel([
        action("search", {"query": "срок"}),
        action("finish", {"document_ids": ["doc-2"]}, call_id="bad"),
        action("finish", {"document_ids": ["doc-1"]}, call_id="fixed"),
    ])
    result = await run_episode(task, model, UnboundedSearch(), EpisodeConfig(search_limit=1, max_snippet_chars=3))
    assert result.trajectory[0].actions[0].result["hits"] == [{"doc_id": "doc-1", "snippet": "abc", "rank": 1}]
    assert result.trajectory[1].actions[0].result["error"]["code"] == "unknown_document"
    assert result.selected_document_ids == ["doc-1"]


async def test_missing_usage_is_an_infrastructure_error(task, search):
    response = action("search", {"query": "срок"})
    response.usage = Usage()
    result = await run_episode(task, ScriptedModel([response]), search, EpisodeConfig())
    assert result.stop_reason == StopReason.INFRASTRUCTURE_ERROR
    assert result.error.code == "missing_usage"
    assert result.usage.completion_tokens is None
    assert result.usage.cost is None
    assert not search.calls


async def test_backend_failure_is_distinct_and_redacted(task):
    class BrokenSearch:
        async def search(self, *args, **kwargs):
            raise RuntimeError("credentials=should-not-appear")

    result = await run_episode(task, MockModel(), BrokenSearch(), EpisodeConfig())
    assert result.stop_reason == StopReason.INFRASTRUCTURE_ERROR
    assert result.error.code == "search_error"
    assert "should-not-appear" not in result.model_dump_json()


async def test_model_failure_is_recorded(task, search):
    class BrokenModel(MockModel):
        async def complete(self, *args, **kwargs):
            raise RuntimeError("credentials=should-not-appear")

    result = await run_episode(task, BrokenModel(), search, EpisodeConfig())
    assert result.error.code == "model_error"
    assert result.usage.cost is None
    assert len(result.trajectory) == 1
    assert "should-not-appear" not in result.model_dump_json()


async def test_text_without_finish_is_unsuccessful_termination(task, search):
    response = ModelResponse(message={"role": "assistant", "content": "Готово."},
                             usage=Usage(prompt_tokens=5, completion_tokens=1), finish_reason="stop")
    model = ScriptedModel([response])
    result = await run_episode(task, model, search, EpisodeConfig())
    assert result.stop_reason == StopReason.INVALID_ACTION
    assert result.error.code == "missing_finish"
    assert len(model.requests) == 1
    assert result.selected_document_ids == []


async def test_multiple_calls_are_sequential_and_finish_stops_remaining_actions(task, search):
    response = action("search", {"query": "срок"})
    response.message["tool_calls"] += action("finish", {"document_ids": ["doc-1"]}, call_id="finish").message["tool_calls"]
    response.message["tool_calls"] += action("search", {"query": "лишний"}, call_id="extra").message["tool_calls"]
    result = await run_episode(task, ScriptedModel([response]), search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert len(search.calls) == 1
    assert result.trajectory[0].actions[-1].result["error"]["code"] == "episode_stopped"


async def test_deterministic_replay_from_saved_model_messages(task, search, tmp_path):
    original = await run_episode(task, MockModel(), search, EpisodeConfig(seed=42))
    saved = EpisodeResult.model_validate_json(save_episode(original, tmp_path / "episode.json").read_text())
    messages = [m for m in saved.history if m["role"] == "assistant"]
    replay_model = ScriptedModel([
        ModelResponse(message=message, usage=step.usage, finish_reason=step.finish_reason)
        for message, step in zip(messages, saved.trajectory, strict=True)
    ])
    replay = await run_episode(saved.task, replay_model, search, saved.config)
    assert replay.selected_document_ids == saved.selected_document_ids
    assert replay.stop_reason == saved.stop_reason
    assert replay.history == saved.history
    assert replay.usage == saved.usage


async def test_progress_is_emitted_before_model_and_tool_execution(task, search):
    events = []

    class ObservedModel(MockModel):
        async def complete(self, *args, **kwargs):
            assert events[-1].kind == "model_started"
            return await super().complete(*args, **kwargs)

    await run_episode(task, ObservedModel(), search, EpisodeConfig(), on_event=events.append)
    assert [event.kind for event in events] == [
        "model_started", "model_finished", "tool_started", "tool_finished",
        "model_started", "model_finished", "tool_started", "tool_finished",
    ]
    assert events[3].details["result"]["hits"][0]["doc_id"] == "doc-1"


async def test_cancellation_preserves_completed_search_and_unknown_pending_usage(task, search):
    waiting = asyncio.Event()
    partial = []

    class BlockedAfterSearch(MockModel):
        async def complete(self, messages, **kwargs):
            if any(message["role"] == "tool" for message in messages):
                waiting.set()
                await asyncio.Event().wait()
            return await super().complete(messages, **kwargs)

    operation = asyncio.create_task(run_episode(task, BlockedAfterSearch(), search, EpisodeConfig(), on_interrupted=partial.append))
    await asyncio.wait_for(waiting.wait(), timeout=15)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    result = partial[0]
    assert result.error.code == "interrupted"
    assert result.search_calls == 1
    assert len(result.trajectory) == 2
    assert result.trajectory[0].actions[0].result["ok"]
    assert result.usage.completion_tokens is None
    assert result.usage.cost is None


async def test_runner_timeout_keeps_standard_asyncio_semantics(task, search):
    class BlockedModel(MockModel):
        async def complete(self, *args, **kwargs):
            await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await run_episode(task, BlockedModel(), search, EpisodeConfig())


async def test_cancellation_during_search_records_pending_action(task):
    waiting = asyncio.Event()
    partial = []

    class BlockedSearch:
        async def search(self, *args, **kwargs):
            waiting.set()
            await asyncio.Event().wait()

    operation = asyncio.create_task(run_episode(task, MockModel(), BlockedSearch(), EpisodeConfig(), on_interrupted=partial.append))
    await asyncio.wait_for(waiting.wait(), timeout=15)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    result = partial[0]
    assert result.error.code == 'interrupted'
    assert result.native_trace['info']['episode']['error']['code'] == 'interrupted'
    assert result.search_calls == 1
    assert result.trajectory[0].actions[0].result['error']['code'] == 'interrupted'
    assert result.usage.completion_tokens == 1
    assert json.loads(result.history[-1]['content'])['error']['code'] == 'interrupted'

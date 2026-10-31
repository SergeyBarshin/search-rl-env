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
        finish_reason="tool_calls",
        model="scripted-test-model",
    )


class ScriptedModel:
    metadata = {"adapter": "scripted"}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def complete(self, messages, **kwargs):
        self.requests.append({"messages": deepcopy(messages), **kwargs})
        return next(self.responses)


async def test_complete_episode_and_json_roundtrip(task, search, tmp_path):
    result = await run_episode(task, MockModel(), search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert result.selected_document_ids == ["doc-1", "doc-2"]
    assert not result.truncated
    assert result.search_calls == 1
    assert result.usage.completion_tokens == 2
    assert [step.actions[0].name for step in result.trajectory] == ["search", "finish"]
    assert search.calls[0] == {"query": task.question, "corpus_id": "demo", "corpus_revision": "v1", "limit": 10}
    assert [m["tool_call_id"] for m in result.history if m["role"] == "tool"] == ["mock-call-1", "mock-call-2"]
    path = save_episode(result, tmp_path / "nested/episode.json")
    assert EpisodeResult.model_validate_json(path.read_text()) == result
    assert list(path.parent.iterdir()) == [path]


def test_task_rejects_reference_fields(task):
    with pytest.raises(ValidationError):
        AgentTask.model_validate({**task.model_dump(), "gold": {"relevant_doc_ids": ["doc-1"]}})


async def test_model_decides_number_of_searches_without_separate_limit(task, search):
    model = ScriptedModel([
        *(action("search", {"query": f"уточнение {i}"}, call_id=f"s-{i}") for i in range(4)),
        action("finish", {"document_ids": ["doc-2", "doc-1", "doc-2"]}, call_id="finish"),
    ])
    result = await run_episode(task, model, search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert result.search_calls == 4
    assert result.selected_document_ids == ["doc-2", "doc-1"]
    assert result.config.max_searches is None
    assert json.loads(model.requests[-1]["messages"][-1]["content"])["remaining_budget"]["searches"] is None


async def test_step_limit_does_not_select_all_found_documents(task, search):
    result = await run_episode(task, MockModel(), search, EpisodeConfig(max_steps=1))
    assert result.stop_reason == StopReason.MAX_STEPS
    assert result.truncated
    assert result.selected_document_ids == []
    assert len(result.trajectory) == 1


async def test_optional_search_limit_blocks_backend_call(task, search):
    model = ScriptedModel([action("search", {"query": "первый"}), action("search", {"query": "второй"}, call_id="call-2")])
    result = await run_episode(task, model, search, EpisodeConfig(max_searches=1))
    assert result.stop_reason == StopReason.MAX_SEARCHES
    assert result.truncated
    assert result.search_calls == len(search.calls) == 1
    assert result.trajectory[-1].actions[0].result["error"]["code"] == "search_budget_exhausted"


async def test_finish_is_allowed_after_last_search(task, search):
    result = await run_episode(task, MockModel(), search, EpisodeConfig(max_searches=1))
    assert result.stop_reason == StopReason.FINISHED


async def test_zero_search_budget(task, search):
    result = await run_episode(task, MockModel(), search, EpisodeConfig(max_searches=0))
    assert result.stop_reason == StopReason.MAX_SEARCHES
    assert not search.calls


async def test_total_output_budget_reduces_next_request_and_allows_exact_finish(task, search):
    model = ScriptedModel([
        action("search", {"query": "срок"}, tokens=2),
        action("finish", {"document_ids": ["doc-1"]}, call_id="call-2", tokens=1),
    ])
    result = await run_episode(task, model, search, EpisodeConfig(max_output_tokens=3, max_tokens_per_step=10))
    assert [request["max_output_tokens"] for request in model.requests] == [3, 1]
    assert result.stop_reason == StopReason.FINISHED
    assert result.usage.completion_tokens == 3


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
    assert result.trajectory[0].actions[0].result["error"]["code"] == "episode_stopped"


@pytest.mark.parametrize(("name", "arguments", "code"), [
    ("read", {"doc_id": "doc-1"}, "unknown_tool"),
    ("search", {"query": 123}, "invalid_arguments"),
    ("search", {"query": "  "}, "invalid_arguments"),
    ("search", {"query": "срок", "corpus_id": "other"}, "invalid_arguments"),
    ("finish", {"document_ids": "doc-1"}, "invalid_arguments"),
    ("finish", {"document_ids": ["invented"]}, "unknown_document"),
])
async def test_invalid_action(task, search, name, arguments, code):
    result = await run_episode(task, ScriptedModel([action(name, arguments)]), search, EpisodeConfig(max_invalid_actions=1))
    assert result.stop_reason == StopReason.INVALID_ACTION
    assert result.error.code == code
    assert not result.truncated
    assert not search.calls
    assert result.selected_document_ids == []


async def test_malformed_json_is_an_action_error(task, search):
    response = action("search", {})
    response.message["tool_calls"][0]["function"]["arguments"] = "{broken"
    result = await run_episode(task, ScriptedModel([response]), search, EpisodeConfig(max_invalid_actions=1))
    assert result.error.code == "invalid_arguments"


async def test_model_can_correct_action_error(task, search):
    model = ScriptedModel([
        action("search", {"query": "срок"}),
        action("finish", {"document_ids": ["invented"]}, call_id="bad-finish"),
        action("finish", {"document_ids": ["doc-1"]}, call_id="fixed-finish"),
    ])
    result = await run_episode(task, model, search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert result.invalid_actions == 1
    assert result.error is None
    tool_result = next(m for m in model.requests[-1]["messages"] if m.get("tool_call_id") == "bad-finish")
    assert json.loads(tool_result["content"])["error"]["code"] == "unknown_document"


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
        action("finish", {"document_ids": ["doc-2"]}, call_id="finish"),
    ])
    result = await run_episode(task, model, UnboundedSearch(), EpisodeConfig(search_limit=1, max_snippet_chars=3, max_invalid_actions=1))
    assert result.trajectory[0].actions[0].result["hits"] == [{"doc_id": "doc-1", "snippet": "abc", "rank": 1}]
    assert result.error.code == "unknown_document"


async def test_final_list_limit(task, search):
    result = await run_episode(task, ScriptedModel([
        action("search", {"query": "срок"}),
        action("finish", {"document_ids": ["doc-1", "doc-2"]}, call_id="finish"),
    ]), search, EpisodeConfig(max_documents=1, max_invalid_actions=1))
    assert result.error.code == "too_many_documents"


async def test_missing_usage_stops_instead_of_guessing_token_budget(task, search):
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
    assert result.search_calls == 1
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


async def test_no_tool_call_can_be_corrected(task, search):
    response = action("finish", {})
    response.message = {"role": "assistant", "content": "Готово."}
    result = await run_episode(task, ScriptedModel([
        response, action("finish", {"document_ids": []}, call_id="finish"),
    ]), search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert result.invalid_actions == 1


async def test_multiple_calls_are_sequential_and_finish_stops_remaining_actions(task, search):
    response = action("search", {"query": "срок"})
    response.message["tool_calls"] += action("finish", {"document_ids": ["doc-1"]}, call_id="finish").message["tool_calls"]
    response.message["tool_calls"] += action("search", {"query": "лишний"}, call_id="extra").message["tool_calls"]
    result = await run_episode(task, ScriptedModel([response]), search, EpisodeConfig())
    assert result.stop_reason == StopReason.FINISHED
    assert len(search.calls) == 1
    assert result.trajectory[0].actions[-1].result["error"]["code"] == "episode_stopped"
    assert [m["tool_call_id"] for m in result.history if m["role"] == "tool"] == ["call-1", "finish", "extra"]


async def test_repeated_call_id_is_a_protocol_error(task, search):
    model = ScriptedModel([action("search", {"query": "срок"}), action("finish", {"document_ids": ["doc-1"]})])
    result = await run_episode(task, model, search, EpisodeConfig())
    assert result.stop_reason == StopReason.INVALID_ACTION
    assert result.error.code == "invalid_tool_calls"


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


async def test_runner_cancellation_propagates(task, search):
    class CancelledModel(MockModel):
        async def complete(self, *args, **kwargs):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_episode(task, CancelledModel(), search, EpisodeConfig())

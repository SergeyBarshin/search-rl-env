import asyncio
import json

import httpx
import pytest

from search_rl_env.agent import run_episode
from search_rl_env.agent.client import ChatCompletionsClient, ModelError, OpenRouterClient
from search_rl_env.agent.mocks import MockSearch
from search_rl_env.agent.tools import tool_schemas
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit


@pytest.mark.parametrize(("client_type", "credentials", "endpoint", "authorization", "adapter"), [
    (OpenRouterClient, {"api_key": "fake-test-key"}, "https://openrouter.ai/api/v1/chat/completions",
     "Bearer fake-test-key", "openrouter-chat-completions"),
    (ChatCompletionsClient, {"api_key": "fake-test-key", "endpoint": "https://model.example.invalid/v1/chat/completions", "auth_scheme": "Token"}, "https://model.example.invalid/v1/chat/completions",
     "Token fake-test-key", "chat-completions"),
])
async def test_chat_completions_tool_protocol_without_network(client_type, credentials, endpoint, authorization, adapter):
    requests = []

    def handler(request):
        assert request.url == endpoint
        assert request.headers["authorization"] == authorization
        body = json.loads(request.content)
        requests.append(body)
        assert {tool["function"]["name"] for tool in body["tools"]} == {"search", "finish"}
        assert body["max_tokens"] == (3 if len(requests) == 1 else 2)
        assert body["seed"] == 42
        assert "parallel_tool_calls" not in body
        assert body["provider"] == {"require_parameters": True}
        name, arguments = "search", {"query": "срок"}
        if len(requests) == 2:
            tool = next(m for m in body["messages"] if m["role"] == "tool")
            assert tool["tool_call_id"] == "call-1"
            assert json.loads(tool["content"])["hits"][0]["doc_id"] == "doc-1"
            assistant = next(m for m in body["messages"] if m["role"] == "assistant")
            assert assistant["reasoning_details"] == [{"type": "reasoning.text", "text": "Нужен поиск."}]
            name, arguments = "finish", {"document_ids": ["doc-1"]}
        return httpx.Response(200, json={
            "id": f"gen-{len(requests)}", "model": "qwen/test", "provider": "test-provider",
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "reasoning_details": [{"type": "reasoning.text", "text": "Нужен поиск."}],
                "tool_calls": [{"id": f"call-{len(requests)}", "index": 0, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments),
                }}],
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11, "cost": 0.001,
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        })

    task = AgentTask(task_id="test", question="Срок?", language="ru", corpus_id="demo")
    search = MockSearch([SearchHit(doc_id="doc-1", snippet="Ноябрь.", rank=1)], corpus_id="demo")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with client_type(**credentials, model="qwen/test", http_client=http) as model:
            result = await run_episode(task, model, search, EpisodeConfig(max_output_tokens=3, seed=42))
        assert not http.is_closed
    assert result.stop_reason.value == "finished"
    assert result.selected_document_ids == ["doc-1"]
    assert result.usage.cost == pytest.approx(0.002)
    assert result.trajectory[0].provider == "test-provider"
    assert result.model_metadata["adapter"] == adapter
    assert result.trajectory[0].usage.model_extra["completion_tokens_details"] == {"reasoning_tokens": 0}
    assert "fake-test-key" not in result.model_dump_json()


async def test_total_request_deadline_stops_a_stalled_transport():
    async def handler(request):
        await asyncio.Event().wait()

    task = AgentTask(task_id="test", question="Срок?", language="ru", corpus_id="demo")
    search = MockSearch([], corpus_id="demo")
    # MockTransport не применяет фазовые таймауты httpx; нужен общий deadline.
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None) as http:
        model = ChatCompletionsClient(api_key="fake-test-key", endpoint="https://model.example.invalid/v1/chat/completions", model="qwen/test", timeout_seconds=0.02, http_client=http)
        result = await asyncio.wait_for(run_episode(task, model, search, EpisodeConfig()), timeout=2)
    assert result.stop_reason.value == "infrastructure_error"
    assert result.error.code == "model_timeout"
    assert "0.02" in result.error.message
    assert result.usage.completion_tokens is None
    assert result.model_metadata["timeout_seconds"] == 0.02


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_request_timeout_must_be_positive_and_finite(seconds):
    with pytest.raises(ValueError, match="timeout_seconds"):
        ChatCompletionsClient(api_key="fake-test-key", endpoint="https://model.example.invalid/v1/chat/completions", model="qwen/test", timeout_seconds=seconds)


@pytest.mark.parametrize(("status", "body"), [
    (401, {"error": {"message": "secret-body"}}),
    (503, {"error": {"message": "secret-body"}}),
    (200, {"error": {"message": "secret-body"}}),
    (200, {"choices": []}),
    (200, {"choices": [{"error": {"message": "secret-body"}}]}),
    (200, {"choices": [{"message": "wrong-type"}]}),
])
async def test_openrouter_errors_are_sanitized_and_not_retried(status, body):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = OpenRouterClient(api_key="fake-test-key", model="qwen/test", http_client=http)
        with pytest.raises(ModelError) as error:
            await model.complete([], tools=tool_schemas(), max_output_tokens=10, seed=None)
    assert "secret-body" not in str(error.value)
    assert "fake-test-key" not in str(error.value)
    assert len(calls) == 1


async def test_api_http_status_is_preserved_in_episode_error():
    task = AgentTask(task_id="test", question="Срок?", language="ru", corpus_id="demo")
    search = MockSearch([], corpus_id="demo")

    def handler(request):
        payload = json.loads(request.content)
        assert "seed" not in payload
        assert "parallel_tool_calls" not in payload
        return httpx.Response(401, json={"error": "secret-body"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        model = ChatCompletionsClient(api_key="fake-test-key", endpoint="https://model.example.invalid/v1/chat/completions", model="qwen/test", http_client=http)
        result = await run_episode(task, model, search, EpisodeConfig())
    assert result.error.code == "model_error"
    assert "HTTP 401" in result.error.message
    assert "secret-body" not in result.model_dump_json()
    assert "fake-test-key" not in result.model_dump_json()

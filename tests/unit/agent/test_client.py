import json

import httpx
import pytest

from search_rl_env.agent import run_episode
from search_rl_env.agent.client import ModelError, OpenRouterClient
from search_rl_env.agent.mocks import MockSearch
from search_rl_env.agent.tools import tool_schemas
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit


async def test_openrouter_tool_protocol_without_network():
    requests = []

    def handler(request):
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer fake-test-key"
        body = json.loads(request.content)
        requests.append(body)
        assert {tool["function"]["name"] for tool in body["tools"]} == {"search", "finish"}
        assert body["max_tokens"] == (3 if len(requests) == 1 else 2)
        assert body["seed"] == 42
        assert body["parallel_tool_calls"] is False
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
                "tool_calls": [{"id": f"call-{len(requests)}", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments),
                }}],
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11, "cost": 0.001,
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        })

    task = AgentTask(task_id="test", question="Срок?", language="ru", corpus_id="demo")
    search = MockSearch([SearchHit(doc_id="doc-1", snippet="Ноябрь.", rank=1)], corpus_id="demo")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with OpenRouterClient(api_key="fake-test-key", model="qwen/test", http_client=http) as model:
            result = await run_episode(task, model, search, EpisodeConfig(max_output_tokens=3, seed=42))
        assert not http.is_closed
    assert result.stop_reason.value == "finished"
    assert result.selected_document_ids == ["doc-1"]
    assert result.usage.cost == pytest.approx(0.002)
    assert result.trajectory[0].provider == "test-provider"
    assert result.trajectory[0].usage.model_extra["completion_tokens_details"] == {"reasoning_tokens": 0}
    assert "fake-test-key" not in result.model_dump_json()


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

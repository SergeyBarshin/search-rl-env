import asyncio
import json
from pathlib import Path

import httpx
import pytest
from aiohttp.test_utils import TestClient, TestServer

from search_rl_env.agent.client import ChatCompletionsClient
from search_rl_env.agent.mocks import MockModel
from search_rl_env.contracts import EpisodeResult
from search_rl_env.ui import server


@pytest.fixture
async def ui(tmp_path, monkeypatch):
    for name in ("MODEL_API_KEY", "MODEL_API_KEY_FILE", "MODEL_ID", "MODEL_API_URL", "MODEL_AUTH_SCHEME"):
        monkeypatch.delenv(name, raising=False)
    client = TestClient(TestServer(server.create_app(output_dir=tmp_path / "runs")))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


async def read_events(ui, run_id, *, after=0):
    async with asyncio.timeout(20):
        response = await ui.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(after)})
        assert response.status == 200
        events = []
        async for line in response.content:
            if line.startswith(b"data: "):
                events.append(json.loads(line[6:]))
        return events


async def start(ui, **overrides):
    response = await ui.post("/api/runs", json={"question": "Какой новый срок проекта?", **overrides})
    assert response.status == 202, await response.text()
    return (await response.json())["id"]


async def test_page_assets_and_settings_do_not_expose_credentials(ui, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", "https://private.example.invalid/v1/chat/completions")
    monkeypatch.setenv("MODEL_API_KEY", "credential-must-stay-on-server")
    monkeypatch.setenv("MODEL_ID", "qwen/test")
    settings = await ui.get("/api/settings")
    text = await settings.text()
    assert "credential-must-stay-on-server" not in text
    assert "private.example.invalid" not in text
    assert (await settings.json())["api_error"] is None
    page = await ui.get("/")
    assert page.status == 200
    assert "Проверим агента" in await page.text()
    for asset in ("app.js", "app.css"):
        response = await ui.get("/assets/" + asset)
        assert response.status == 200
        assert len(await response.read()) > 100


async def test_real_mock_episode_streams_and_saves_native_trace(ui):
    run_id = await start(ui)
    events = await read_events(ui, run_id)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    progress = [event["data"] for event in events if event["type"] == "progress"]
    assert [event["details"]["name"] for event in progress if event["kind"] == "tool_started"] == ["search", "finish"]
    complete = events[-1]
    assert complete["type"] == "complete"
    result = complete["data"]["result"]
    assert result["stop_reason"] == "finished"
    assert result["task"]["question"] == "Какой новый срок проекта?"
    assert result["selected_document_ids"] == ["doc-001", "doc-002"]
    assert result["native_trace"]["verifiers"]["version"] == "0.4.0"
    assert Path(complete["data"]["saved_to"]).is_file()
    download = await ui.get(complete["data"]["download_url"])
    assert download.status == 200
    assert "attachment" in download.headers["Content-Disposition"]
    assert EpisodeResult.model_validate_json(await download.text()).episode_id == result["episode_id"]
    replay = await read_events(ui, run_id, after=complete["seq"] - 1)
    assert replay == [complete]


async def test_api_mode_uses_configured_client_without_external_requests(ui, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", "https://model.example.invalid/v1/chat/completions")
    monkeypatch.setenv("MODEL_API_KEY", "fake-ui-credential")
    monkeypatch.setenv("MODEL_ID", "qwen/test")
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["Authorization"] == "Bearer fake-ui-credential"
        assert body["model"] == "qwen/override"
        reply = await MockModel().complete(body["messages"], tools=body["tools"], max_output_tokens=body["max_tokens"], seed=None)
        return httpx.Response(200, json={
            "id": "response-" + str(len(requests)), "model": body["model"],
            "choices": [{"message": reply.message, "finish_reason": reply.finish_reason}],
            "usage": reply.usage.model_dump(),
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        monkeypatch.setattr(server, "ChatCompletionsClient", lambda **kwargs: ChatCompletionsClient(**kwargs, http_client=http))
        events = await read_events(ui, await start(ui, mode="api", model_id="qwen/override"))
    result = events[-1]["data"]["result"]
    assert result["stop_reason"] == "finished"
    assert result["model_metadata"]["adapter"] == "chat-completions"
    assert len(requests) == 2
    assert "fake-ui-credential" not in json.dumps(events)


async def test_stopping_pending_model_saves_partial_result(ui, monkeypatch):
    waiting = asyncio.Event()

    class WaitingModel(MockModel):
        async def complete(self, *args, **kwargs):
            waiting.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(server, "MockModel", WaitingModel)
    run_id = await start(ui)
    await asyncio.wait_for(waiting.wait(), timeout=15)
    response = await ui.post(f"/api/runs/{run_id}/cancel", json={})
    assert response.status == 200
    events = await read_events(ui, run_id)
    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["data"]["result"]["error"]["code"] == "interrupted"
    assert complete["data"]["result"]["usage"]["completion_tokens"] is None
    assert Path(complete["data"]["saved_to"]).is_file()


async def test_step_cap_is_preserved_through_ui(ui):
    run_id = await start(ui, max_steps=1)
    complete = (await read_events(ui, run_id))[-1]
    assert complete["data"]["result"]["stop_reason"] == "max_steps"
    assert complete["data"]["result"]["truncated"] is True
    assert complete["data"]["result"]["selected_document_ids"] == []


@pytest.mark.parametrize("payload", [
    {"question": " "}, {"question": "Срок?", "max_steps": 0},
    {"question": "Срок?", "max_output_tokens": -1},
    {"question": "Срок?", "max_tokens_per_step": "many"},
])
async def test_invalid_input_does_not_start_a_run(ui, payload):
    response = await ui.post("/api/runs", json=payload)
    assert response.status == 400
    assert not ui.server.app[server.WORKBENCH].runs


async def test_unconfigured_api_is_rejected_before_execution(ui):
    response = await ui.post("/api/runs", json={"question": "Срок?", "mode": "api"})
    assert response.status == 400
    assert "MODEL_API_URL" in (await response.json())["error"]
    assert not ui.server.app[server.WORKBENCH].runs


async def test_other_origins_and_nonlocal_hosts_cannot_start_runs(ui):
    for headers in ({"Origin": "https://other.example.invalid"}, {"Host": "other.example.invalid"}):
        response = await ui.post("/api/runs", json={"question": "Срок?"}, headers=headers)
        assert response.status == 403
    response = await ui.post("/api/runs", data={"question": "Срок?"})
    assert response.status == 415
    assert not ui.server.app[server.WORKBENCH].runs


async def test_unknown_run_returns_not_found(ui):
    response = await ui.get("/api/runs/not-found/result")
    assert response.status == 404

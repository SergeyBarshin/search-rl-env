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
    assert result['profile']['setup_seconds'] > 0
    assert result['profile']['wall_seconds'] == result['duration_seconds']
    assert Path(complete["data"]["saved_to"]).is_file()
    download = await ui.get(complete["data"]["download_url"])
    assert download.status == 200
    assert "attachment" in download.headers["Content-Disposition"]
    assert EpisodeResult.model_validate_json(await download.text()).episode_id == result["episode_id"]
    replay = await read_events(ui, run_id, after=complete["seq"] - 1)
    assert replay == [complete]
    log = await ui.get(complete['data']['events_url'])
    assert log.status == 200
    log_text = await log.text()
    records = [json.loads(line) for line in log_text.splitlines()]
    assert records[-1]['event'] == 'complete'
    assert records[-1]['episode_id'] == result['episode_id']
    assert 'Какой новый срок проекта?' not in log_text
    assert any(record.get('kind') == 'episode_phase' for record in records)
    status = await (await ui.get('/api/runs')).json()
    assert status['runs'][0]['done']
    ui.server.app[server.WORKBENCH].runs.clear()
    recovered = await ui.get(complete['data']['download_url'])
    assert recovered.status == 200
    assert (await recovered.json())['episode_id'] == result['episode_id']


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
        monkeypatch.setattr(server, "ChatCompletionsClient", lambda **kwargs: ChatCompletionsClient(**{**kwargs, 'http_client': http}))
        events = await read_events(ui, await start(ui, mode="api", model_id="qwen/override"))
    result = events[-1]["data"]["result"]
    assert result["stop_reason"] == "finished"
    assert result["model_metadata"]["adapter"] == "chat-completions"
    assert len(requests) == 2
    assert "fake-ui-credential" not in json.dumps(events)
    ui.server.app[server.WORKBENCH].runs.clear()
    restored = await ui.get(events[-1]['data']['download_url'])
    assert restored.status == 200
    statuses = await (await ui.get('/api/runs')).json()
    assert statuses['runs'][0]['mode'] == 'api'


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


async def test_hybrid_scenario_returns_sql_rows_answer_and_sources(ui):
    settings = await (await ui.get('/api/settings')).json()
    scenario = next(item for item in settings['scenarios'] if item['id'] == 'sql_rag')
    assert 'CREATE TABLE orders' in scenario['sql_schema']
    assert len(scenario['rows']) == 6
    run_id = await start(ui, question=scenario['question'], scenario='sql_rag')
    events = await read_events(ui, run_id)
    result = events[-1]['data']['result']
    assert result['schema_version'] == '3'
    assert result['answer'] == ['ORD-101', 'ORD-105']
    assert result['selected_document_ids'] == ['policy-001']
    assert result['sql_calls'] == 1
    assert result['stop_reason'] == 'finished'
    assert [event['data']['details']['name'] for event in events if event['type'] == 'progress' and event['data']['kind'] == 'tool_started'] == ['search', 'sql', 'finish']


async def test_ui_api_episodes_reuse_the_application_http_pool(ui, monkeypatch):
    from aiohttp import web

    monkeypatch.setenv('MODEL_API_KEY', 'test-pool-credential')
    monkeypatch.setenv('MODEL_ID', 'qwen/test')
    shared = []
    peers = []

    async def provider(request):
        peers.append(request.transport.get_extra_info('peername'))
        body = await request.json()
        reply = await MockModel().complete(body['messages'], tools=body['tools'], max_output_tokens=body['max_tokens'], seed=None)
        return web.json_response({'model': 'qwen/test', 'choices': [
            {'message': reply.message, 'finish_reason': reply.finish_reason}], 'usage': reply.usage.model_dump()})

    def factory(**kwargs):
        shared.append(kwargs['http_client'])
        return ChatCompletionsClient(**kwargs)

    provider_app = web.Application()
    provider_app.router.add_post('/v1/chat/completions', provider)
    async with TestServer(provider_app) as endpoint:
        monkeypatch.setenv('MODEL_API_URL', str(endpoint.make_url('/v1/chat/completions')))
        monkeypatch.setattr(server, 'ChatCompletionsClient', factory)
        reused = []
        for _ in range(2):
            events = await read_events(ui, await start(ui, mode='api'))
            result = events[-1]['data']['result']
            assert result['stop_reason'] == 'finished'
            reused.extend(step['http_profile']['connection_reused'] for step in result['trajectory'])
        assert shared[0] is shared[1] is ui.server.app[server.WORKBENCH].http
        assert not shared[0].is_closed
        assert reused == [False, True, True, True]
        assert len(peers) == 4 and len(set(peers)) == 1

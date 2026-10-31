import asyncio
import json

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from search_rl_env.agent import run_episode
from search_rl_env.agent.client import ChatCompletionsClient
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit
from search_rl_env.contracts.types import ModelTimeoutError
from search_rl_env.profiling import EventLog, HttpRecorder, observe_http


def completion():
    return {'id': 'test-response', 'model': 'qwen/test',
            'choices': [{'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 1, 'total_tokens': 6}}


async def test_loopback_http_separates_headers_body_and_reuses_connection():
    payload = json.dumps(completion()).encode()

    async def handler(request):
        await request.read()
        response = web.StreamResponse(headers={
            'Content-Type': 'application/json', 'x-request-id': 'provider-request-123',
            'Authorization': 'response-header-secret',
            'Server-Timing': 'queue;dur=25, inference;dur=40;desc="private-description", private;dur=99',
        })
        await response.prepare(request)
        await asyncio.sleep(0.015)
        await response.write(payload[:5])
        await asyncio.sleep(0.015)
        await response.write(payload[5:])
        return response

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handler)
    async with TestServer(app) as endpoint:
        async with ChatCompletionsClient(api_key='fake-key', model='qwen/test', endpoint=str(endpoint.make_url('/v1/chat/completions'))) as model:
            profiles = []
            events = []
            for _ in range(2):
                with observe_http(lambda profile, phase, status: events.append((profile, phase, status))):
                    result = await model.complete([], tools=[], max_output_tokens=10, seed=None)
                profiles.append(result.http_profile)
    first, second = profiles
    assert first.connection_reused is False
    assert second.connection_reused is True
    assert first.request_id != second.request_id
    for profile in profiles:
        assert 0 <= profile.time_to_headers_seconds < profile.time_to_first_body_byte_seconds < profile.duration_seconds
        assert profile.body_read_seconds > 0
        assert profile.decode_seconds >= 0
        assert profile.request_bytes > 0
        assert profile.response_bytes == len(payload)
        assert profile.provider_request_id == 'provider-request-123'
        assert profile.provider_timing_seconds == {'queue': 0.025, 'inference': 0.04}
        assert profile.time_to_transport_seconds is not None
        assert 'response-header-secret' not in profile.model_dump_json()
        assert 'private-description' not in profile.model_dump_json()
        assert all(phase.status == 'complete' for phase in profile.phases)
    assert any(phase.name.endswith('connect_tcp') for phase in first.phases)
    assert not any(phase.name.endswith('connect_tcp') for phase in second.phases)
    assert any(phase == 'first_body_byte' for _, phase, _ in events)


async def test_timeout_keeps_partial_body_profile_and_closes_stream():
    class WaitingBody(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"partial":'
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    body = WaitingBody()

    async def handler(request):
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = ChatCompletionsClient(api_key='fake-key', model='qwen/test', endpoint='https://model.example.invalid/v1', http_client=http, timeout_seconds=0.03)
        with pytest.raises(ModelTimeoutError) as error:
            await model.complete([], tools=[], max_output_tokens=10, seed=None)
    profile = error.value.http_profile
    assert profile.status == 'failed'
    assert profile.error_code == 'model_timeout'
    assert profile.status_code == 200
    assert profile.time_to_first_body_byte_seconds is not None
    assert profile.body_read_seconds > 0
    assert profile.decode_seconds is None
    assert profile.response_bytes == len(b'{"partial":')
    assert profile.connection_reused is None
    assert body.closed


async def test_cancellation_keeps_profile_without_faking_response_time():
    waiting = asyncio.Event()

    async def handler(request):
        waiting.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = ChatCompletionsClient(api_key='fake-key', model='qwen/test', endpoint='https://model.example.invalid/v1', http_client=http)
        task = asyncio.create_task(model.complete([], tools=[], max_output_tokens=10, seed=None))
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as error:
            await task
    profile = error.value.http_profile
    assert profile.status == 'cancelled'
    assert profile.error_code == 'interrupted'
    assert profile.time_to_headers_seconds is None
    assert profile.time_to_first_body_byte_seconds is None
    assert profile.request_bytes > 0


async def test_trace_drops_private_info_and_retains_failed_phase():
    recorder = HttpRecorder()
    await recorder.trace('connection.connect_tcp.started', {'host': 'secret-host'})
    await recorder.trace('connection.connect_tcp.complete', {'return_value': 'secret-socket'})
    await recorder.trace('connection.start_tls.started', {'server_hostname': 'secret-host'})
    await recorder.trace('connection.start_tls.failed', {'exception': RuntimeError('credential-secret')})
    profile = recorder.finish(error_code='model_error')
    assert [phase.name for phase in profile.phases] == ['connection.connect_tcp', 'connection.start_tls']
    assert profile.phases[-1].status == 'failed'
    assert profile.phases[-1].error_type == 'RuntimeError'
    assert 'secret' not in profile.model_dump_json()


def test_event_log_contains_metrics_without_payloads_or_auth(tmp_path):
    log = EventLog(tmp_path / 'events.jsonl', run_id='ui-run')
    log.write('progress', {'kind': 'tool_finished', 'step': 1, 'episode_id': 'episode', 'elapsed_seconds': 0.2,
              'details': {'name': 'sql', 'duration_seconds': 0.1, 'arguments': 'sensitive-query',
                          'headers': {'Authorization': 'secret-key'}, 'result': {'ok': True, 'rows': [['private-row']]}}}, seq=1)
    text = log.path.read_text()
    data = json.loads(text)
    assert data['run_id'] == 'ui-run'
    assert data['episode_id'] == 'episode'
    assert data['details']['duration_seconds'] == 0.1
    assert data['details']['ok'] is True
    assert not any(value in text for value in ('sensitive-query', 'secret-key', 'private-row'))


async def test_parallel_episodes_share_client_but_not_profiles_or_tool_state():
    async def handler(request):
        body = json.loads(request.content)
        reply = await MockModel().complete(body['messages'], tools=body['tools'], max_output_tokens=body['max_tokens'], seed=None)
        return httpx.Response(200, json={'model': 'qwen/test', 'choices': [{'message': reply.message, 'finish_reason': reply.finish_reason}], 'usage': reply.usage.model_dump()})

    events = {'one': [], 'two': []}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = ChatCompletionsClient(api_key='fake-key', model='qwen/test', endpoint='https://model.example.invalid/v1', http_client=http)

        async def episode(name):
            task = AgentTask(task_id=name, question='Срок?', language='ru', corpus_id=name)
            search = MockSearch([SearchHit(doc_id=name, rank=1, snippet='Ноябрь.')], corpus_id=name)
            return await run_episode(task, model, search, EpisodeConfig(), on_event=events[name].append)

        first, second = await asyncio.gather(episode('one'), episode('two'))
        assert not http.is_closed
    assert first.selected_document_ids == ['one']
    assert second.selected_document_ids == ['two']
    request_ids = [step.http_profile.request_id for result in (first, second) for step in result.trajectory]
    assert len(set(request_ids)) == 4
    for name, result in (('one', first), ('two', second)):
        assert result.profile.wall_seconds == result.duration_seconds
        assert result.profile.setup_seconds > 0
        assert result.profile.model_seconds > 0
        assert result.profile.model_seconds < result.profile.wall_seconds
        assert all(event.episode_id == result.episode_id for event in events[name])
        assert result.native_trace['info']['profile'] == result.profile.model_dump(mode='json')
        assert 'fake-key' not in result.model_dump_json()

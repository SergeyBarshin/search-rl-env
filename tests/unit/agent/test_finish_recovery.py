"""Проверки транспорта harness без открытия TCP-портов и обращения к API."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from search_rl_env.agent.harness import SearchHarness, SearchHarnessConfig


class Reply:
    def __init__(self, message, status=200):
        self.message, self.status = message, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        return {'choices': [{'message': self.message}]}


class Executor:
    config = SimpleNamespace(enable_sql=True)

    def __init__(self, trace):
        self.trace = trace
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, call):
        self.calls.append(call)
        if call['function']['name'] == 'finish':
            self.trace.stop_condition = 'task_finished'
        return {'ok': True}


class Transport:
    def __init__(self, trace, replies):
        self.trace, self.replies = trace, iter(replies)
        self.requests = []

    def post(self, url, *, headers, json):
        # Аналог native interception после finish: новая генерация не запускается.
        if self.trace.stop_condition:
            return Reply(None, 400)
        self.requests.append(deepcopy(json))
        return Reply(next(self.replies))


async def run_harness(replies, retries):
    trace = SimpleNamespace(stop_condition=None, state=SimpleNamespace())
    executor = trace.state._executor = Executor(trace)
    client = Transport(trace, replies)
    harness = SearchHarness(SearchHarnessConfig(max_finish_retries=retries))
    harness.resolve_prompt = lambda data: ('Use tools.', 'A public question')
    await harness._run_local(client, SimpleNamespace(model='test'), trace,
                             'http://unused.invalid', 'not-a-key', None, None)
    return client, executor


@pytest.mark.parametrize('content', [None, 'Here is my answer.'])
async def test_missing_finish_reminder_preserves_original_response_and_tools(content):
    finish = {'role': 'assistant', 'tool_calls': [{'id': 'done', 'type': 'function',
        'function': {'name': 'finish', 'arguments': '{"answer":["supported"],"document_ids":[]}'}}]}
    client, executor = await run_harness([
        {'role': 'assistant', 'content': content}, finish,
    ], retries=2)
    assert len(client.requests) == 2
    history = client.requests[1]['messages']
    assert history[-2] == {'role': 'assistant', 'content': content}
    assert history[-1]['role'] == 'user'
    assert 'No final answer has been submitted' in history[-1]['content']
    assert executor.calls == finish['tool_calls']


async def test_empty_responses_stop_after_bounded_recovery_without_fabricated_finish():
    client, executor = await run_harness([{'role': 'assistant', 'content': None}] * 3, retries=2)
    assert len(client.requests) == 3
    assert executor.calls == []


async def test_default_does_not_retry_or_change_strict_finish_contract():
    client, executor = await run_harness([{'role': 'assistant', 'content': None}], retries=0)
    assert len(client.requests) == 1
    assert executor.calls == []

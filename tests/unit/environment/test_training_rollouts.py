"""Контракт TrainClient -> native graph. Детерминированный renderer, без весов/GPU."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import verifiers.v1 as vf
import pytest
from renderers import RenderedTokens

from search_rl_env.agent.harness import SearchHarnessConfig
from search_rl_env.agent.mocks import MockModel
from search_rl_env.agent import RolloutRunner
from search_rl_env.environment.taskset import SearchTaskset, SearchTasksetConfig


@pytest.mark.parametrize('bridge_interception', [False, True])
async def test_parallel_native_train_client_keeps_token_logprob_and_tool_masks(monkeypatch, bridge_interception):
    import renderers.client
    import verifiers.v1.clients.train as training

    class TinyRenderer:
        def render(self, messages, **kwargs):
            ids, indices, sampled = [], [], []
            for index, message in enumerate(messages):
                role = message['role']
                tokens = {'system': [10], 'user': [20], 'tool': [40]}.get(role)
                if tokens is None:
                    name = message['tool_calls'][0]['function']['name']
                    tokens = [30, 31] if name == 'search' else [50, 51]
                ids.extend(tokens)
                indices.extend([index] * len(tokens))
                sampled.extend([role == 'assistant'] * len(tokens))
            return RenderedTokens(token_ids=ids, message_indices=indices, sampled_mask=sampled,
                                  message_roles=[message['role'] for message in messages])

        def bridge_to_next_turn(self, *args, **kwargs):
            return None

    class Slot:
        renderer = TinyRenderer()

        async def run(self, fn):
            return fn()

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def warm(self):
            pass

        @asynccontextmanager
        async def acquire(self):
            yield Slot()

    requests = []

    async def generate(**kwargs):
        requests.append(kwargs)
        await asyncio.sleep(.001)
        reply = await MockModel().complete(kwargs['messages'], tools=kwargs['tools'], max_output_tokens=10, seed=0)
        call = reply.message['tool_calls'][0]
        searching = call['function']['name'] == 'search'
        return {
            'request_id': kwargs['extra_headers']['X-Session-ID'],
            'prompt_ids': kwargs['prompt_ids'], 'prompt_attribution': kwargs['prompt_attribution'],
            'completion_ids': [30, 31] if searching else [50, 51],
            'completion_logprobs': [-.1, -.2] if searching else [-.3, -.4],
            'tool_calls': [SimpleNamespace(id=call['id'], **call['function'])], 'finish_reason': 'tool_calls',
        }

    monkeypatch.setattr(training, 'ElasticRendererPool', Pool)
    monkeypatch.setattr(renderers.client, 'generate', generate)
    task = next(SearchTaskset(SearchTasksetConfig()).load())
    config = vf.AgentConfig(model='tiny-test', harness=SearchHarnessConfig(), runtime=vf.SubprocessConfig(),
        network_notice=False, client=vf.TrainClientConfig(base_url='http://127.0.0.1:1',
            api_key_var='UNUSED_TEST_API_KEY', renderer_model_name='tiny-test'))
    async with RolloutRunner(concurrency=4) as runner, vf.Agent(
        config, interception=runner.interception if bridge_interception else None,
    ) as agent:
        # Отдельные Task: callbacks и приватные инструменты тоже принадлежат rollout.
        traces = await asyncio.gather(*(agent.run(type(task)(task.data.model_copy(update={'id': str(index)}),
                                                                          task.config)) for index in range(4)))
        assert len((runner.interception if bridge_interception else agent._server).clients) == 1
    assert len(requests) == 8
    assert len({request['extra_headers']['X-Session-ID'] for request in requests}) == 4
    for trace in traces:
        assert trace.ok, trace.errors
        assert trace.reward == 1
        assert trace.stop_condition == 'task_finished'
        assert trace.num_turns == 2
        branch = trace.branches[0]
        assert branch.token_ids == [10, 20, 30, 31, 40, 50, 51]
        assert branch.sampled_mask == [False, False, True, True, False, True, True]
        assert branch.logprobs == [0., 0., -.1, -.2, 0., -.3, -.4]
        observation = next(node for node in trace.nodes if node.message.role == 'tool')
        assert observation.token_ids == [40]
        assert not observation.sampled and not any(observation.mask)
        assert 'reference_document_ids' not in json.dumps(requests[0]['messages'])
        restored = vf.Trace.model_validate(trace.to_record())
        assert restored.branches[0].sampled_mask == branch.sampled_mask

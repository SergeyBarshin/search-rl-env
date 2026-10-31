import asyncio
import json
import os
from contextlib import aclosing

import pytest
import verifiers.v1 as vf

from search_rl_env.agent import RolloutRequest, RolloutRunner
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit


def task(index):
    return AgentTask(task_id=str(index), question=f'Вопрос {index}', language='ru', corpus_id='demo')


def search():
    return MockSearch([SearchHit(doc_id='a', snippet='Срок изменён.', rank=1)], corpus_id='demo')


async def test_bounded_parallelism_reuses_clients_and_keeps_episode_ids():
    active = peak = 0

    class Delayed(MockModel):
        async def complete(self, messages, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.01)
                return await super().complete(messages, **kwargs)
            finally:
                active -= 1

    events = []
    runner = RolloutRunner(Delayed(), search(), concurrency=3)
    async with runner:
        results = [result async for result in runner.iter_episodes(
            (RolloutRequest(task(i), reference_document_ids=['a']) for i in range(12)), on_event=events.append)]
        assert len(runner.interception.clients) == 1
        assert runner.interception.load == 0
        assert runner.bridge.bridges == runner.bridge.traces == {}
        assert not runner.http.closed
        key_var = runner.bridge.key_var
        assert key_var in os.environ
    assert key_var not in os.environ
    assert runner.http.closed
    assert 1 < peak <= 3
    assert len({result.episode_id for result in results}) == 12
    assert {result.task_id for result in results} == {str(i) for i in range(12)}
    assert all(result.reward == 1 and result.selected_document_ids == ['a'] for result in results)
    assert {event.episode_id for event in events} == {result.episode_id for result in results}
    assert len({result.native_trace['agent']['config']['client']['base_url'] for result in results}) == 1


async def test_results_are_delivered_before_slow_episode_and_batch_close_cancels_it():
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class BlockFirst(MockModel):
        async def complete(self, messages, **kwargs):
            if json.loads(messages[1]['content'])['task_id'] == 'slow':
                started.set()
                try:
                    await release.wait()
                finally:
                    cancelled.set()
            return await super().complete(messages, **kwargs)

    async with RolloutRunner(BlockFirst(), search(), concurrency=2) as runner:
        async with aclosing(runner.iter_episodes([task('slow'), task('fast')])) as results:
            first = await asyncio.wait_for(anext(results), 5)
            assert started.is_set()
            assert first.task_id == 'fast'
            assert not cancelled.is_set()
        assert cancelled.is_set()
        assert runner.interception.load == 0
        assert runner.bridge.bridges == runner.bridge.traces == {}
        # Отмена batch не закрывает общий транспорт.
        assert (await runner.run(task('again'))).stop_reason == 'finished'


async def test_async_source_and_bounded_buffer_when_consumer_pauses():
    consumed = 0

    async def source():
        nonlocal consumed
        for i in range(1000):
            consumed += 1
            yield task(i)

    async with RolloutRunner(MockModel(), search(), concurrency=2) as runner:
        async with aclosing(runner.iter_episodes(source())) as results:
            await anext(results)
            await asyncio.sleep(.08)
            # Два workers + буфер из двух результатов + уже полученный результат.
            assert consumed <= 5
        assert not runner._active and not runner._workers


async def test_runner_cancels_owned_runs_before_closing_transport():
    started = asyncio.Event()
    partials = []

    class Blocked(MockModel):
        async def complete(self, *args, **kwargs):
            started.set()
            await asyncio.Event().wait()

    runner = RolloutRunner(Blocked(), search(), concurrency=1)
    async with runner:
        operation = asyncio.create_task(runner.run(task('blocked'), on_interrupted=partials.append))
        await asyncio.wait_for(started.wait(), 5)
    assert operation.cancelled()
    assert partials[0].error.code == 'interrupted'
    assert runner.bridge.bridges == runner.bridge.traces == {}
    assert runner.interception.load == 0


async def test_bad_source_cancels_siblings_and_propagates_error():
    def broken():
        yield task('one')
        raise ValueError('bad task source')

    async with RolloutRunner(MockModel(), search(), concurrency=2) as runner:
        with pytest.raises(ValueError, match='bad task source'):
            async for _ in runner.iter_episodes(broken()):
                pass
        assert not runner._active
        assert runner.bridge.bridges == runner.bridge.traces == {}
        assert (await runner.run(task('after'))).stop_reason == 'finished'


@pytest.mark.parametrize('concurrency', [0, -1, True, 1.5])
def test_invalid_concurrency(concurrency):
    with pytest.raises(ValueError):
        RolloutRunner(concurrency=concurrency)


async def test_native_state_isolated_between_searches_failures_and_finish():
    class PerTaskSearch:
        async def search(self, query, **kwargs):
            await asyncio.sleep(.001)
            if query == 'Вопрос broken':
                raise RuntimeError('secret-backend-value')
            return [SearchHit(doc_id=query, snippet=query, rank=1)]

    async with RolloutRunner(MockModel(), PerTaskSearch(), concurrency=4) as runner:
        results = {result.task_id: result async for result in runner.iter_episodes(
            [task('a'), task('broken'), task('b'), RolloutRequest(task('cap'), config=EpisodeConfig(max_steps=1))])}
    assert results['broken'].error.code == 'search_error'
    assert results['cap'].stop_reason == 'max_steps'
    for name in ['a', 'b']:
        assert results[name].selected_document_ids == ['Вопрос ' + name]
        assert results[name].error is None
    assert 'secret-backend-value' not in ''.join(result.model_dump_json() for result in results.values())


async def test_subprocess_mcp_fallback_keeps_same_behavior():
    results = []
    for execution in ['in_process', 'subprocess']:
        async with RolloutRunner(MockModel(), search(), concurrency=1, execution=execution) as runner:
            results.append(await runner.run(task('same'), reference_document_ids=['a']))
    first, second = results
    assert first.stop_reason == second.stop_reason == 'finished'
    assert first.selected_document_ids == second.selected_document_ids == ['a']
    assert first.usage == second.usage
    assert first.reward == second.reward == 1
    assert [action.result for step in first.trajectory for action in step.actions] == [
        action.result for step in second.trajectory for action in step.actions]


async def test_native_tool_gate_prevents_execution_and_keeps_observation(monkeypatch):
    import search_rl_env.agent.loop as loop
    from search_rl_env.environment.taskset import SearchTask

    class GuardedTask(SearchTask):
        @vf.intercept
        async def deny_search(self, request: vf.Request) -> vf.Request:
            message = request.messages[-1]
            if message.role == 'tool' and message.name == 'search':
                message.content = json.dumps({'ok': False, 'error': {'code': 'test_gate', 'message': 'Denied by test.'}})
            return request

    monkeypatch.setattr(loop, 'SearchTask', GuardedTask)
    backend = search()
    async with RolloutRunner(MockModel(), backend) as runner:
        result = await runner.run(task('gated'), reference_document_ids=['a'])
    assert not backend.calls
    assert result.stop_reason == 'finished'
    assert result.reward == 0
    assert result.invalid_actions == 1
    assert result.trajectory[0].actions[0].result['ok'] is False
    assert result.native_trace['request_rewrites']
    assert 'test_gate' in next(message['content'] for message in result.history if message['role'] == 'tool')

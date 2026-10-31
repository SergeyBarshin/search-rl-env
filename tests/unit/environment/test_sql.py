import json

import pytest

from search_rl_env.agent import run_episode
from search_rl_env.agent.mocks import MockSearch, MockSqlRagModel
from search_rl_env.contracts import EpisodeConfig, SqlQueryError
from search_rl_env.environment.demo import HYBRID_HITS, HYBRID_REFERENCE, hybrid_task
from search_rl_env.environment.sql import MockSql
from search_rl_env.environment.taskset import SearchTaskset, SearchTasksetConfig


async def test_sql_executes_different_queries_on_fixture():
    sql = MockSql()
    rows = await sql.query("SELECT order_id FROM orders WHERE delay_days > 3 AND region = 'RU' AND status = 'delivered' ORDER BY order_id")
    assert rows.columns == ['order_id']
    assert rows.rows == [['ORD-101'], ['ORD-105']]
    assert not rows.truncated
    count = await sql.query('SELECT count(*) AS total FROM orders')
    assert count.rows == [[6]]
    assert count.columns == ['total']


@pytest.mark.parametrize('query', [
    'DELETE FROM orders', "UPDATE orders SET region = 'RU'", 'DROP TABLE orders',
    "ATTACH DATABASE ':memory:' AS other", 'PRAGMA user_version = 1',
    'CREATE TABLE another (id INTEGER)', 'SELECT 1; SELECT 2',
])
async def test_sql_is_read_only_and_single_statement(query):
    with pytest.raises(SqlQueryError):
        await MockSql().query(query)


async def test_sql_errors_do_not_mutate_later_queries():
    sql = MockSql()
    with pytest.raises(SqlQueryError):
        await sql.query('SELECT missing FROM orders')
    assert (await sql.query('SELECT count(*) FROM orders')).rows == [[6]]


async def test_sql_result_size_and_compute_are_bounded():
    result = await MockSql(max_rows=2).query('SELECT order_id FROM orders ORDER BY order_id')
    assert result.truncated
    assert result.rows == [['ORD-101'], ['ORD-102']]
    with pytest.raises(SqlQueryError, match='interrupted'):
        await MockSql(max_vm_steps=1000).query('WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n')
    with pytest.raises(SqlQueryError):
        await MockSql().query('SELECT randomblob(2000000)')


async def test_hybrid_rollout_has_answer_sources_native_trace_and_reward():
    task = hybrid_task()
    search = MockSearch(HYBRID_HITS, corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)

    class ObservedModel(MockSqlRagModel):
        async def complete(self, messages, **kwargs):
            prompt = json.loads(messages[1]['content'])
            assert 'CREATE TABLE orders' in prompt['sql_schema']
            assert 'reference_answer' not in prompt
            assert 'ORD-101' not in messages[1]['content']
            return await super().complete(messages, **kwargs)

    result = await run_episode(task, ObservedModel(), search, EpisodeConfig(), sql=MockSql(), reference_answer=HYBRID_REFERENCE)
    assert result.stop_reason.value == 'finished'
    assert result.answer == HYBRID_REFERENCE
    assert result.selected_document_ids == ['policy-001']
    assert result.sql_calls == result.search_calls == 1
    assert result.reward == 1
    assert list(result.native_trace['rewards']) == ['answer_accuracy']
    assert [step.actions[0].name for step in result.trajectory] == ['search', 'sql', 'finish']
    assert result.trajectory[1].actions[0].result['rows'] == [['ORD-101'], ['ORD-105']]


async def test_bad_sql_can_be_corrected_without_ending_episode():
    class RepairModel(MockSqlRagModel):
        async def complete(self, messages, **kwargs):
            reply = await super().complete(messages, **kwargs)
            calls = [message for message in messages if message['role'] == 'tool']
            if len(calls) == 1:
                reply.message['tool_calls'][0]['function']['arguments'] = json.dumps({'query': 'SELECT missing FROM orders'})
            return reply

    task = hybrid_task()
    result = await run_episode(task, RepairModel(), MockSearch(HYBRID_HITS, corpus_id=task.corpus_id, corpus_revision='v1'), EpisodeConfig(), sql=MockSql())
    assert result.stop_reason.value == 'finished'
    assert result.answer == HYBRID_REFERENCE
    assert result.sql_calls == 2
    assert result.invalid_actions == 1
    assert result.trajectory[1].actions[0].result['error']['code'] == 'sql_query_error'
    assert result.reward is None


async def test_broken_sql_backend_is_infrastructure_error_and_redacted():
    class BrokenSql(MockSql):
        async def query(self, query):
            raise RuntimeError('credentials=do-not-expose')

    task = hybrid_task()
    result = await run_episode(task, MockSqlRagModel(), MockSearch(HYBRID_HITS, corpus_id=task.corpus_id, corpus_revision='v1'), EpisodeConfig(), sql=BrokenSql())
    assert result.stop_reason.value == 'infrastructure_error'
    assert result.error.code == 'sql_error'
    assert result.answer is None
    assert 'do-not-expose' not in result.model_dump_json()


def test_hybrid_native_taskset_uses_private_answer_and_public_schema():
    taskset = SearchTaskset(SearchTasksetConfig(scenario='sql_rag', require_references=True))
    task = next(iter(taskset.load()))
    assert task.data.reference_answer == HYBRID_REFERENCE
    assert 'reference_answer' not in task.data.prompt
    assert 'ORD-101' not in task.data.prompt
    assert task.config.tools.enable_sql
    assert {hook.__name__ for hook in task.hooks('reward')} == {'answer_accuracy'}


@pytest.mark.parametrize(('answer', 'reference', 'reward'), [
    (['ORD-102'], HYBRID_REFERENCE, 0.0),
    ([], [], 1.0),
])
async def test_answer_reward_checks_exact_set_including_empty_answer(answer, reference, reward):
    class FinishModel(MockSqlRagModel):
        async def complete(self, messages, **kwargs):
            response = await super().complete(messages, **kwargs)
            function = response.message['tool_calls'][0]['function']
            function['name'] = 'finish'
            function['arguments'] = json.dumps({'document_ids': [], 'answer': answer})
            return response

    task = hybrid_task()
    result = await run_episode(task, FinishModel(), MockSearch(HYBRID_HITS, corpus_id=task.corpus_id, corpus_revision='v1'), EpisodeConfig(), sql=MockSql(), reference_answer=reference)
    assert result.stop_reason.value == 'finished'
    assert result.answer == answer
    assert result.reward == reward
    assert result.sql_calls == result.search_calls == 0


def test_sql_result_rejects_invalid_row_shape():
    from pydantic import ValidationError
    from search_rl_env.contracts import SqlResult

    with pytest.raises(ValidationError):
        SqlResult(columns=['id'], rows=[['a', 'b']])

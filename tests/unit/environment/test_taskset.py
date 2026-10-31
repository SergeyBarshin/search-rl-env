import json

import pytest
import verifiers.v1 as vf

from search_rl_env import SearchHarness, SearchTaskset
from search_rl_env.agent import run_episode
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit
from search_rl_env.environment.rewards import recall_at_k
from search_rl_env.environment.taskset import SearchTasksetConfig


def test_native_plugins_and_private_task_data():
    assert issubclass(SearchHarness, vf.Harness)
    assert issubclass(SearchTaskset, vf.Taskset)
    task = next(iter(SearchTaskset(SearchTasksetConfig()).load()))
    assert isinstance(task, vf.Task)
    assert task.data.reference_document_ids
    assert 'reference_document_ids' not in task.data.prompt
    assert 'doc-001' not in task.data.prompt
    assert task.config.tools.backend_url is None


def test_training_taskset_requires_labels(tmp_path):
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps([{'task_id': 'missing', 'question': 'Срок?', 'language': 'ru', 'corpus_id': 'demo'}]))
    taskset = SearchTaskset(SearchTasksetConfig(tasks_path=path, require_references=True))
    with pytest.raises(ValueError, match='reference_document_ids'):
        list(taskset.load())


@pytest.mark.parametrize(('selected', 'gold', 'expected'), [
    ([], ['a'], 0.0),
    (['a', 'a', 'b'], ['a', 'b', 'b'], 1.0),
    (['a'], ['a', 'b'], 0.5),
    ([str(i) for i in range(11)], ['10'], 0.0),
    (['0'] * 12 + ['1'], ['0', '1'], 1.0),
])
def test_recall_deduplicates_before_top_ten(selected, gold, expected):
    assert recall_at_k(selected, gold) == expected


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k([], [])


async def test_native_reward_does_not_expose_reference_to_model():
    task = AgentTask(task_id='reward', question='Срок?', language='ru', corpus_id='demo')
    backend = MockSearch([SearchHit(doc_id='a', snippet='Ноябрь.', rank=1)], corpus_id='demo')

    class ObservedModel(MockModel):
        async def complete(self, messages, **kwargs):
            assert 'private-gold-id' not in json.dumps(messages)
            assert 'reference_document_ids' not in json.dumps(messages)
            return await super().complete(messages, **kwargs)

    result = await run_episode(task, ObservedModel(), backend, EpisodeConfig(), reference_document_ids=['a', 'private-gold-id'])
    assert result.reward == 0.5
    assert result.native_trace['rewards']['retrieval_recall']['score'] == 0.5


async def test_concurrent_episodes_keep_corpora_and_state_separate():
    import asyncio

    async def episode(corpus_id):
        task = AgentTask(task_id=corpus_id, question='Срок?', language='ru', corpus_id=corpus_id)
        search = MockSearch([SearchHit(doc_id=corpus_id + '-doc', snippet='Ноябрь.', rank=1)], corpus_id=corpus_id)
        return await run_episode(task, MockModel(), search, EpisodeConfig())

    first, second = await asyncio.gather(episode('first'), episode('second'))
    assert first.episode_id != second.episode_id
    assert first.selected_document_ids == ['first-doc']
    assert second.selected_document_ids == ['second-doc']
    assert first.search_calls == second.search_calls == 1

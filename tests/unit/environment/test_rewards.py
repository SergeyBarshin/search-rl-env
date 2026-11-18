import json

import pytest

from search_rl_env.agent import run_episode, save_episode
from search_rl_env.agent.mocks import MockModel, MockSearch
from search_rl_env.contracts import AgentTask, EpisodeConfig, SearchHit
from search_rl_env.environment.rewards import recall_at_k, score_episode


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


async def test_reward_does_not_expose_reference_to_model_or_public_artifact(tmp_path):
    task = AgentTask(task_id='reward', question='Срок?', language='ru', corpus_id='demo')
    backend = MockSearch([SearchHit(doc_id='a', snippet='Ноябрь.', rank=1)], corpus_id='demo')

    class ObservedModel(MockModel):
        async def complete(self, messages, **kwargs):
            assert 'private-gold-id' not in json.dumps(messages)
            assert 'reference_document_ids' not in json.dumps(messages)
            return await super().complete(messages, **kwargs)

    result = await run_episode(task, ObservedModel(), backend, EpisodeConfig(), reference_document_ids=['a', 'private-gold-id'])
    assert result.reward == 0.5
    saved = save_episode(result, tmp_path / 'episode.json').read_text()
    assert 'private-gold-id' not in saved
    assert 'reward' not in json.loads(saved)
    from search_rl_env.contracts import EpisodeResult
    assert score_episode(EpisodeResult.model_validate_json(saved), reference_document_ids=['a', 'private-gold-id']) == 0.5


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

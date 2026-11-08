import asyncio

from search_rl_env.agent import RolloutRunner
from search_rl_env.agent.mocks import MockModel
from search_rl_env.contracts import AgentTask, EpisodeConfig
from search_rl_env.search import Document, HarnessSearch, LocalSearch, build_snapshot


async def test_real_search_in_sixteen_mock_episodes(tmp_path):
    path = tmp_path / "corpus.sqlite"
    manifest = build_snapshot([
        Document("delivery", "Equipment delivery postponed until November."),
        Document("library", "The library opens on Monday."),
    ], path, corpus_id="integration")
    async with LocalSearch(path) as backend:
        async with RolloutRunner(MockModel(), HarnessSearch(backend), EpisodeConfig(max_steps=4), concurrency=16) as runner:
            async def episode(index):
                task = AgentTask(task_id=f"task-{index}", question="delivery", language="en",
                                 corpus_id="integration", corpus_revision=manifest["corpus_revision"])
                result = await runner.run(task)
                assert result.stop_reason == "finished", result.error
                assert result.selected_document_ids == ["delivery"]
                assert result.search_calls == 1
                return result.episode_id
            ids = await asyncio.gather(*(episode(index) for index in range(16)))
            assert len(set(ids)) == 16

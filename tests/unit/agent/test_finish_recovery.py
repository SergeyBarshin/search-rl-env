"""Восстановление протокола при отсутствующем finish."""

from copy import deepcopy

import pytest

from search_rl_env.agent import run_episode
from search_rl_env.agent.mocks import MockSearch
from search_rl_env.contracts import EpisodeConfig
from search_rl_env.contracts.types import ModelResponse, Usage
from search_rl_env.environment.demo import hybrid_task
from search_rl_env.environment.sql import MockSql


class ScriptedModel:
    metadata = {"model": "finish-recovery-test"}

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, messages, *, tools, **kwargs):
        self.requests.append({"messages": deepcopy(messages), "tools": tools})
        message = next(self.replies)
        return ModelResponse(
            message=message, usage=Usage(prompt_tokens=0, completion_tokens=1, total_tokens=1),
            finish_reason="tool_calls" if message.get("tool_calls") else "stop",
        )


async def run_recovery(replies, retries):
    task = hybrid_task()
    model = ScriptedModel(replies)
    search = MockSearch([], corpus_id=task.corpus_id, corpus_revision=task.corpus_revision)
    result = await run_episode(task, model, search, EpisodeConfig(max_finish_retries=retries), sql=MockSql())
    return model.requests, result


@pytest.mark.parametrize("content", [None, "Here is my answer."])
async def test_missing_finish_reminder_preserves_original_response_and_tools(content):
    finish = {"role": "assistant", "tool_calls": [{"id": "done", "type": "function",
        "function": {"name": "finish", "arguments": '{"answer":["supported"],"document_ids":[]}'}}]}
    requests, result = await run_recovery([{"role": "assistant", "content": content}, finish], retries=2)
    assert len(requests) == 2
    history = requests[1]["messages"]
    assert history[-2] == {"role": "assistant", "content": content}
    assert history[-1]["role"] == "user"
    assert "No final answer has been submitted" in history[-1]["content"]
    assert requests[0]["tools"] == requests[1]["tools"]
    assert result.answer == ["supported"]
    assert result.trajectory[-1].actions[0].tool_call_id == "done"


@pytest.mark.parametrize("retries", [0, 2])
async def test_empty_responses_stop_after_bounded_recovery_without_fabricated_finish(retries):
    requests, result = await run_recovery([{"role": "assistant", "content": None}] * (1 + retries), retries)
    assert len(requests) == 1 + retries
    assert result.error.code == "missing_finish"
    assert all(not step.actions for step in result.trajectory)

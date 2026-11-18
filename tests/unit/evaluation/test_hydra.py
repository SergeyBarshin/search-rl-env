import json
from datetime import datetime, timezone

import pytest

from search_rl_env.contracts.types import AgentTask, EpisodeConfig, EpisodeResult, ModelResponse, StopReason, Usage
from search_rl_env.evaluation.hydra import judge_episode, summarize_verdicts


def episode(task_id, stop=StopReason.FINISHED, answer=("candidate",)):
    return EpisodeResult(
        schema_version="4", task_id=task_id, episode_id=task_id,
        task=AgentTask(task_id=task_id, question="question", language="en", corpus_id="wiki"),
        config=EpisodeConfig(), model_metadata={}, selected_document_ids=[], stop_reason=stop,
        truncated=False, history=[], trajectory=[], usage=Usage(), search_calls=0, invalid_actions=0,
        started_at=datetime.now(timezone.utc), duration_seconds=1, answer=list(answer), sql_enabled=True,
    )


@pytest.mark.parametrize(("reply", "score"), [('{"score":1}', 1), ('{"score":0}', 0), ('{"score":true}', None)])
async def test_judge_sees_only_question_candidate_and_reference(reply, score):
    class Model:
        async def complete(self, messages, *, tools, **kwargs):
            assert tools == []
            assert json.loads(messages[1]["content"]) == {
                "question": "question", "candidate": ["candidate"], "reference": "private",
            }
            return ModelResponse(message={"content": reply}, usage=Usage(total_tokens=9, cost=0.0))

    verdict = await judge_episode(episode("scored"), "private", Model())
    assert verdict["score"] == score
    assert verdict["judge_called"]
    assert "private" not in json.dumps(verdict)
    assert verdict["usage"]["total_tokens"] == 9
    assert verdict["usage"]["cost"] == 0


async def test_missing_answer_and_infrastructure_failure_have_distinct_coverage():
    no_answer = await judge_episode(episode("limit", StopReason.MAX_STEPS, ()), "", None)
    failed = await judge_episode(episode("failed", StopReason.INFRASTRUCTURE_ERROR, ()), "", None)
    summary = summarize_verdicts(["limit", "failed"], [no_answer, failed])
    assert summary["accuracy"] is None
    assert summary["coverage"] == 0.5
    assert summary["accuracy_bounds"] == [0, 0.5]
    assert summary["judge_calls"] == 0
    assert summarize_verdicts(["limit"], [no_answer])["accuracy"] == 0


def test_repeat_judgement_fills_missing_score_and_counts_both_calls():
    invalid = {"task_id": "a", "episode_id": "first", "judge_called": True,
               "status": "invalid_verdict", "score": None, "duration_seconds": 0.1, "usage": {"cost": 0.01}}
    retry = {**invalid, "status": "scored", "score": 1}
    summary = summarize_verdicts(["a"], [invalid], retries=[retry])
    assert summary["accuracy"] == 1
    assert summary["judge_calls"] == 2
    assert summary["judge_usage"]["cost"] == 0.02
    with pytest.raises(ValueError):
        summarize_verdicts(["a"], [retry], retries=[retry])

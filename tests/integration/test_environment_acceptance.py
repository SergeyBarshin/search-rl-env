import asyncio
import json
from pathlib import Path

import pytest

from search_rl_env.search.acceptance import run_acceptance

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def acceptance_config(tmp_path):
    pytest.importorskip("faiss", reason="Установите .[environments] для сквозной приёмки.")
    pytest.importorskip("psutil")
    config = json.loads((ROOT / "configs/environments/acceptance.json").read_text())
    config["datasets"] = [str(ROOT / "examples/fixtures/environments" / name / "manifest.json") for name in ["alpha", "beta"]]
    config["tasks_path"] = str(ROOT / "examples/fixtures/environments/tasks.json")
    config["episodes"], config["concurrency"], config["mock_latency_seconds"] = 40, [40], 0.
    config["warmup_episodes"] = 0
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(config))
    return path


async def test_forty_actual_hybrid_episodes_have_private_tools_and_saved_revisions(acceptance_config, tmp_path):
    output = tmp_path / "run"
    report = await run_acceptance(acceptance_config, output)
    assert report["status"] == "passed" and report["resources_released"]
    batch = report["batches"][0]
    assert batch["peak_active_episodes"] == batch["peak_active_environments"] == 40
    assert batch["first_wave_model_arrivals"] == 40
    assert batch["sql"]["count"] == batch["search"]["count"] == 40
    assert batch["memory"]["peak_rss_bytes"] > 0
    assert len(report["episodes"]) == 40
    assert len({row["environment"]["environment_id"] for row in report["episodes"]}) == 40
    assert {row["environment"]["dataset_id"] for row in report["episodes"]} == {"alpha", "beta"}
    for binding in report["episodes"]:
        result = json.loads((output / binding["episode"]).read_text())
        environment = binding["environment"]
        assert result["task"]["corpus_revision"] == environment["search"]["corpus_revision"]
        assert result["sql_calls"] == result["search_calls"] == 1
        assert result["reward"] == 1
        expected = "ALPHA-17" if environment["dataset_id"] == "alpha" else "BETA-29"
        assert result["answer"] == [expected]
        other = "betakey" if environment["dataset_id"] == "alpha" else "alphakey"
        assert other not in json.dumps(result["history"], ensure_ascii=False)


async def test_cancellation_saves_partial_episodes_and_closes_all_environments(acceptance_config, tmp_path):
    config = json.loads(acceptance_config.read_text())
    config["episodes"], config["concurrency"] = 2, [2]
    acceptance_config.write_text(json.dumps(config))
    entered, count = asyncio.Event(), 0

    class BlockedModel:
        metadata = {"model": "cancel-probe"}

        async def complete(self, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                entered.set()
            await asyncio.Event().wait()

    output = tmp_path / "interrupted"
    operation = asyncio.create_task(run_acceptance(acceptance_config, output, model=BlockedModel()))
    try:
        await asyncio.wait_for(entered.wait(), 30)
    finally:
        operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    report = json.loads((output / "manifest.json").read_text())
    assert report["status"] == "cancelled" and report["resources_released"]
    assert len(report["episodes"]) == 2
    for binding in report["episodes"]:
        partial = json.loads((output / binding["episode"]).read_text())
        assert partial["error"]["code"] == "interrupted"

import asyncio
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from search_rl_env.contracts.types import ModelError, ModelResponse, Usage
from search_rl_env.search.acceptance import AcceptanceConfig, ReservedBudgetModel, StartGate, distribution


def test_config_and_percentiles():
    base = {"version": 1, "datasets": ["example.json"], "tasks_path": "tasks.json"}
    for invalid in [[0], [1, 1]]:
        with pytest.raises(ValidationError):
            AcceptanceConfig.model_validate({**base, "concurrency": invalid})
    assert distribution([3, 1, 2])["p95_seconds"] == 3
    assert distribution([])["p50_seconds"] is None


async def test_start_gate_waits_for_all_real_model_calls():
    gate = StartGate(3, 1)
    first = asyncio.create_task(gate.arrive())
    second = asyncio.create_task(gate.arrive())
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    await gate.arrive()
    await asyncio.gather(first, second)
    await gate.arrive()
    assert gate.arrivals == 3


class CostModel:
    metadata = {"model": "cost-probe"}

    def __init__(self, cost):
        self.cost, self.calls = cost, 0
        self.release = asyncio.Event()

    async def complete(self, *args, **kwargs):
        self.calls += 1
        await self.release.wait()
        return ModelResponse(message={"role": "assistant", "content": "ok"}, usage=Usage(cost=self.cost))


async def test_budget_reserves_before_dispatch_and_releases_known_cost():
    delegate = CostModel(.2)
    model = ReservedBudgetModel(delegate, .5, .5)
    first = asyncio.create_task(model.complete())
    await asyncio.sleep(0)
    with pytest.raises(ModelError):
        await model.complete()
    assert delegate.calls == 1
    delegate.release.set()
    await first
    assert model.spent == .2 and model.reserved == 0


@pytest.mark.parametrize("cost", [None, .6])
async def test_budget_blocks_unknown_or_underestimated_cost(cost):
    delegate = CostModel(cost)
    delegate.release.set()
    model = ReservedBudgetModel(delegate, 1., .5)
    with pytest.raises(ModelError):
        await model.complete()
    with pytest.raises(ModelError):
        await model.complete()
    assert delegate.calls == 1 and model.blocked


def test_cli_requires_explicit_paid_flag_before_loading_credentials():
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run([sys.executable, str(root / "examples/accept_environments.py"),
        "--openrouter-model", "unused-test-model", "--budget-usd", "1", "--request-reserve-usd", ".1"],
        cwd=root, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 2
    assert "--allow-paid" in completed.stderr

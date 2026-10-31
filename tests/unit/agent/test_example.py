import asyncio
import json
import signal
import subprocess
import sys
from pathlib import Path


def test_mock_example_saves_finished_episode(tmp_path):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "episode.json"
    completed = subprocess.run(
        [sys.executable, str(root / "examples/run_episode.py"), "--output", str(output),
         "--question", "Какой новый срок проекта?", "--verbose"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "stop_reason=finished" in completed.stdout
    result = json.loads(output.read_text())
    assert result["selected_document_ids"] == ["doc-001", "doc-002"]
    assert result["search_calls"] == 1
    assert result["task"]["question"] == "Какой новый срок проекта?"
    assert "search(" in completed.stdout
    assert "finish(" in completed.stdout
    assert "Usage:" in completed.stdout
    assert result['profile']['wall_seconds'] > 0
    assert output.with_suffix('.events.jsonl').is_file()


async def test_ctrl_c_saves_partial_episode_without_traceback(tmp_path):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "episode.json"
    output.write_text('{"existing": true}')
    script = f"""
import asyncio
import runpy
import sys

namespace = runpy.run_path({str(root / 'examples/run_episode.py')!r})
async def wait_forever(self, *args, **kwargs):
    print('MODEL_WAITING', flush=True)
    await asyncio.Event().wait()
namespace['MockModel'].complete = wait_forever
sys.argv = ['run_episode.py', '--model', 'mock', '--verbose', '--output', {str(output)!r}]
raise SystemExit(asyncio.run(namespace['main']()))
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    before_interrupt = []
    try:
        async with asyncio.timeout(15):
            while True:
                line = await process.stdout.readline()
                assert line, "CLI exited before reaching the pending model request"
                before_interrupt.append(line.decode())
                if "MODEL_WAITING" in before_interrupt[-1]:
                    break
        assert any("Шаг 1: отправляю запрос" in line for line in before_interrupt)
        process.send_signal(signal.SIGINT)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 130
    assert "Traceback" not in stderr.decode()
    assert "Прервано" in stdout.decode()
    assert json.loads(output.read_text()) == {"existing": True}
    partial = json.loads((tmp_path / "episode.interrupted.json").read_text())
    assert partial["error"]["code"] == "interrupted"
    assert partial["usage"]["completion_tokens"] is None
    assert len(partial["trajectory"]) == 1


def test_cli_rejects_invalid_request_timeout(tmp_path):
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(root / "examples/run_episode.py"), "--request-timeout", "nan"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "положительное" in result.stderr


def test_parallel_example_saves_results_and_summary(tmp_path):
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [sys.executable, str(root / 'examples/run_rollouts.py'), '--episodes', '8', '--concurrency', '3',
         '--output', str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=20,
    )
    summary = json.loads(completed.stdout)
    assert summary == json.loads((tmp_path / 'summary.json').read_text())
    assert summary['episodes'] == 8 and summary['concurrency'] == 3
    assert summary['mean_reward'] == 1
    episodes = [json.loads(path.read_text()) for path in tmp_path.glob('rollout-*.json')]
    assert len({result['episode_id'] for result in episodes}) == 8
    assert all(result['answer'] == ['ORD-101', 'ORD-105'] for result in episodes)
    assert all(len(result['native_trace']['calls']) == 3 for result in episodes)
    assert "Traceback" not in completed.stderr

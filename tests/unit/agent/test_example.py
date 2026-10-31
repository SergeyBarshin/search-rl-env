import json
import subprocess
import sys
from pathlib import Path


def test_mock_example_saves_finished_episode(tmp_path):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "episode.json"
    completed = subprocess.run(
        [sys.executable, str(root / "examples/run_episode.py"), "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "stop_reason=finished" in completed.stdout
    result = json.loads(output.read_text())
    assert result["selected_document_ids"] == ["doc-001", "doc-002"]
    assert result["search_calls"] == 1

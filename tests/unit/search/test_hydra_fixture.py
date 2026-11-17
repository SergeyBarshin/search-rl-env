import hashlib
import json
from pathlib import Path

from search_rl_env.contracts import AgentTask
from search_rl_env.search.semantic import fuse_rankings

FIXTURE = Path(__file__).resolve().parents[3] / "examples/fixtures/hydra"


def test_real_corpus_provenance_and_private_reference_are_separate():
    task = AgentTask.model_validate_json((FIXTURE / "task.json").read_text())
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    data = (FIXTURE / "corpus.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == provenance["corpus"]["sha256"]
    rows = [json.loads(line) for line in data.splitlines()]
    assert len(rows) == 999 == provenance["corpus"]["documents"]
    ids = {row["doc_id"] for row in rows}
    assert len(ids) == len(rows)
    assert set(provenance["manual_reference_doc_ids"]) <= ids
    assert all(set(row) == {"doc_id", "text", "title", "language"} for row in rows)
    assert task.task_id == "train_a2s_4"
    assert "answer" not in task.model_dump()
    assert all(provenance["corpus"]["row_to_wikipedia_id"][identifier] == "237029"
               for identifier in provenance["manual_reference_doc_ids"])


def test_fusion_combines_ranks_without_comparing_bm25_and_cosine_scores():
    bm25 = [{"doc_id": "a", "score": -100, "snippet": "a"}, {"doc_id": "b", "score": -1, "snippet": "b"}]
    dense = [{"doc_id": "b", "score": 0.9, "snippet": "b"}, {"doc_id": "c", "score": 0.8, "snippet": "c"}]
    result = fuse_rankings(bm25, dense)
    assert [hit["doc_id"] for hit in result] == ["b", "a", "c"]
    assert [hit["rank"] for hit in result] == [1, 2, 3]

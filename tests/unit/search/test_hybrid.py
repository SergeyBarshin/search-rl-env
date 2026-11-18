import json
from shutil import copyfile

import pytest

from search_rl_env.search.models import UnknownId
from search_rl_env.search.server import SearchStore, build_snapshot


async def test_hybrid_uses_both_indexes_and_keeps_episode_scope(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    from search_rl_env.search.hybrid import HybridIndex
    from search_rl_env.search.local import FaissSearch

    corpus = tmp_path / "corpus.jsonl"
    passage = "Hockey arena " + "description " * 50 + "last detail"
    corpus.write_text(json.dumps({"doc_id": "one", "title": "Example", "text": passage}) + "\n")
    snapshot = tmp_path / "search.sqlite"
    build_snapshot(corpus, snapshot, "hockey")
    template = HybridIndex(SearchStore(snapshot))
    episode_path = tmp_path / "episode.sqlite"
    copyfile(snapshot, episode_path)
    search = template.open(SearchStore(episode_path))
    other = template.open(SearchStore(snapshot))
    calls = []
    bm25_search, faiss_search = SearchStore.search, FaissSearch.search

    def bm25(self, request):
        calls.append("bm25")
        return bm25_search(self, request)

    async def faiss(self, *args, **kwargs):
        calls.append("faiss")
        return await faiss_search(self, *args, **kwargs)

    monkeypatch.setattr(SearchStore, "search", bm25)
    monkeypatch.setattr(FaissSearch, "search", faiss)
    scope = {"corpus_id": "hockey", "corpus_revision": template.manifest["corpus_revision"], "limit": 2}
    try:
        hits = await search.search("hockey", **scope)
        assert calls == ["bm25", "faiss"]
        assert [(hit.doc_id, hit.rank) for hit in hits] == [("one", 1)]
        assert hits[0].snippet == "Example\n" + passage
        with pytest.raises(UnknownId):
            await search.search("hockey", **{**scope, "corpus_id": "other"})
    finally:
        await search.aclose()
    try:
        assert await other.search("hockey", **scope) == hits
    finally:
        await other.aclose()

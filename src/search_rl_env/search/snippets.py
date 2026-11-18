"""Ограничение текста поискового наблюдения токенизатором модели."""

from pathlib import Path

from search_rl_env.contracts.types import SearchBackend, SearchHit

from .local import digest_file


class SnippetTokenSearch:
    def __init__(self, search: SearchBackend, tokenizer_path: Path, limit: int):
        from tokenizers import Tokenizer

        if limit < 1:
            raise ValueError("Лимит токенов сниппета должен быть положительным.")
        self._search = search
        self._limit = limit
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        self._tokenizer_sha256 = digest_file(tokenizer_path)

    @property
    def observation(self) -> dict:
        return {"max_snippet_tokens": self._limit, "tokenizer_sha256": self._tokenizer_sha256}

    async def search(self, query: str, *, corpus_id: str, corpus_revision: str | None, limit: int) -> list[SearchHit]:
        hits = await self._search.search(query, corpus_id=corpus_id, corpus_revision=corpus_revision, limit=limit)
        return [hit.model_copy(update={"snippet": self._prefix(hit.snippet)}) for hit in hits]

    def _prefix(self, snippet: str) -> str:
        encoding = self._tokenizer.encode(snippet, add_special_tokens=False)
        while len(encoding.ids) > self._limit:
            # Несколько токенов могут соответствовать одному символу Unicode.
            end = min(encoding.offsets[self._limit][0], len(snippet) - 1)
            snippet = snippet[:end].rstrip()
            encoding = self._tokenizer.encode(snippet, add_special_tokens=False)
        return snippet

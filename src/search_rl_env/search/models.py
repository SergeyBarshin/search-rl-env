"""Локальные типы поиска; не меняют контракт инструментов агента."""

from dataclasses import dataclass


class SearchError(Exception):
    """Ошибка с безопасным сообщением без исходного запроса и путей."""


class InvalidRequest(SearchError):
    pass


class UnknownId(SearchError):
    pass


class SnapshotConflict(SearchError):
    pass


class StorageError(SearchError):
    pass


class SearchTimeout(SearchError):
    pass


class SearchBusy(SearchError):
    pass


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    language: str = "und"


@dataclass(frozen=True)
class Match:
    doc_id: str
    snippet: str
    rank: int
    score: float


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[Match, ...]
    corpus_revision: str
    index_revision: str
    search_revision: str


@dataclass(frozen=True)
class TextResult:
    doc_id: str
    chunk_id: str | None
    text: str
    offset: int
    next_offset: int | None
    total_chars: int
    corpus_revision: str


@dataclass(frozen=True)
class SearchConfig:
    max_results: int = 100
    snippet_chars: int = 500
    max_read_chars: int = 8000
    max_query_chars: int = 2000
    max_query_terms: int = 32
    workers: int = 4
    max_pending: int = 32
    timeout_seconds: float = 5.0

    def __post_init__(self):
        import math

        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "timeout_seconds":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("timeout_seconds должен быть конечным положительным числом.")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом.")

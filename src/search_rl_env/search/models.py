"""HTTP-контракт Go-сервиса; индексация и SQLite находятся в Go."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from search_rl_env.contracts import SqlResult


class SearchError(Exception):
    """Безопасное сообщение без URL, тела ответа и текста запроса."""


class InvalidRequest(SearchError):
    pass


class UnknownId(SearchError):
    pass


class SearchBusy(SearchError):
    pass


class SearchTimeout(SearchError):
    pass


class StorageError(SearchError):
    pass


class SearchProtocolError(SearchError):
    pass


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ServiceConfig(WireModel):
    max_results: int = Field(ge=1)
    snippet_chars: int = Field(ge=1)
    max_read_chars: int = Field(ge=1)
    max_query_chars: int = Field(ge=1)
    max_query_terms: int = Field(ge=1)
    workers: int = Field(ge=1)
    max_pending: int = Field(ge=1)
    timeout_nanoseconds: int = Field(ge=1)


class Manifest(WireModel):
    schema_version: Literal[2]
    corpus_id: str = Field(min_length=1)
    corpus_revision: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_chars: int = Field(ge=1)
    chunk_algorithm: Literal["chars-v1"]
    documents: int = Field(ge=1)
    index_algorithm: Literal["go-fts5-unicode61-document-bm25-v1"]
    sqlite_version: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    index_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    api_version: Literal["1"]
    search_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: ServiceConfig


class Versions(WireModel):
    corpus_id: str = Field(min_length=1)
    corpus_revision: str = Field(min_length=1)
    index_revision: str = Field(min_length=1)
    search_revision: str = Field(min_length=1)


class Match(WireModel):
    doc_id: str = Field(min_length=1)
    snippet: str
    rank: int = Field(ge=1)
    score: float


class SearchResult(Versions):
    hits: list[Match]


class TextResult(Versions):
    doc_id: str = Field(min_length=1)
    chunk_id: str | None
    text: str
    offset: int = Field(ge=0)
    next_offset: int | None = Field(ge=0)
    total_chars: int = Field(ge=0)


class ChunksResult(Versions):
    chunk_ids: list[str]
    next_offset: int | None = Field(ge=0)


class ErrorBody(WireModel):
    code: str
    message: str


class ErrorEnvelope(WireModel):
    error: ErrorBody


class SQLInfo(WireModel):
    db_id: str = Field(min_length=1)
    db_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_: str = Field(alias="schema", min_length=1)


class SQLResponse(SqlResult):
    db_id: str = Field(min_length=1)
    db_revision: str = Field(pattern=r"^[0-9a-f]{64}$")

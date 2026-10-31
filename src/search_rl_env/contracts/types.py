"""Контракты первого retrieval-эпизода без эталонных данных."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentTask(Contract):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    corpus_revision: str | None = None


class SearchHit(Contract):
    doc_id: str = Field(min_length=1)
    snippet: str
    rank: int = Field(ge=1)


class SearchBackend(Protocol):
    async def search(
        self,
        query: str,
        *,
        corpus_id: str,
        corpus_revision: str | None,
        limit: int,
    ) -> list[SearchHit]: ...


class EpisodeConfig(Contract):
    max_steps: int = Field(default=6, ge=1)
    max_searches: int | None = Field(default=None, ge=0)
    max_output_tokens: int = Field(default=2048, ge=1)
    max_tokens_per_step: int = Field(default=512, ge=1)
    max_invalid_actions: int = Field(default=2, ge=1)
    search_limit: int = Field(default=10, ge=1)
    max_snippet_chars: int = Field(default=2000, ge=1)
    max_documents: int = Field(default=10, ge=1)
    seed: int | None = 0


class Usage(BaseModel):
    # Сохраняем дополнительные поля провайдера, включая reasoning и cached tokens.
    model_config = ConfigDict(extra="allow", strict=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    def add(self, other: Usage) -> Usage:
        totals = {}
        for field in type(self).model_fields:
            left, right = getattr(self, field), getattr(other, field)
            totals[field] = None if left is None or right is None else left + right
        return Usage(**totals)


class ModelResponse(Contract):
    message: dict[str, Any]
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    response_id: str | None = None
    model: str | None = None
    provider: str | None = None


class ChatModel(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        seed: int | None,
    ) -> ModelResponse: ...


class StopReason(str, Enum):
    FINISHED = "finished"
    MAX_STEPS = "max_steps"
    MAX_SEARCHES = "max_searches"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    INVALID_ACTION = "invalid_action"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class EpisodeError(Contract):
    code: str
    message: str


class ToolExecution(Contract):
    tool_call_id: str
    name: str
    arguments: str
    result: dict[str, Any]
    duration_seconds: float = Field(ge=0)


class EpisodeStep(Contract):
    step: int
    request_max_output_tokens: int
    model_duration_seconds: float = Field(default=0, ge=0)
    usage: Usage = Field(default_factory=Usage)
    response_id: str | None = None
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    actions: list[ToolExecution] = Field(default_factory=list)
    error: EpisodeError | None = None


class EpisodeResult(Contract):
    schema_version: Literal["1"] = "1"
    task_id: str
    episode_id: str
    task: AgentTask
    config: EpisodeConfig
    model_metadata: dict[str, Any]
    selected_document_ids: list[str]
    stop_reason: StopReason
    truncated: bool
    history: list[dict[str, Any]]
    trajectory: list[EpisodeStep]
    usage: Usage
    search_calls: int
    invalid_actions: int
    started_at: datetime
    duration_seconds: float = Field(ge=0)
    error: EpisodeError | None = None

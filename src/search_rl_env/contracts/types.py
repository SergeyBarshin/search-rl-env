"""Контракты первого retrieval-эпизода без эталонных данных."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class SqlResult(Contract):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    columns: list[str]
    rows: list[list[str | int | float | None]]
    truncated: bool = False

    @model_validator(mode="after")
    def matching_columns(self):
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("Число значений в строке должно совпадать с числом колонок.")
        return self


class SqlQueryError(ValueError):
    """Ошибка запроса с безопасным для наблюдения сообщением."""


class SqlBackend(Protocol):
    @property
    def schema(self) -> str: ...

    async def query(self, query: str) -> SqlResult: ...


class EpisodeConfig(Contract):
    max_steps: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tokens_per_step: int = Field(default=512, ge=1)
    search_limit: int = Field(default=10, ge=1)
    max_snippet_chars: int = Field(default=2000, ge=1)
    seed: int | None = None


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


class HttpPhase(Contract):
    name: str
    offset_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    status: Literal["running", "complete", "failed", "cancelled"]
    error_type: str | None = None


class HttpProfile(Contract):
    request_id: str
    duration_seconds: float = Field(ge=0)
    status: Literal["running", "complete", "failed", "cancelled"] = "running"
    time_to_headers_seconds: float | None = None
    time_to_first_body_byte_seconds: float | None = None
    request_prepare_seconds: float | None = None
    time_to_transport_seconds: float | None = None
    body_read_seconds: float | None = None
    decode_seconds: float | None = None
    status_code: int | None = None
    http_version: str | None = None
    connection_reused: bool | None = None
    request_bytes: int | None = None
    response_bytes: int = 0
    provider_request_id: str | None = None
    provider_timing_seconds: dict[str, float] = Field(default_factory=dict)
    error_code: str | None = None
    phases: list[HttpPhase] = Field(default_factory=list)


class EpisodeProfile(Contract):
    version: Literal['1'] = '1'
    wall_seconds: float = Field(ge=0)
    setup_seconds: float = Field(ge=0)
    model_seconds: float = Field(ge=0)
    tools_seconds: float = Field(ge=0)
    finalize_seconds: float = Field(ge=0)
    scoring_seconds: float = Field(ge=0)
    other_seconds: float = Field(ge=0)
    framework_seconds: dict[str, float] = Field(default_factory=dict)


class ModelResponse(Contract):
    message: dict[str, Any]
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    response_id: str | None = None
    model: str | None = None
    provider: str | None = None
    http_profile: HttpProfile | None = None


class ModelError(RuntimeError):
    """Безопасное для журнала сообщение об ошибке адаптера модели."""

    code = "model_error"
    http_profile: HttpProfile | None = None


class ModelTimeoutError(ModelError):
    code = "model_timeout"


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
    http_profile: HttpProfile | None = None


class EpisodeEvent(Contract):
    kind: Literal["model_started", "model_finished", "tool_started", "tool_finished",
                  "episode_started", "episode_phase", "episode_finished", "model_progress"]
    step: int
    details: dict[str, Any]
    episode_id: str | None = None
    elapsed_seconds: float | None = None
    timestamp: datetime | None = None


class EpisodeResult(Contract):
    schema_version: Literal["2", "3"] = "3"
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
    reward: float | None = None
    native_trace: dict[str, Any] | None = None
    answer: list[str] | None = None
    sql_calls: int = 0
    profile: EpisodeProfile | None = None

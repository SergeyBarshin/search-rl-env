from pydantic import Field
from verifiers.v1 import State

from search_rl_env.contracts.types import EpisodeError, StopReason, ToolExecution


class SearchState(State):
    discovered_ids: set[str] = Field(default_factory=set)
    selected_ids: list[str] = Field(default_factory=list)
    searches: int = 0
    sql_calls: int = 0
    answer: list[str] | None = None
    invalid_actions: int = 0
    stop_reason: StopReason | None = None
    error: EpisodeError | None = None
    pending_calls: list[dict] = Field(default_factory=list)
    actions: list[ToolExecution] = Field(default_factory=list)

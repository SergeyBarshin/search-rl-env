"""Состояние инструментов одного эпизода."""

from dataclasses import dataclass, field

from search_rl_env.contracts.types import EpisodeError, StopReason


@dataclass
class SearchState:
    discovered_ids: set[str] = field(default_factory=set)
    selected_ids: list[str] = field(default_factory=list)
    searches: int = 0
    sql_calls: int = 0
    answer: list[str] | None = None
    invalid_actions: int = 0
    stop_reason: StopReason | None = None
    error: EpisodeError | None = None

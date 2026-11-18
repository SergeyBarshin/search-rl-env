"""Публичные точки запуска и сохранения эпизода."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from search_rl_env.contracts.types import (
    AgentTask, ChatModel, EpisodeConfig, EpisodeEvent, EpisodeResult, SearchBackend, SqlBackend,
)


async def run_episode(
    task: AgentTask,
    model: ChatModel,
    search: SearchBackend,
    config: EpisodeConfig,
    *,
    on_event: Callable[[EpisodeEvent], None] | None = None,
    on_interrupted: Callable[[EpisodeResult], None] | None = None,
    reference_document_ids: list[str] | None = None,
    sql: SqlBackend | None = None,
    reference_answer: list[str] | None = None,
) -> EpisodeResult:
    """Одиночный запуск. Для серии эпизодов переиспользуйте RolloutRunner."""
    from .runner import RolloutRunner
    async with RolloutRunner(model, search, config, sql=sql, concurrency=1) as runner:
        return await runner.run(task, on_event=on_event, on_interrupted=on_interrupted,
            reference_document_ids=reference_document_ids, reference_answer=reference_answer)


def save_episode(result: EpisodeResult, path: str | Path) -> Path:
    """Атомарно записать один результат; runner сам выбирает путь и возобновление."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = stream.name
            stream.write(result.model_dump_json(indent=2, exclude={"reward"} if result.schema_version == "4" else None) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return path

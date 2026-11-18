"""Детерминированная оценка сохранённого ответа для локальных задач."""

from search_rl_env.contracts.types import EpisodeResult, StopReason


def recall_at_k(document_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Для Recall нужно непустое множество эталонных документов.")
    selected = set(list(dict.fromkeys(document_ids))[:k])
    return len(selected & relevant) / len(relevant)


def score_episode(
    result: EpisodeResult, *, reference_document_ids: list[str] | None = None,
    reference_answer: list[str] | None = None,
) -> float | None:
    if result.stop_reason == StopReason.INFRASTRUCTURE_ERROR:
        return None
    legacy_sql = (result.native_trace or {}).get("task", {}).get("data", {}).get("sql_schema")
    if result.sql_enabled or legacy_sql is not None:
        if reference_answer is None:
            return None
        return float(result.answer is not None and set(result.answer) == set(reference_answer))
    if reference_document_ids is None:
        return None
    return recall_at_k(result.selected_document_ids, reference_document_ids)

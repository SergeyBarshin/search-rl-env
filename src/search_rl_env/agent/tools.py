"""Схемы инструментов и проверка аргументов до обращения к бэкенду."""

from typing import Literal

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from search_rl_env.contracts.types import (
    AgentTask,
    Contract,
    EpisodeError,
    SearchBackend,
    SearchHit,
    StopReason,
)


class SearchArguments(Contract):
    query: str = Field(min_length=1)


class FinishArguments(Contract):
    document_ids: list[str]


class FunctionCall(Contract):
    name: str = Field(min_length=1)
    arguments: str


class ToolCall(Contract):
    # Например, OpenRouter может вернуть index и в нестриминговом ответе.
    # Эти метаданные сохраняются в history, но не меняют проверку аргументов.
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str = Field(min_length=1)
    type: Literal["function"]
    function: FunctionCall


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": arguments.model_json_schema(),
            },
        }
        for name, description, arguments in [
            ("search", "Найти документы по запросу в корпусе задачи.", SearchArguments),
            ("finish", "Завершить с ранжированными id показанных документов.", FinishArguments),
        ]
    ]


def tool_error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def invalid_action(state, code: str, message: str) -> dict:
    state.invalid_actions += 1
    return tool_error(code, message)


async def execute_tool(
    call: ToolCall,
    *,
    task: AgentTask,
    search: SearchBackend,
    config,
    state,
) -> dict:
    if state.stop_reason is not None:
        return tool_error("episode_stopped", "Эпизод уже остановлен; действие не выполнено.")

    name, arguments = call.function.name, call.function.arguments
    if name not in {"search", "finish"}:
        return invalid_action(state, "unknown_tool", "Доступны только search и finish.")
    try:
        if name == "search":
            parsed = SearchArguments.model_validate_json(arguments)
            if not parsed.query.strip():
                return invalid_action(state, "invalid_arguments", "Запрос не может быть пустым.")
        else:
            parsed = FinishArguments.model_validate_json(arguments)
    except ValidationError:
        return invalid_action(state, "invalid_arguments", f"Аргументы {name} не соответствуют схеме.")

    if isinstance(parsed, FinishArguments):
        document_ids = list(dict.fromkeys(parsed.document_ids))
        if any(doc_id not in state.discovered_ids for doc_id in document_ids):
            return invalid_action(state, "unknown_document", "В finish разрешены только показанные id.")
        state.selected_ids = document_ids
        state.stop_reason = StopReason.FINISHED
        return {"ok": True, "document_ids": document_ids}

    state.searches += 1
    try:
        raw_hits = await search.search(
            parsed.query,
            corpus_id=task.corpus_id,
            corpus_revision=task.corpus_revision,
            limit=config.search_limit,
        )
        hits = TypeAdapter(list[SearchHit]).validate_python(raw_hits, strict=True)
    except Exception as exc:
        state.stop_reason = StopReason.INFRASTRUCTURE_ERROR
        # Текст исключения бэкенда может содержать URL, credentials или тело запроса.
        state.error = EpisodeError(code="search_error", message=f"Ошибка поиска: {type(exc).__name__}.")
        return tool_error(state.error.code, state.error.message)

    shown = []
    seen_in_search: set[str] = set()
    for hit in hits:
        if hit.doc_id in seen_in_search:
            continue
        seen_in_search.add(hit.doc_id)
        shown.append(hit.model_copy(update={"snippet": hit.snippet[: config.max_snippet_chars]}))
        if len(shown) == config.search_limit:
            break
    state.discovered_ids.update(hit.doc_id for hit in shown)
    return {"ok": True, "hits": [hit.model_dump() for hit in shown]}

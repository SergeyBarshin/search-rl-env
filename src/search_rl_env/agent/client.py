"""Адаптер OpenRouter Chat Completions без автоматических повторов запросов."""

from typing import Any

import httpx
from pydantic import ValidationError

from search_rl_env.contracts.types import ModelResponse, Usage


class ModelError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("Нужны непустые api_key и model.")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._owns_client = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient(timeout=timeout_seconds)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "openrouter-chat-completions",
            "model": self._model,
            "temperature": self._temperature,
            "endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "provider": {"require_parameters": True},
        }

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        seed: int | None,
    ) -> ModelResponse:
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tokens": max_output_tokens,
            "temperature": self._temperature,
            "provider": {"require_parameters": True},
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        try:
            response = await self._http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
        except httpx.HTTPError:
            raise ModelError("Ошибка транспорта OpenRouter.") from None
        if not response.is_success:
            raise ModelError(f"OpenRouter HTTP {response.status_code}.")
        try:
            data = response.json()
            if data.get("error"):
                raise ModelError("OpenRouter вернул ошибку API.")
            choice = data["choices"][0]
            if choice.get("error"):
                raise ModelError("Провайдер OpenRouter вернул ошибку.")
            return ModelResponse(
                message=choice["message"],
                usage=Usage.model_validate(data.get("usage") or {}),
                finish_reason=choice.get("finish_reason"),
                response_id=data.get("id"),
                model=data.get("model"),
                provider=data.get("provider"),
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError, ValidationError):
            raise ModelError("Неверный формат ответа OpenRouter.") from None

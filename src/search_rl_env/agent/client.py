"""Chat Completions через OpenRouter или api, без автоповторов запросов."""

import asyncio
import math
from typing import Any

import httpx
from pydantic import ValidationError

from search_rl_env.contracts.types import ModelError, ModelResponse, ModelTimeoutError, Usage


class ChatCompletionsClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str,
        auth_scheme: str = "Bearer",
        adapter: str = "chat-completions",
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("Нужны непустые api_key и model.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds должен быть конечным положительным числом.")
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._auth_scheme = auth_scheme
        self._adapter = adapter
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._owns_client = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient(timeout=timeout_seconds)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": self._adapter,
            "model": self._model,
            "temperature": self._temperature,
            "timeout_seconds": self._timeout_seconds,
            "endpoint": self._endpoint,
            "provider": {"require_parameters": True},
        }

    async def __aenter__(self) -> "ChatCompletionsClient":
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
            "max_tokens": max_output_tokens,
            "temperature": self._temperature,
            "provider": {"require_parameters": True},
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._http.post(
                    self._endpoint,
                    headers={"Authorization": f"{self._auth_scheme} {self._api_key}"},
                    json=payload,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise ModelTimeoutError(f"Таймаут запроса к модели (лимит {self._timeout_seconds:g} с).") from None
        except httpx.HTTPError:
            raise ModelError("Ошибка транспорта Chat Completions.") from None
        if not response.is_success:
            raise ModelError(f"Chat Completions HTTP {response.status_code}.")
        try:
            data = response.json()
            if data.get("error"):
                raise ModelError("Chat Completions вернул ошибку API.")
            choice = data["choices"][0]
            if choice.get("error"):
                raise ModelError("Провайдер Chat Completions вернул ошибку.")
            return ModelResponse(
                message=choice["message"],
                usage=Usage.model_validate(data.get("usage") or {}),
                finish_reason=choice.get("finish_reason"),
                response_id=data.get("id"),
                model=data.get("model"),
                provider=data.get("provider"),
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError, ValidationError):
            raise ModelError("Неверный формат ответа Chat Completions.") from None


class OpenRouterClient(ChatCompletionsClient):
    def __init__(self, *, api_key: str, model: str, **kwargs: Any) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            auth_scheme="Bearer",
            adapter="openrouter-chat-completions",
            **kwargs,
        )


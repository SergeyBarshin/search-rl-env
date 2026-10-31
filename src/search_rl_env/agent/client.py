"""Клиенты Chat Completions с настраиваемым endpoint, без автоповторов."""

import asyncio
import json
import math
from time import perf_counter
from typing import Any

import httpx
from pydantic import ValidationError

from search_rl_env.contracts.types import ModelError, ModelResponse, ModelTimeoutError, Usage
from search_rl_env.profiling import HttpRecorder


class ChatCompletionsClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str,
        auth_scheme: str = "Bearer",
        adapter: str = "chat-completions",
        require_parameters: bool = False,
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
        self._require_parameters = require_parameters
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
            "require_parameters": self._require_parameters,
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
        profiler = HttpRecorder()
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": max_output_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        if self._require_parameters:
            payload["provider"] = {"require_parameters": True}
        if seed is not None:
            payload["seed"] = seed
        try:
            request = self._http.build_request(
                'POST', self._endpoint,
                headers={"Authorization": f"{self._auth_scheme} {self._api_key}"},
                json=payload,
                extensions={'trace': profiler.trace},
            )
            profiler.data.request_bytes = len(request.content)
            profiler.data.request_prepare_seconds = perf_counter() - profiler.started
            profiler.notify('request', 'started')
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    response = await self._http.send(request, stream=True)
                    try:
                        profiler.headers(response)
                        if not response.is_success:
                            raise ModelError(f"Chat Completions HTTP {response.status_code}.")
                        body_started = perf_counter()
                        body = bytearray()
                        try:
                            async for chunk in response.aiter_bytes():
                                profiler.body_chunk(len(chunk))
                                body.extend(chunk)
                        finally:
                            profiler.data.body_read_seconds = perf_counter() - body_started
                    finally:
                        await response.aclose()
            except (TimeoutError, httpx.TimeoutException):
                raise ModelTimeoutError(f"Таймаут запроса к модели (лимит {self._timeout_seconds:g} с).") from None
            except httpx.HTTPError:
                raise ModelError("Ошибка транспорта Chat Completions.") from None
            decode_started = perf_counter()
            try:
                data = json.loads(body)
                if data.get("error"):
                    raise ModelError("Chat Completions вернул ошибку API.")
                choice = data["choices"][0]
                if choice.get("error"):
                    raise ModelError("Провайдер Chat Completions вернул ошибку.")
                result = ModelResponse(
                    message=choice["message"], usage=Usage.model_validate(data.get("usage") or {}),
                    finish_reason=choice.get("finish_reason"), response_id=data.get("id"),
                    model=data.get("model"), provider=data.get("provider"),
                )
            except (ValueError, KeyError, IndexError, TypeError, AttributeError, ValidationError):
                raise ModelError("Неверный формат ответа Chat Completions.") from None
            finally:
                profiler.data.decode_seconds = perf_counter() - decode_started
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            exc.http_profile = profiler.finish(error_code='interrupted' if cancelled else getattr(exc, 'code', 'model_error'),
                                               cancelled=cancelled)
            raise
        result.http_profile = profiler.finish()
        return result


class OpenRouterClient(ChatCompletionsClient):
    def __init__(self, *, api_key: str, model: str, **kwargs: Any) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            auth_scheme="Bearer",
            adapter="openrouter-chat-completions",
            require_parameters=True,
            **kwargs,
        )

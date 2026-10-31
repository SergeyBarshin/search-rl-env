"""Локальный адаптер существующих клиентов модели и SearchBackend к Verifiers."""

import asyncio
import json
import os
from copy import deepcopy
from time import perf_counter, time
from uuid import uuid4

from aiohttp import web

from search_rl_env.contracts.types import EpisodeError, EpisodeEvent, ModelError, Usage


class BackendBridge:
    def __init__(self, task, model, search, config, on_event=None):
        self.task, self.model, self.search, self.config = task, model, search, config
        self.on_event = on_event
        self.token = uuid4().hex
        self.key_var = 'SEARCH_RL_API_KEY_' + uuid4().hex.upper()
        self.records = []
        self.search_calls = 0
        self.error = None
        self._active = set()

    async def __aenter__(self):
        app = web.Application(middlewares=[self._authorize])
        app.add_routes([web.post('/v1/chat/completions', self._model), web.post('/search', self._search)])
        self.runner = web.AppRunner(app, shutdown_timeout=0.1, handle_signals=False)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        os.environ[self.key_var] = self.token
        return self

    async def __aexit__(self, *args):
        for task in list(self._active):
            task.cancel()
        if self._active:
            await asyncio.gather(*self._active, return_exceptions=True)
        await self.runner.cleanup()
        os.environ.pop(self.key_var, None)

    @web.middleware
    async def _authorize(self, request, handler):
        if request.headers.get('Authorization') != 'Bearer ' + self.token:
            raise web.HTTPUnauthorized()
        task = asyncio.current_task()
        self._active.add(task)
        try:
            return await handler(request)
        finally:
            self._active.discard(task)

    def emit(self, kind, step, **details):
        if self.on_event:
            self.on_event(EpisodeEvent(kind=kind, step=step, details=deepcopy(details)))

    async def _model(self, request):
        body = await request.json()
        step = len(self.records) + 1
        max_tokens = min(body.get('max_tokens') or self.config.max_tokens_per_step, self.config.max_tokens_per_step)
        record = {'messages': deepcopy(body['messages']), 'max_tokens': max_tokens,
                  'response': None, 'duration': 0.0}
        self.records.append(record)
        self.emit('model_started', step, max_output_tokens=max_tokens)
        started = perf_counter()
        try:
            response = await self.model.complete(body['messages'], tools=body.get('tools', []),
                max_output_tokens=max_tokens, seed=self.config.seed)
            record['response'] = response
            if response.usage.prompt_tokens is None or response.usage.completion_tokens is None:
                self.error = EpisodeError(code='missing_usage', message='Нет токенной статистики ответа модели.')
                return web.json_response({'error': {'message': self.error.message}}, status=502)
            if response.finish_reason == 'error':
                self.error = EpisodeError(code='provider_error', message='Провайдер вернул ошибку генерации.')
                return web.json_response({'error': {'message': self.error.message}}, status=502)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = EpisodeError(code=exc.code if isinstance(exc, ModelError) else 'model_error',
                message=str(exc) if isinstance(exc, ModelError) else f'Ошибка модели: {type(exc).__name__}.')
            return web.json_response({'error': {'message': self.error.message}}, status=504 if self.error.code == 'model_timeout' else 502)
        finally:
            record['duration'] = perf_counter() - started
        self.emit('model_finished', step, model=response.model, usage=response.usage.model_dump(),
                  duration_seconds=record['duration'])
        message = deepcopy(response.message)
        if message.get('tool_calls'):
            for index, call in enumerate(message['tool_calls']):
                call['index'] = index
        metadata = {'id': response.response_id or 'chatcmpl-' + uuid4().hex,
                    'created': int(time()), 'model': response.model or 'model'}
        usage = response.usage.model_dump(exclude_none=True)
        # OpenAI SDK требует total_tokens; исходный usage в EpisodeResult не меняем.
        usage.setdefault('total_tokens', usage['prompt_tokens'] + usage['completion_tokens'])
        if not body.get('stream'):
            return web.json_response({**metadata, 'object': 'chat.completion',
                'choices': [{'index': 0, 'message': message, 'finish_reason': response.finish_reason}], 'usage': usage})
        chunks = [
            {**metadata, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': message, 'finish_reason': None}]},
            {**metadata, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {},
                'finish_reason': response.finish_reason or 'stop'}], 'usage': usage},
        ]
        return web.Response(text=''.join('data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n' for chunk in chunks)
                            + 'data: [DONE]\n\n', content_type='text/event-stream')

    async def _search(self, request):
        body = await request.json()
        self.search_calls += 1
        try:
            hits = await self.search.search(body['query'], corpus_id=self.task.corpus_id,
                corpus_revision=self.task.corpus_revision, limit=self.config.search_limit)
            return web.json_response([hit.model_dump() for hit in hits])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return web.json_response({'error': f'Ошибка поиска: {type(exc).__name__}.'}, status=502)

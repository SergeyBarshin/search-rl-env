"""Локальный адаптер существующих клиентов модели и SearchBackend к Verifiers."""

import asyncio
import json
import os
from copy import deepcopy
from time import perf_counter, time
from uuid import uuid4

from aiohttp import web

from search_rl_env.contracts.types import EpisodeError, EpisodeEvent, ModelError, Usage, SqlQueryError
from search_rl_env.profiling import observe_http


def completion_events(payload):
    metadata = {key: payload[key] for key in ('id', 'created', 'model')}
    choice = payload['choices'][0]
    chunks = [
        {**metadata, 'object': 'chat.completion.chunk', 'choices': [{'index': 0,
            'delta': choice['message'], 'finish_reason': None}]},
        {**metadata, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {},
            'finish_reason': choice['finish_reason'] or 'stop'}], 'usage': payload['usage']},
    ]
    return [('data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n').encode() for chunk in chunks] + [b'data: [DONE]\n\n']


class BridgeServer:
    """Один listener на runner; регистрация trace id и отдельные URL инструментов."""

    def __init__(self):
        self.bridges = {}
        self.traces = {}
        self.token = uuid4().hex
        self.key_var = 'SEARCH_RL_API_KEY_' + uuid4().hex.upper()

    async def __aenter__(self):
        app = web.Application()
        app.add_routes([web.post(path, self._dispatch) for path in
                        ('/v1/chat/completions', '/episodes/{token}/search', '/episodes/{token}/sql')])
        self.runner = web.AppRunner(app, shutdown_timeout=0.1, handle_signals=False)
        await self.runner.setup()
        try:
            site = web.TCPSite(self.runner, '127.0.0.1', 0)
            await site.start()
            self.url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
            os.environ[self.key_var] = self.token
        except BaseException:
            await self.runner.cleanup()
            raise
        return self

    async def __aexit__(self, *exc):
        try:
            await self.runner.cleanup()
        finally:
            os.environ.pop(self.key_var, None)

    async def _dispatch(self, request):
        if request.headers.get('Authorization') != 'Bearer ' + self.token:
            raise web.HTTPUnauthorized()
        bridge = (self.bridges.get(request.match_info['token']) if 'token' in request.match_info
                  else self.traces.get(request.headers.get('X-Session-ID')))
        if bridge is None:
            raise web.HTTPUnauthorized()
        route = request.path.rsplit('/', 1)[-1]
        handler = {'completions': bridge._model, 'search': bridge._search, 'sql': bridge._sql}[route]
        return await bridge._authorize(request, handler)


class BackendBridge:
    def __init__(self, task, model, search, config, on_event=None, *, server, sql=None):
        self.task, self.model, self.search, self.config = task, model, search, config
        self.on_event = on_event
        self.token = uuid4().hex
        self.key_var = server.key_var
        self.records = []
        self.search_calls = 0
        self.sql_calls = 0
        self.sql = sql
        self.error = None
        self._active = set()
        self.server = server
        self.trace_id = None

    async def __aenter__(self):
        self.server.bridges[self.token] = self
        self.url = self.server.url + '/episodes/' + self.token
        self.model_url = self.server.url + '/v1'
        return self

    async def __aexit__(self, *args):
        for task in list(self._active):
            task.cancel()
        if self._active:
            await asyncio.gather(*self._active, return_exceptions=True)
        self.server.bridges.pop(self.token, None)
        self.server.traces.pop(self.trace_id, None)

    @web.middleware
    async def _authorize(self, request, handler):
        if request.headers.get('Authorization') != 'Bearer ' + self.server.token:
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
        payload, status = await self.complete(body)
        if status != 200 or not body.get('stream'):
            return web.json_response(payload, status=status)
        return web.Response(body=b''.join(completion_events(payload)), content_type='text/event-stream')

    async def complete(self, body):
        step = len(self.records) + 1
        max_tokens = min(body.get('max_tokens') or self.config.max_tokens_per_step, self.config.max_tokens_per_step)
        # Текущее продолжение уже содержит всю историю. Предыдущие snapshots
        # не нужны проекции и росли квадратично с длиной эпизода.
        if self.records:
            self.records[-1].pop('messages', None)
        record = {'messages': body['messages'], 'max_tokens': max_tokens,
                  'response': None, 'duration': 0.0, 'http_profile': None, 'started': perf_counter()}
        self.records.append(record)
        self.emit('model_started', step, max_output_tokens=max_tokens)
        started = perf_counter()

        def http_progress(profile, phase, status):
            record['http_profile'] = profile
            self.emit('model_progress', step, phase=phase, status=status, http_profile=profile.model_dump(mode='json'))

        try:
            with observe_http(http_progress):
                response = await self.model.complete(body['messages'], tools=body.get('tools', []),
                    max_output_tokens=max_tokens, seed=self.config.seed)
            record['response'] = response
            record['http_profile'] = response.http_profile
            if response.usage.prompt_tokens is None or response.usage.completion_tokens is None:
                self.error = EpisodeError(code='missing_usage', message='Нет токенной статистики ответа модели.')
                return {'error': {'message': self.error.message}}, 502
            if response.finish_reason == 'error':
                self.error = EpisodeError(code='provider_error', message='Провайдер вернул ошибку генерации.')
                return {'error': {'message': self.error.message}}, 502
        except asyncio.CancelledError as exc:
            record['http_profile'] = getattr(exc, 'http_profile', record['http_profile'])
            raise
        except Exception as exc:
            record['http_profile'] = getattr(exc, 'http_profile', record['http_profile'])
            self.error = EpisodeError(code=exc.code if isinstance(exc, ModelError) else 'model_error',
                message=str(exc) if isinstance(exc, ModelError) else f'Ошибка модели: {type(exc).__name__}.')
            return {'error': {'message': self.error.message}}, 504 if self.error.code == 'model_timeout' else 502
        finally:
            record['duration'] = perf_counter() - started
        self.emit('model_finished', step, model=response.model, usage=response.usage.model_dump(),
                  duration_seconds=record['duration'],
                  response_id=response.response_id,
                  http_profile=record['http_profile'].model_dump(mode='json') if record['http_profile'] else None)
        message = deepcopy(response.message)
        if message.get('tool_calls'):
            for index, call in enumerate(message['tool_calls']):
                call['index'] = index
        metadata = {'id': response.response_id or 'chatcmpl-' + uuid4().hex,
                    'created': int(time()), 'model': response.model or 'model'}
        usage = response.usage.model_dump(exclude_none=True)
        # OpenAI SDK требует total_tokens; исходный usage в EpisodeResult не меняем.
        usage.setdefault('total_tokens', usage['prompt_tokens'] + usage['completion_tokens'])
        return {**metadata, 'object': 'chat.completion',
                'choices': [{'index': 0, 'message': message, 'finish_reason': response.finish_reason}], 'usage': usage}, 200

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

    async def _sql(self, request):
        if self.sql is None:
            raise web.HTTPNotFound()
        body = await request.json()
        self.sql_calls += 1
        try:
            return web.json_response((await self.sql.query(body['query'])).model_dump())
        except SqlQueryError as exc:
            return web.json_response({'query_error': str(exc)})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return web.json_response({'error': f'Ошибка SQL: {type(exc).__name__}.'}, status=502)

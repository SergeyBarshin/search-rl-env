"""Адаптер Client для локальной модели, sampling/graph остаются в Verifiers."""

from verifiers.v1.clients import Client
from verifiers.v1.clients.client import RelayReply
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.errors import ProviderError
from verifiers.v1.interception import InterceptionServer

from .bridge import completion_events


class BridgeClient(Client):
    def __init__(self, bridge):
        self.bridge = bridge

    async def get_response(self, dialect, body, sampling, session_id=None, turn=None, headers=None):
        payload = await self._complete(body, session_id)
        response = dialect.parse_response(dialect.validate_response(payload))
        response.raw = payload
        return response

    async def _complete(self, body, session_id):
        episode = self.bridge.traces.get(session_id)
        if episode is None:
            raise ProviderError('Эпизод уже освобождён.', status_code=409)
        payload, status = await episode.complete(body)
        if status != 200:
            raise ProviderError(payload['error']['message'], status_code=status)
        return payload

    async def relay(self, dialect, body, session_id=None, headers=None):
        payload = await self._complete(body, session_id)

        async def chunks():
            for event in completion_events(payload):
                yield event

        async def close():
            pass

        return RelayReply(content_type='text/event-stream', chunks=chunks(), close=close)


class HarnessInterception(InterceptionServer):
    """Изолированный factory hook закреплённого Verifiers 0.4.0.

    Только bridge нашей библиотеки получает прямой Client. Иные endpoint,
    включая TrainClient, создаются штатной фабрикой. Каждый запрос harness
    по-прежнему проходит HTTP interception с отдельной capability и graph.
    """

    def __init__(self, bridge):
        super().__init__(requires_tunnel=False)
        self.bridge = bridge

    def _client(self, config):
        if not isinstance(config, EvalClientConfig) or config.base_url != self.bridge.url + '/v1':
            return super()._client(config)
        key = config.model_dump_json()
        if key not in self.clients:
            client = self.clients[key] = BridgeClient(self.bridge)
            self.stack.push_async_callback(client.close)
        return self.clients[key]

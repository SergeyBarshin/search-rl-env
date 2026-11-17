import json

import httpx
import pytest

from search_rl_env.agent.client import ChatCompletionsClient
from search_rl_env.contracts.types import ModelContextError, EpisodeConfig


async def test_no_generation_cap_or_history_truncation_when_unset():
    history=[{'role':'user','content':'evidence '*20000}]
    def handler(request):
        body=json.loads(request.content)
        assert 'max_tokens' not in body
        assert body['messages']==history
        return httpx.Response(200,json={'choices':[{'message':{'role':'assistant',
            'content':'Answer'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':20000,'completion_tokens':1}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model=ChatCompletionsClient(api_key='test-key',endpoint='https://example.invalid',
                                    model='test-model',http_client=http)
        await model.complete(history,tools=[],max_output_tokens=None,seed=None)
    config=EpisodeConfig(max_tokens_per_step=None,max_snippet_chars=None)
    assert config.max_steps is config.max_output_tokens is config.max_tokens_per_step is config.max_snippet_chars is None


async def test_provider_context_error_is_explicit_and_does_not_leak_response():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(400,
            json={'error':{'code':'context_length_exceeded','message':'private provider body'}}))) as http:
        model=ChatCompletionsClient(api_key='test-key',endpoint='https://example.invalid',
                                    model='test-model',http_client=http)
        with pytest.raises(ModelContextError) as error:
            await model.complete([],tools=[],max_output_tokens=None,seed=None)
        assert error.value.code=='model_context_limit'
        assert 'private provider body' not in str(error.value)

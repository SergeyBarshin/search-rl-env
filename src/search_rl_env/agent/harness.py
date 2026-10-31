"""Harness Verifiers: локальное исполнение или отдельный процесс с MCP."""

import json
import sys
from contextvars import ContextVar
from typing import Literal

import aiohttp

from verifiers.v1 import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult

from .tools import tool_schemas

__all__ = ["SearchHarness"]

# Контекст принадлежит runner, не глобальному event loop. Между rollout
# передаётся только транспорт; messages, credentials и инструменты — локальные.
harness_http: ContextVar[aiohttp.ClientSession | None] = ContextVar('harness_http', default=None)


class SearchHarnessConfig(HarnessConfig):
    id: str = "search_rl_env"
    execution: Literal['in_process', 'subprocess'] = 'in_process'


class SearchHarness(Harness[SearchHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_TOOL_INTERCEPTION = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url=None):
        if self.config.execution == 'subprocess':
            return await self._launch_subprocess(ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                                                 tool_interception_url)
        if not runtime.is_local or mcp_urls or trace.state._executor is None:
            raise ValueError('in_process требует локальный runtime и tools.execution=in_process.')
        borrowed = harness_http.get()
        if borrowed is not None:
            return await self._run_local(borrowed, ctx, trace, endpoint, secret, data, tool_interception_url)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=5)) as client:
            return await self._run_local(client, ctx, trace, endpoint, secret, data, tool_interception_url)

    async def _run_local(self, client, ctx, trace, endpoint, secret, data, gate_url):
        from verifiers.v1.dialects.chat import message_to_wire
        system, prompt = self.resolve_prompt(data)
        messages = [{'role': 'system', 'content': system}] if system else []
        messages += ([{'role': 'user', 'content': prompt}] if isinstance(prompt, str)
                     else [message_to_wire(item) for item in prompt or []])
        headers = {'Authorization': 'Bearer ' + secret}
        executor = trace.state._executor
        async with executor:
            while True:
                # Любой sampling проходит через Verifiers interception: TrainClient
                # сохраняет настоящие token ids/logprobs, а graph строит loss masks.
                async with client.post(endpoint.rstrip('/') + '/chat/completions', headers=headers,
                    json={'model': ctx.model, 'messages': messages,
                          'tools': tool_schemas(executor.config.enable_sql), 'stream': False}) as response:
                    if response.status >= 400:
                        if trace.stop_condition is not None:
                            break
                        response.raise_for_status()
                    completion = await response.json()
                if trace.stop_condition is not None:
                    break
                message = completion['choices'][0]['message']
                messages.append(message)
                calls = message.get('tool_calls') or []
                if not calls:
                    break
                for call in calls:
                    if gate_url:
                        raw_args = call['function']['arguments']
                        try:
                            arguments = json.loads(raw_args)
                        except ValueError:
                            arguments = raw_args
                        async with client.post(gate_url, headers=headers, json={
                            'tool_call_id': call['id'], 'name': call['function']['name'],
                            'arguments': arguments,
                        }) as response:
                            response.raise_for_status()
                            decision = await response.json()
                        if decision['action'] == 'stop':
                            return ProgramResult(0, '', '')
                        if decision['action'] == 'deny':
                            messages.append(decision['message'])
                            continue
                    result = await executor.execute(call)
                    messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                     'name': call['function']['name'],
                                     'content': json.dumps(result, ensure_ascii=False)})
        return ProgramResult(0, '', '')

    async def _launch_subprocess(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                                 tool_interception_url):
        system, prompt = self.resolve_prompt(data)
        path = f".initial-messages-{trace.id}.json"
        if not isinstance(prompt, str):
            from verifiers.v1.dialects.chat import message_to_wire
            await runtime.write(path, json.dumps([message_to_wire(m) for m in prompt or []]).encode())
        executable = sys.executable if runtime.type == "subprocess" else "python3"
        args = [executable, "-m", "search_rl_env.agent.program",
                f"--base-url={endpoint}", f"--api-key={secret}", f"--model={ctx.model}",
                f"--system-prompt={system or ''}",
                "--mcp-config=" + json.dumps({"mcpServers": {
                    name: {"url": url, "timeout": self.config.tool_timeout}
                    for name, url in mcp_urls.items()
                }})]
        args.append(f"--prompt={prompt}" if isinstance(prompt, str) else f"--initial-messages-file={path}")
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        return await runtime.run_program(args, self.config.resolved_env)

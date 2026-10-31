"""Штатный chat loop Verifiers с установленными локально зависимостями."""

import json
import sys

from verifiers.v1 import Harness, HarnessConfig

__all__ = ["SearchHarness"]


class SearchHarnessConfig(HarnessConfig):
    id: str = "search_rl_env"


class SearchHarness(Harness[SearchHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_TOOL_INTERCEPTION = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url=None):
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

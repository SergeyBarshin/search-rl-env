"""Точка запуска штатной программы Verifiers без uv и автоповторов SDK."""

import asyncio
from functools import partial

from openai import AsyncOpenAI
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE


def main():
    namespace = {"__name__": "verifiers_chat_program"}
    exec(compile(CHAT_PROGRAM_SOURCE, "<verifiers-chat-program>", "exec"), namespace)
    namespace["AsyncOpenAI"] = partial(AsyncOpenAI, max_retries=0)
    asyncio.run(namespace["main"]())


if __name__ == "__main__":
    main()

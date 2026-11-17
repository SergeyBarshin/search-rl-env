"""Приёмка собственных SQLite/FAISS-сред: default бесплатный, без моделей."""

import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
import signal
from uuid import uuid4

async def main():
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/environments/acceptance.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--openrouter-model")
    parser.add_argument("--allow-paid", action="store_true", help="Только для отдельно разрешённого API-прогона.")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--request-reserve-usd", type=float)
    args = parser.parse_args()
    if args.openrouter_model and (not args.allow_paid or args.budget_usd is None or args.request_reserve_usd is None):
        parser.error("OpenRouter требует --allow-paid, --budget-usd и --request-reserve-usd после отдельного разрешения бюджета.")
    if not args.openrouter_model and (args.allow_paid or args.budget_usd is not None or args.request_reserve_usd is not None):
        parser.error("Параметры платного прогона требуют --openrouter-model.")
    # Проверяем параметры до загрузки Verifiers и зависимостей модели.
    from search_rl_env.search.acceptance import ReservedBudgetModel, run_acceptance

    output = args.output or Path("runs/environments") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    async with AsyncExitStack() as stack:
        model = None
        if args.openrouter_model:
            from search_rl_env.agent.auth import load_api_key
            from search_rl_env.agent.client import OpenRouterClient

            client = await stack.enter_async_context(OpenRouterClient(api_key=load_api_key(), model=args.openrouter_model))
            model = ReservedBudgetModel(client, args.budget_usd, args.request_reserve_usd)
        report = await run_acceptance(args.config, output, model=model)
    print(f"{report['status']}: {output / 'manifest.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except asyncio.CancelledError:
        raise SystemExit(143) from None

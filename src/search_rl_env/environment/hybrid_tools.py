"""Инструменты связанного эпизода SQL + RAG."""

import verifiers.v1 as vf

from .tools import SearchToolset


class HybridToolset(SearchToolset):
    @vf.tool
    async def sql(self, query: str) -> dict:
        """Выполнить SQL-запрос только на чтение к таблицам из схемы задачи (SQLite)."""
        return await self._execute('sql', {'query': query})

    @vf.tool
    async def finish(self, document_ids: list[str], answer: list[str]) -> dict:
        """Завершить с ответом: id подходящих заказов и id документов-источников."""
        return await self._execute('finish', {'document_ids': document_ids, 'answer': answer})


if __name__ == '__main__':
    HybridToolset.run()

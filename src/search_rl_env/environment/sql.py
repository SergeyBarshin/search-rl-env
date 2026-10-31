"""Подставной SQL-бэкенд: SQLite в памяти с фиксированными заказами."""

import sqlite3

from search_rl_env.contracts.types import SqlQueryError, SqlResult

ORDERS_SCHEMA = """CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    status TEXT NOT NULL, -- delivered, cancelled, pending
    delay_days INTEGER NOT NULL,
    region TEXT NOT NULL -- RU, KZ
);"""

ORDERS = [
    ("ORD-101", "delivered", 5, "RU"),
    ("ORD-102", "delivered", 3, "RU"),
    ("ORD-103", "cancelled", 8, "RU"),
    ("ORD-104", "delivered", 7, "KZ"),
    ("ORD-105", "delivered", 4, "RU"),
    ("ORD-106", "pending", 6, "RU"),
]


class MockSql:
    schema = ORDERS_SCHEMA

    def __init__(self, *, max_rows: int = 100, max_vm_steps: int = 100_000):
        if max_rows < 1 or max_vm_steps < 1:
            raise ValueError("Ограничения результата и вычислений SQL должны быть положительными.")
        self.max_rows = max_rows
        self.max_vm_steps = max_vm_steps
        self.calls: list[str] = []

    async def query(self, query: str) -> SqlResult:
        self.calls.append(query)
        db = sqlite3.connect(":memory:")
        try:
            db.executescript(ORDERS_SCHEMA)
            db.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", ORDERS)
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
            db.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
            remaining = self.max_vm_steps

            def progress():
                nonlocal remaining
                remaining -= 100
                return int(remaining <= 0)

            db.set_progress_handler(progress, 100)
            cursor = db.execute(query)
            if cursor.description is None:
                raise SqlQueryError("SQL должен возвращать таблицу; разрешено только чтение.")
            rows = cursor.fetchmany(self.max_rows + 1)
            if any(isinstance(cell, bytes) for row in rows for cell in row):
                raise SqlQueryError("Бинарные значения в результате не поддерживаются.")
            return SqlResult(columns=[column[0] for column in cursor.description],
                             rows=[list(row) for row in rows[:self.max_rows]],
                             truncated=len(rows) > self.max_rows)
        except sqlite3.Error as exc:
            raise SqlQueryError(str(exc)) from None
        finally:
            db.close()

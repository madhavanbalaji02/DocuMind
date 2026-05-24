"""SQLite connection with asyncpg-compatible API.

Drop-in replacement for the asyncpg pool. Translates $N params → ?
and strips PostgreSQL type casts (::uuid, ::jsonb, etc.) automatically.
Data lives in a local file — no RDS needed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import aiosqlite

from src.core.logging import get_logger

logger = get_logger(__name__)

_DB_PATH = os.environ.get("SQLITE_PATH", "data/documind.db")
_db: aiosqlite.Connection | None = None


def _to_sqlite(query: str, args: tuple) -> tuple[str, tuple]:
    """Convert PostgreSQL-style query to SQLite.

    Handles repeated $N refs (e.g. WHERE a=$1 OR b=$1) by expanding args.
    """
    new_args: list = []

    def _replace(m: re.Match) -> str:
        idx = int(m.group(1)) - 1  # $1 → index 0
        new_args.append(args[idx] if idx < len(args) else None)
        return "?"

    converted = re.sub(r"\$(\d+)", _replace, query)
    converted = re.sub(r"::\w+", "", converted)
    return converted, tuple(new_args)


async def _conn() -> aiosqlite.Connection:
    global _db
    if _db is None:
        Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _db = await aiosqlite.connect(_DB_PATH, check_same_thread=False)
        _db.row_factory = aiosqlite.Row
        await _db.execute("PRAGMA journal_mode=WAL")
        await _db.execute("PRAGMA foreign_keys=ON")
        await _db.execute("PRAGMA synchronous=NORMAL")
        logger.info("SQLite connected: %s", _DB_PATH)
    return _db


async def close_pool() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("SQLite connection closed")


async def execute(query: str, *args: Any) -> None:
    db = await _conn()
    q, a = _to_sqlite(query, args)
    await db.execute(q, a)
    await db.commit()


async def fetch(query: str, *args: Any) -> list[dict]:
    db = await _conn()
    q, a = _to_sqlite(query, args)
    async with db.execute(q, a) as cur:
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def fetchrow(query: str, *args: Any) -> dict | None:
    db = await _conn()
    q, a = _to_sqlite(query, args)
    async with db.execute(q, a) as cur:
        row = await cur.fetchone()
        return dict(row) if row else None


async def fetchval(query: str, *args: Any) -> Any:
    row = await fetchrow(query, *args)
    if row:
        return next(iter(row.values()))
    return None


async def run_migrations() -> None:
    db = await _conn()
    migrations_dir = Path(__file__).parent / "migrations"
    sql_file = migrations_dir / "001_sqlite.sql"
    if not sql_file.exists():
        logger.warning("SQLite migration not found: %s", sql_file)
        return
    logger.info("Running migration: %s", sql_file.name)
    sql = sql_file.read_text()
    # Execute each statement individually (SQLite executescript needs no async)
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            try:
                await db.execute(stmt)
            except Exception as exc:
                logger.warning("Migration stmt skipped (%s): %.80s", exc, stmt)
    await db.commit()
    logger.info("Migration %s completed", sql_file.name)


# Keep get_pool for any legacy callers
async def get_pool():
    return await _conn()

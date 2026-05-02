"""asyncpg connection pool with migration support."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import asyncpg
from asyncpg import Pool, Record

logger = logging.getLogger(__name__)

_pool: Pool | None = None


async def get_pool() -> Pool:
    """Return singleton asyncpg connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        dsn = os.environ["DATABASE_URL_ASYNC"].replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=2,
            max_size=10,
            command_timeout=60,
            server_settings={"application_name": "documind"},
        )
        logger.info("asyncpg pool created (min=2, max=10)")
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("asyncpg pool closed")


async def execute(query: str, *args: Any) -> str:
    """Execute a write query and return its status string."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args: Any) -> list[Record]:
    """Execute a SELECT and return all rows."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args: Any) -> Record | None:
    """Execute a SELECT and return the first row, or None."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetchval(query: str, *args: Any) -> Any:
    """Execute a SELECT and return a single scalar value."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def run_migrations() -> None:
    """Read and execute all SQL migration files in order."""
    migrations_dir = Path(__file__).parent / "migrations"
    sql_files = sorted(migrations_dir.glob("*.sql"))

    if not sql_files:
        logger.warning("No migration files found in %s", migrations_dir)
        return

    pool = await get_pool()
    async with pool.acquire() as conn:
        for sql_file in sql_files:
            logger.info("Running migration: %s", sql_file.name)
            sql = sql_file.read_text()
            try:
                await conn.execute(sql)
                logger.info("Migration %s completed", sql_file.name)
            except Exception as exc:
                logger.error("Migration %s failed: %s", sql_file.name, exc)
                raise

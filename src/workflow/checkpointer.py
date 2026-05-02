"""PostgreSQL-backed LangGraph checkpointer factory."""

from __future__ import annotations

import os


def get_postgres_checkpointer():
    """Return a langgraph-checkpoint-postgres checkpointer using DATABASE_URL."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    conn_string = os.environ["DATABASE_URL"].replace(
        "postgresql+asyncpg://", "postgresql://"
    )
    return AsyncPostgresSaver.from_conn_string(conn_string)

"""Sessions listing and detail routes."""

from __future__ import annotations

from src.core.logging import get_logger

from fastapi import APIRouter, HTTPException

from src.db import connection as db

logger = get_logger(__name__)
router = APIRouter()


@router.get("")
async def list_sessions(limit: int = 20, offset: int = 0, user_id: str | None = None):
    """Return paginated list of research sessions for this browser session."""
    if user_id:
        rows = await db.fetch(
            """
            SELECT s.id, s.user_id, s.topic, s.status, s.created_at, s.completed_at,
                   COUNT(ar.id) AS agent_run_count
            FROM sessions s
            LEFT JOIN agent_runs ar ON ar.session_id = s.id
            WHERE s.user_id = $1
            GROUP BY s.id
            ORDER BY s.created_at DESC
            LIMIT $2 OFFSET $3
            """,
            user_id, limit, offset,
        )
        total = await db.fetchval("SELECT COUNT(*) FROM sessions WHERE user_id = $1", user_id)
    else:
        rows = await db.fetch(
            """
            SELECT s.id, s.user_id, s.topic, s.status, s.created_at, s.completed_at,
                   COUNT(ar.id) AS agent_run_count
            FROM sessions s
            LEFT JOIN agent_runs ar ON ar.session_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset,
        )
        total = await db.fetchval("SELECT COUNT(*) FROM sessions")
    return {"sessions": [dict(r) for r in rows], "total": total}


@router.get("/{session_id}/agents")
async def get_agent_runs(session_id: str):
    """Return all agent run records for a session."""
    rows = await db.fetch(
        """
        SELECT id, agent_name, input, output, tokens_used, duration_ms, created_at
        FROM agent_runs
        WHERE session_id = $1::uuid
        ORDER BY created_at ASC
        """,
        session_id,
    )
    if not rows:
        session = await db.fetchrow("SELECT id FROM sessions WHERE id = $1::uuid", session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
    return [dict(r) for r in rows]

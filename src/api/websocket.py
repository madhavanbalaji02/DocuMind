"""WebSocket endpoint: streams agent output events in real time."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.logging import get_logger
from src.db import connection as db

logger = get_logger(__name__)
router = APIRouter()

POLL_MS = 500


@router.websocket("/ws/{session_id}")
async def research_stream(websocket: WebSocket, session_id: str):
    """Stream session events to the client.

    Event types pushed to client:
      {"type": "status_change", "status": "...", "timestamp": "..."}
      {"type": "agent_update",  "agent": "...", "content": "...", "duration_ms": N}
      {"type": "final_report",  "title": "...", "content": "...", "word_count": N}
      {"type": "done",          "status": "completed"|"failed"}
      {"type": "error",         "message": "..."}
    """
    await websocket.accept()
    logger.info("WS connected session=%s", session_id)

    last_agent_count = 0
    last_status = ""

    try:
        while True:
            session_row = await db.fetchrow(
                "SELECT status, topic FROM sessions WHERE id = $1::uuid", session_id
            )
            if session_row is None:
                await websocket.send_json({"type": "error", "message": "session not found"})
                break

            current_status = session_row["status"]

            # Push any new agent_run rows since last poll
            agent_rows = await db.fetch(
                """
                SELECT agent_name, output, duration_ms, tokens_used, created_at
                FROM agent_runs
                WHERE session_id = $1::uuid
                ORDER BY created_at ASC
                """,
                session_id,
            )
            if len(agent_rows) > last_agent_count:
                for run in agent_rows[last_agent_count:]:
                    await websocket.send_json({
                        "type": "agent_update",
                        "agent": run["agent_name"],
                        "content": (run["output"] or "")[:3000],
                        "duration_ms": run["duration_ms"],
                        "tokens_used": run["tokens_used"],
                        "timestamp": run["created_at"].isoformat(),
                    })
                last_agent_count = len(agent_rows)

            # Push status transitions
            if current_status != last_status:
                await websocket.send_json({
                    "type": "status_change",
                    "status": current_status,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                last_status = current_status

            # Terminal — push report and close
            if current_status in ("completed", "failed"):
                if current_status == "completed":
                    report = await db.fetchrow(
                        "SELECT title, content, word_count FROM reports WHERE session_id = $1::uuid",
                        session_id,
                    )
                    if report:
                        await websocket.send_json({
                            "type": "final_report",
                            "title": report["title"],
                            "content": report["content"],
                            "word_count": report["word_count"],
                        })
                await websocket.send_json({"type": "done", "status": current_status})
                break

            await asyncio.sleep(POLL_MS / 1000)

    except WebSocketDisconnect:
        logger.info("WS disconnected session=%s", session_id)
    except Exception as exc:
        logger.error("WS error session=%s: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        logger.info("WS closed session=%s", session_id)

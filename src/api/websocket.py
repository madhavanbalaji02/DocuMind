"""WebSocket endpoint for streaming agent output to the frontend."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.db import connection as db

logger = logging.getLogger(__name__)
router = APIRouter()

POLL_INTERVAL = 0.5  # seconds


@router.websocket("/ws/{session_id}")
async def research_stream(websocket: WebSocket, session_id: str):
    """Stream session state updates to a connected WebSocket client.

    Polls the database every 500 ms and pushes deltas to the client.
    Closes automatically when the session reaches a terminal state.
    """
    await websocket.accept()
    logger.info("WS connected for session %s", session_id)

    last_agent_count = 0
    last_status = ""

    try:
        while True:
            # Fetch current session state
            session_row = await db.fetchrow(
                "SELECT status, topic FROM sessions WHERE id = $1::uuid", session_id
            )
            if session_row is None:
                await websocket.send_json({"type": "error", "message": "Session not found"})
                break

            current_status = session_row["status"]

            # Fetch latest agent runs since last poll
            agent_rows = await db.fetch(
                """
                SELECT agent_name, output, created_at
                FROM agent_runs
                WHERE session_id = $1::uuid
                ORDER BY created_at ASC
                """,
                session_id,
            )

            if len(agent_rows) > last_agent_count:
                new_runs = agent_rows[last_agent_count:]
                for run in new_runs:
                    await websocket.send_json(
                        {
                            "type": "agent_output",
                            "agent": run["agent_name"],
                            "output": (run["output"] or "")[:2000],
                            "timestamp": run["created_at"].isoformat(),
                        }
                    )
                last_agent_count = len(agent_rows)

            # Push status change
            if current_status != last_status:
                await websocket.send_json(
                    {
                        "type": "status_change",
                        "status": current_status,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                last_status = current_status

            # Terminal states — push final report if available and disconnect
            if current_status in ("completed", "failed"):
                if current_status == "completed":
                    report_row = await db.fetchrow(
                        "SELECT title, content, word_count FROM reports WHERE session_id = $1::uuid",
                        session_id,
                    )
                    if report_row:
                        await websocket.send_json(
                            {
                                "type": "final_report",
                                "title": report_row["title"],
                                "content": report_row["content"],
                                "word_count": report_row["word_count"],
                            }
                        )
                await websocket.send_json({"type": "done", "status": current_status})
                break

            await asyncio.sleep(POLL_INTERVAL)

    except WebSocketDisconnect:
        logger.info("WS disconnected for session %s", session_id)
    except Exception as exc:
        logger.error("WS error for session %s: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        logger.info("WS closed for session %s", session_id)

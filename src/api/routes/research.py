"""Research workflow API routes."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import anthropic
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.logging import get_logger
from src.db import connection as db

logger = get_logger(__name__)
router = APIRouter()


class ResearchRequest(BaseModel):
    topic: str
    user_id: str | None = None
    related_session_id: str | None = None  # inject prior context


class ResearchResponse(BaseModel):
    session_id: str
    status: str
    message: str


async def _run_workflow(session_id: str, topic: str, prior_context: str = "") -> None:
    from src.workflow.graph import build_research_graph
    from src.workflow.state import ResearchState

    graph = build_research_graph(use_checkpointer=False)
    initial_state: ResearchState = {
        "session_id": session_id,
        "topic": topic,
        "research_plan": [],
        "retrieved_context": [],
        "analyst_insights": prior_context,
        "draft_report": "",
        "critic_feedback": "",
        "final_report": "",
        "citations": [],
        "iteration": 0,
        "status": "running",
        "error": None,
    }
    try:
        await db.execute(
            "UPDATE sessions SET status = 'running' WHERE id = $1::uuid", session_id
        )
        await graph.ainvoke(initial_state)
    except anthropic.APIError as exc:
        logger.error("LLM API error for session %s: %s", session_id, exc)
        await db.execute(
            "UPDATE sessions SET status = 'failed' WHERE id = $1::uuid", session_id
        )
    except (UnexpectedResponse, ConnectionError) as exc:
        logger.error("Knowledge base unavailable for session %s: %s", session_id, exc)
        await db.execute(
            "UPDATE sessions SET status = 'failed' WHERE id = $1::uuid", session_id
        )
    except Exception as exc:
        logger.error("Workflow failed for session %s: %s", session_id, exc)
        await db.execute(
            "UPDATE sessions SET status = 'failed' WHERE id = $1::uuid", session_id
        )


@router.post("", response_model=ResearchResponse, status_code=202)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new research workflow."""
    session_id = str(uuid.uuid4())

    # Optionally inject a prior session's report as context
    prior_context = ""
    if request.related_session_id:
        row = await db.fetchrow(
            "SELECT content, title FROM reports WHERE session_id = $1::uuid",
            request.related_session_id,
        )
        if row:
            prior_context = (
                f"## Prior Research Context\n\n"
                f"**Previous report:** {row['title']}\n\n"
                f"{row['content'][:2000]}\n\n"
                f"---\nUse the above as background. Build on it, do not repeat it."
            )

    await db.execute(
        "INSERT INTO sessions (id, user_id, topic, status) VALUES ($1::uuid, $2, $3, 'pending')",
        session_id,
        request.user_id,
        request.topic,
    )
    background_tasks.add_task(_run_workflow, session_id, request.topic, prior_context)
    return ResearchResponse(
        session_id=session_id,
        status="pending",
        message="Research workflow started. Poll /research/{session_id} for status.",
    )


@router.get("/{session_id}")
async def get_research_status(session_id: str):
    """Get session status and agent run count."""
    try:
        row = await db.fetchrow(
            """
            SELECT s.*, COUNT(ar.id) AS agent_run_count
            FROM sessions s
            LEFT JOIN agent_runs ar ON ar.session_id = s.id
            WHERE s.id = $1::uuid
            GROUP BY s.id
            """,
            session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "database unavailable", "detail": str(exc)})

    if row is None:
        raise HTTPException(status_code=404, detail={"error": "session not found", "session_id": session_id})
    return dict(row)


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    """Get the final research report for a completed session."""
    try:
        row = await db.fetchrow(
            "SELECT * FROM reports WHERE session_id = $1::uuid", session_id
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "database unavailable", "detail": str(exc)})

    if row is None:
        session = await db.fetchrow(
            "SELECT status FROM sessions WHERE id = $1::uuid", session_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session not found", "session_id": session_id})
        raise HTTPException(
            status_code=404,
            detail={"error": "report not ready", "session_status": session["status"]},
        )
    return dict(row)


@router.post("/{session_id}/approve")
async def approve_research(session_id: str):
    """Resume a workflow paused at the critic_review interrupt."""
    row = await db.fetchrow(
        "SELECT status FROM sessions WHERE id = $1::uuid", session_id
    )
    if row is None:
        raise HTTPException(status_code=404, detail={"error": "session not found", "session_id": session_id})

    from src.workflow.graph import build_research_graph
    graph = build_research_graph(use_checkpointer=True)
    config = {"configurable": {"thread_id": session_id}}

    async def _resume():
        try:
            await graph.ainvoke(None, config=config)
        except Exception as exc:
            logger.error("Resume failed for session %s: %s", session_id, exc)

    asyncio.create_task(_resume())
    return {"session_id": session_id, "message": "Workflow resumed after human approval"}

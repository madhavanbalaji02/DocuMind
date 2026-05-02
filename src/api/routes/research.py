"""Research workflow API routes."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.db import connection as db

logger = logging.getLogger(__name__)
router = APIRouter()


class ResearchRequest(BaseModel):
    topic: str
    user_id: str | None = None


class ResearchResponse(BaseModel):
    session_id: str
    status: str
    message: str


async def _run_workflow(session_id: str, topic: str) -> None:
    """Background task: run the full LangGraph research workflow."""
    from src.workflow.graph import build_research_graph
    from src.workflow.state import ResearchState

    graph = build_research_graph(use_checkpointer=False)
    initial_state: ResearchState = {
        "session_id": session_id,
        "topic": topic,
        "research_plan": [],
        "retrieved_context": [],
        "analyst_insights": "",
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
    except Exception as exc:
        logger.error("Workflow failed for session %s: %s", session_id, exc)
        await db.execute(
            "UPDATE sessions SET status = 'failed' WHERE id = $1::uuid", session_id
        )


@router.post("", response_model=ResearchResponse, status_code=202)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new research workflow."""
    session_id = str(uuid.uuid4())
    await db.execute(
        """
        INSERT INTO sessions (id, user_id, topic, status)
        VALUES ($1::uuid, $2, $3, 'pending')
        """,
        session_id,
        request.user_id,
        request.topic,
    )
    background_tasks.add_task(_run_workflow, session_id, request.topic)
    return ResearchResponse(
        session_id=session_id,
        status="pending",
        message="Research workflow started. Poll /research/{session_id} for status.",
    )


@router.get("/{session_id}")
async def get_research_status(session_id: str):
    """Get session status and latest LangGraph state."""
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
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return dict(row)


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    """Get the final research report for a completed session."""
    row = await db.fetchrow(
        "SELECT * FROM reports WHERE session_id = $1::uuid", session_id
    )
    if row is None:
        session = await db.fetchrow(
            "SELECT status FROM sessions WHERE id = $1::uuid", session_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        raise HTTPException(
            status_code=404,
            detail=f"Report not ready. Session status: {session['status']}",
        )
    return dict(row)


@router.post("/{session_id}/approve")
async def approve_research(session_id: str):
    """Resume a LangGraph workflow that is paused at the critic_review interrupt."""
    row = await db.fetchrow(
        "SELECT status FROM sessions WHERE id = $1::uuid", session_id
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Resume with checkpointer (requires the graph to be compiled with checkpointer)
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

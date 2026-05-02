"""LangGraph ResearchState definition."""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    session_id: str
    topic: str
    research_plan: list[str]         # questions to investigate
    retrieved_context: list[dict]    # from RAG
    analyst_insights: str            # from SQL analysis
    draft_report: str
    critic_feedback: str
    final_report: str
    citations: list[dict]
    iteration: int                   # revision count (max 2)
    status: str
    error: Optional[str]

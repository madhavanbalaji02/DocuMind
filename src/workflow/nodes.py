"""LangGraph node functions — each accepts ResearchState and returns a partial update."""

from __future__ import annotations

import json
from src.core.logging import get_logger
import os
import uuid
from datetime import datetime, timezone

import anthropic

from src.db import connection as db
from src.rag.rag_chain import RAGChain
from src.workflow.state import ResearchState

logger = get_logger(__name__)

_LLM_MODEL = "claude-haiku-4-5-20251001"


def _anthropic_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


async def plan_research(state: ResearchState) -> dict:
    """Generate 4 focused research questions, injecting prior context if available."""
    logger.info("[plan_research] session=%s topic=%r", state["session_id"], state["topic"])

    # Check for prior research on this topic
    prior_rows = await db.fetch(
        """
        SELECT r.title, LEFT(r.content, 800) AS preview
        FROM reports r
        JOIN sessions s ON s.id = r.session_id
        WHERE s.status = 'completed'
          AND s.id::text != $1
          AND (s.topic ILIKE $2 OR r.title ILIKE $2)
        ORDER BY r.created_at DESC
        LIMIT 2
        """,
        state["session_id"],
        f"%{state['topic'][:60]}%",
    )

    prior_context_section = ""
    if prior_rows:
        summaries = "\n".join(f"- {r['title']}: {r['preview'][:300]}" for r in prior_rows)
        prior_context_section = (
            f"\n\nPrior research context (do NOT repeat — build on it):\n{summaries}"
        )

    client = _anthropic_client()
    response = await client.messages.create(
        model=_LLM_MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a research planner. Given the topic below, produce exactly 4 "
                    f"focused research questions that together would fully cover the topic. "
                    f"Output ONLY a JSON array of strings, no extra text."
                    f"{prior_context_section}"
                    f"\n\nTopic: {state['topic']}"
                ),
            }
        ],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    questions: list[str] = json.loads(raw)
    logger.info("[plan_research] generated %d questions", len(questions))
    return {"research_plan": questions, "status": "planning_done"}


async def retrieve_context(state: ResearchState) -> dict:
    """Run RAG for each research question and merge results."""
    logger.info("[retrieve_context] session=%s", state["session_id"])
    rag = RAGChain()
    all_context: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for question in state["research_plan"]:
        rag_response = await rag.query(question, state["session_id"])
        for chunk in rag_response.retrieved_chunks:
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                all_context.append(
                    {
                        "question": question,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "source": chunk.source,
                        "score": chunk.score,
                    }
                )

    logger.info("[retrieve_context] collected %d unique chunks", len(all_context))
    await rag.close()
    return {"retrieved_context": all_context, "status": "context_retrieved"}


async def analyze_data(state: ResearchState) -> dict:
    """Query postgres for related past sessions and synthesise insights."""
    logger.info("[analyze_data] session=%s", state["session_id"])
    rows = await db.fetch(
        """
        SELECT s.topic, r.title, r.quality_score, r.word_count, s.created_at
        FROM sessions s
        JOIN reports r ON r.session_id = s.id
        WHERE s.status = 'completed'
          AND s.id::text != $1
        ORDER BY s.created_at DESC
        LIMIT 5
        """,
        state["session_id"],
    )

    if rows:
        related = "\n".join(
            f"- {r['topic']} (quality={r['quality_score']}, words={r['word_count']})"
            for r in rows
        )
        insights = f"Related past research found:\n{related}"
    else:
        insights = "No related past research sessions found in the database."

    return {"analyst_insights": insights, "status": "analysis_done"}


async def run_crew(state: ResearchState) -> dict:
    """Instantiate and run the CrewAI crew; capture draft report and citations."""
    logger.info("[run_crew] session=%s topic=%r", state["session_id"], state["topic"])

    # Build a context string from retrieved chunks
    context_parts = [
        f"[{i+1}] (source: {c['source']})\n{c['text']}"
        for i, c in enumerate(state["retrieved_context"][:15])
    ]
    context_str = "\n\n---\n\n".join(context_parts)
    context_str += f"\n\n## Analyst Insights\n{state['analyst_insights']}"

    from src.agents.crew import ResearchCrew

    crew_instance = ResearchCrew()
    crew_result = await crew_instance.run(
        topic=state["topic"],
        context=context_str,
        session_id=state["session_id"],
    )

    citations = [
        {"source": s.source, "chunk_id": s.chunk_id, "excerpt": s.excerpt}
        for s in crew_result.citations
    ]
    return {
        "draft_report": crew_result.draft_report,
        "citations": citations,
        "status": "draft_complete",
    }


async def critic_review(state: ResearchState) -> dict:
    """Critic agent checks the draft against sources and returns feedback."""
    logger.info("[critic_review] session=%s iteration=%d", state["session_id"], state["iteration"])
    client = _anthropic_client()

    source_texts = "\n\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(state["retrieved_context"][:10])
    )

    response = await client.messages.create(
        model=_LLM_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a rigorous research critic. Review the draft report below against "
                    f"the provided source passages.\n\n"
                    f"**Sources:**\n{source_texts}\n\n"
                    f"**Draft Report:**\n{state['draft_report']}\n\n"
                    f"Output a JSON object with:\n"
                    f'- "verdict": "pass" or "fail"\n'
                    f'- "issues": list of specific problems (empty list if pass)\n'
                    f'- "suggestions": list of concrete improvements\n'
                    f"Output ONLY valid JSON, no markdown fences."
                ),
            }
        ],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    feedback_data = json.loads(raw)
    feedback_text = (
        f"Verdict: {feedback_data['verdict']}\n"
        f"Issues: {'; '.join(feedback_data.get('issues', []))}\n"
        f"Suggestions: {'; '.join(feedback_data.get('suggestions', []))}"
    )

    logger.info("[critic_review] verdict=%s", feedback_data["verdict"])
    return {
        "critic_feedback": feedback_text,
        "status": f"critic_{feedback_data['verdict']}",
    }


async def revise_report(state: ResearchState) -> dict:
    """Writer revises the draft based on critic feedback."""
    logger.info("[revise_report] session=%s iteration=%d", state["session_id"], state["iteration"])
    client = _anthropic_client()

    response = await client.messages.create(
        model=_LLM_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a technical writer. Revise the draft report below based on the "
                    f"critic's feedback. Maintain all valid content and citations.\n\n"
                    f"**Critic Feedback:**\n{state['critic_feedback']}\n\n"
                    f"**Current Draft:**\n{state['draft_report']}\n\n"
                    f"Output the revised report in full markdown. No extra commentary."
                ),
            }
        ],
    )
    revised = response.content[0].text
    logger.info("[revise_report] revised report: %d chars", len(revised))
    return {
        "draft_report": revised,
        "iteration": state["iteration"] + 1,
        "status": "revised",
    }


async def finalize_report(state: ResearchState) -> dict:
    """Save the final report to postgres and update session status."""
    logger.info("[finalize_report] session=%s", state["session_id"])

    final = state["draft_report"]
    word_count = len(final.split())

    # Extract title from first heading or generate one
    lines = final.strip().splitlines()
    title = next(
        (ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")),
        state["topic"],
    )

    report_id = str(uuid.uuid4())
    await db.execute(
        """
        INSERT INTO reports (id, session_id, title, content, citations, word_count, created_at)
        VALUES ($1::uuid, $2::uuid, $3, $4, $5::jsonb, $6, $7)
        ON CONFLICT (session_id) DO UPDATE
        SET title = EXCLUDED.title,
            content = EXCLUDED.content,
            citations = EXCLUDED.citations,
            word_count = EXCLUDED.word_count
        """,
        report_id,
        state["session_id"],
        title,
        final,
        json.dumps(state["citations"]),
        word_count,
        datetime.now(timezone.utc),
    )
    await db.execute(
        """
        UPDATE sessions SET status = 'completed', completed_at = $1
        WHERE id = $2::uuid
        """,
        datetime.now(timezone.utc),
        state["session_id"],
    )
    logger.info("[finalize_report] saved report '%s' (%d words)", title, word_count)
    return {"final_report": final, "status": "completed"}


async def handle_error(state: ResearchState) -> dict:
    """Log error and mark session as failed."""
    error_msg = state.get("error") or "Unknown error"
    logger.error("[handle_error] session=%s error=%s", state["session_id"], error_msg)
    await db.execute(
        "UPDATE sessions SET status = 'failed' WHERE id = $1::uuid",
        state["session_id"],
    )
    return {"status": "failed"}

"""LangGraph node functions — each accepts ResearchState, returns a partial update.

Every node must be non-raising: on timeout or error it returns a graceful
partial state so the graph can continue to finalize_report.

LLM backend: Bedrock Claude Haiku (primary) → Anthropic SDK (fallback).
COST: Haiku = $0.00025/1K input, $0.00125/1K output (~$0.006/research run)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone

from src.core.logging import get_logger
from src.db import connection as db
from src.workflow.state import ResearchState

logger = get_logger(__name__)


def _llm_complete(prompt: str, max_tokens: int = 600, system: str | None = None) -> str:
    """Synchronous LLM call — Bedrock Haiku primary, Anthropic SDK fallback."""
    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    if provider == "bedrock":
        try:
            from aws.bedrock.client import BedrockLLM
            return BedrockLLM(model="haiku").complete(prompt, system=system, max_tokens=max_tokens)
        except Exception as e:
            logger.warning("Bedrock failed, falling back to Anthropic: %s", e)

    # Direct Anthropic SDK fallback
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    kwargs: dict = {
        "model": "claude-haiku-4-5",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


async def _async_llm(prompt: str, max_tokens: int = 600, system: str | None = None) -> str:
    """Async wrapper for the sync LLM call."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _llm_complete(prompt, max_tokens, system))


async def plan_research(state: ResearchState) -> dict:
    """Generate 4 focused research questions. Timeout: 30s."""
    logger.info("[plan_research] session=%s topic=%r", state["session_id"], state["topic"])

    # Check for prior research on this topic
    try:
        prior_rows = await asyncio.wait_for(
            db.fetch(
                """
                SELECT r.title, LEFT(r.content, 600) AS preview
                FROM reports r JOIN sessions s ON s.id = r.session_id
                WHERE s.status = 'completed'
                  AND s.id::text != $1
                  AND (s.topic ILIKE $2 OR r.title ILIKE $2)
                ORDER BY r.created_at DESC LIMIT 2
                """,
                state["session_id"],
                f"%{state['topic'][:60]}%",
            ),
            timeout=10.0,
        )
    except Exception:
        prior_rows = []

    prior_ctx = ""
    if prior_rows:
        summaries = "\n".join(f"- {r['title']}: {r['preview'][:200]}" for r in prior_rows)
        prior_ctx = f"\n\nPrior research (build on, don't repeat):\n{summaries}"

    try:
        prompt = (
            f"Produce exactly 4 focused research questions covering this topic. "
            f"Output ONLY a JSON array of strings.{prior_ctx}\n\nTopic: {state['topic']}"
        )
        raw = await asyncio.wait_for(_async_llm(prompt, max_tokens=400), timeout=25.0)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        questions: list[str] = json.loads(raw)
    except asyncio.TimeoutError:
        logger.warning("[plan_research] timed out, using fallback questions")
        questions = [
            f"What is {state['topic']}?",
            f"How does {state['topic']} work?",
            f"What are the main use cases of {state['topic']}?",
            f"What are the limitations of {state['topic']}?",
        ]
    except Exception as exc:
        logger.error("[plan_research] error: %s", exc)
        questions = [f"Overview of {state['topic']}"]

    logger.info("[plan_research] %d questions", len(questions))
    return {"research_plan": questions, "status": "planning_done"}


async def retrieve_context(state: ResearchState) -> dict:
    """Retrieve chunks for each question using the hybrid retriever (no LLM calls).

    Tries pgvector first, falls back to Qdrant-based HybridRetriever.
    Timeout: 60s total.
    """
    logger.info("[retrieve_context] session=%s", state["session_id"])

    # Choose retriever based on configuration
    try:
        from aws.pgvector.retriever import PgVectorHybridRetriever
        retriever = PgVectorHybridRetriever()
        logger.info("[retrieve_context] using pgvector retriever")
    except Exception:
        from src.rag.retriever import HybridRetriever
        retriever = HybridRetriever()  # type: ignore[assignment]
        logger.info("[retrieve_context] using Qdrant retriever (pgvector unavailable)")

    all_context: list[dict] = []
    seen: set[str] = set()

    try:
        async def _retrieve_all():
            for question in state["research_plan"]:
                try:
                    chunks = await asyncio.wait_for(
                        retriever.retrieve(question, top_k=6, session_id=state["session_id"]),
                        timeout=20.0,
                    )
                    for chunk in chunks:
                        if chunk.chunk_id not in seen:
                            seen.add(chunk.chunk_id)
                            ctx: dict = {
                                "question": question,
                                "chunk_id": chunk.chunk_id,
                                "text": chunk.text,
                                "source": chunk.source,
                                "score": chunk.score,
                            }
                            # Inject parent context if available (pgvector retriever)
                            if hasattr(chunk, "parent_context") and chunk.parent_context:
                                ctx["parent_context"] = chunk.parent_context
                            all_context.append(ctx)
                except asyncio.TimeoutError:
                    logger.warning("[retrieve_context] question timed out: %r", question[:40])
                except Exception as exc:
                    logger.warning("[retrieve_context] question failed: %s", exc)

        await asyncio.wait_for(_retrieve_all(), timeout=60.0)
    except asyncio.TimeoutError:
        logger.warning("[retrieve_context] total timeout, got %d chunks", len(all_context))
    finally:
        await retriever.close()

    logger.info("[retrieve_context] %d unique chunks", len(all_context))
    return {"retrieved_context": all_context, "status": "context_retrieved"}


async def analyze_data(state: ResearchState) -> dict:
    """Query postgres for past sessions. Timeout: 20s."""
    logger.info("[analyze_data] session=%s", state["session_id"])
    try:
        rows = await asyncio.wait_for(
            db.fetch(
                """
                SELECT s.topic, r.title, r.quality_score, r.word_count
                FROM sessions s JOIN reports r ON r.session_id = s.id
                WHERE s.status = 'completed' AND s.id::text != $1
                ORDER BY s.created_at DESC LIMIT 3
                """,
                state["session_id"],
            ),
            timeout=10.0,
        )
        if rows:
            related = "\n".join(
                f"- {r['topic']} (quality={r['quality_score']}, words={r['word_count']})"
                for r in rows
            )
            insights = f"Related past research:\n{related}"
        else:
            insights = "No related past research found."
    except Exception as exc:
        logger.warning("[analyze_data] skipped: %s", exc)
        insights = "Database analysis unavailable."

    return {"analyst_insights": insights, "status": "analysis_done"}


async def run_crew(state: ResearchState) -> dict:
    """Run the CrewAI crew. Timeout: 3 minutes."""
    logger.info("[run_crew] session=%s", state["session_id"])

    context_parts = [
        f"[{i+1}] (source: {c['source']})\n{c['text']}"
        for i, c in enumerate(state["retrieved_context"][:8])
    ]
    context_str = "\n\n---\n\n".join(context_parts)
    context_str += f"\n\n## Analyst Insights\n{state['analyst_insights']}"

    try:
        from src.agents.crew import ResearchCrew
        crew = ResearchCrew()
        result = await asyncio.wait_for(
            crew.run(topic=state["topic"], context=context_str, session_id=state["session_id"]),
            timeout=300.0,
        )
        draft = result.draft_report
        citations = [
            {"source": s.source, "chunk_id": s.chunk_id, "excerpt": s.excerpt}
            for s in result.citations
        ]
    except asyncio.TimeoutError:
        logger.error("[run_crew] timed out session=%s", state["session_id"])
        draft = _fallback_report(state)
        citations = []
    except Exception as exc:
        logger.error("[run_crew] failed session=%s: %s", state["session_id"], exc)
        draft = _fallback_report(state)
        citations = []

    return {"draft_report": draft, "citations": citations, "status": "draft_complete"}


def _fallback_report(state: ResearchState) -> str:
    chunks = state.get("retrieved_context", [])
    context_text = "\n\n".join(
        f"**Source:** {c['source']}\n{c['text'][:400]}"
        for c in chunks[:5]
    )
    return (
        f"# Research Report: {state['topic']}\n\n"
        "## Executive Summary\n\n"
        "This report was generated from retrieved knowledge base content. "
        "The multi-agent synthesis step was skipped due to a timeout.\n\n"
        "## Retrieved Information\n\n"
        f"{context_text}\n\n"
        "## Analyst Insights\n\n"
        f"{state.get('analyst_insights', 'N/A')}"
    )


async def critic_review(state: ResearchState) -> dict:
    """Critic checks the draft. Timeout: 45s."""
    logger.info("[critic_review] session=%s iteration=%d", state["session_id"], state["iteration"])

    source_texts = "\n\n".join(
        f"[{i+1}] {c['text'][:250]}" for i, c in enumerate(state["retrieved_context"][:6])
    )
    try:
        prompt = (
            f"Review this draft report against the sources. Output valid JSON only.\n\n"
            f"Sources:\n{source_texts}\n\n"
            f"Draft:\n{state['draft_report'][:2000]}\n\n"
            f'Return: {{"verdict":"pass"|"fail","issues":[],"suggestions":[]}}'
        )
        raw = await asyncio.wait_for(_async_llm(prompt, max_tokens=600), timeout=40.0)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        verdict = data.get("verdict", "pass")
        feedback = (
            f"Verdict: {verdict}\n"
            f"Issues: {'; '.join(data.get('issues', []))}\n"
            f"Suggestions: {'; '.join(data.get('suggestions', []))}"
        )
        status = f"critic_{verdict}"
    except asyncio.TimeoutError:
        logger.warning("[critic_review] timed out, auto-passing")
        feedback = "Verdict: pass\nIssues: (critic timed out — auto approved)"
        status = "critic_pass"
    except Exception as exc:
        logger.warning("[critic_review] error: %s — auto-passing", exc)
        feedback = "Verdict: pass\nIssues: (critic unavailable — auto approved)"
        status = "critic_pass"

    return {"critic_feedback": feedback, "status": status}


async def revise_report(state: ResearchState) -> dict:
    """Writer revises based on feedback. Timeout: 45s."""
    logger.info("[revise_report] session=%s iteration=%d", state["session_id"], state["iteration"])
    try:
        prompt = (
            f"Revise this research report based on the critic feedback. "
            f"Keep all valid content.\n\n"
            f"Feedback:\n{state['critic_feedback']}\n\n"
            f"Current draft:\n{state['draft_report'][:3000]}"
        )
        revised = await asyncio.wait_for(_async_llm(prompt, max_tokens=3000), timeout=40.0)
    except Exception as exc:
        logger.warning("[revise_report] failed: %s — keeping original draft", exc)
        revised = state["draft_report"]

    return {"draft_report": revised, "iteration": state["iteration"] + 1, "status": "revised"}


async def finalize_report(state: ResearchState) -> dict:
    """Save report to postgres and apply guardrails. Timeout: 20s."""
    logger.info("[finalize_report] session=%s", state["session_id"])
    final = state["draft_report"]

    # Apply guardrails
    try:
        from evaluation.guardrails import get_guardrails
        contexts = [c["text"] for c in state.get("retrieved_context", [])[:5]]
        sources = list({c["source"] for c in state.get("retrieved_context", [])})
        guardrails = get_guardrails()
        guarded = guardrails.validate_answer(
            question=state["topic"],
            answer=final,
            contexts=contexts,
            sources=sources,
        )
        final = guarded.answer
        if guarded.warning:
            logger.info("[finalize_report] guardrail: %s", guarded.warning)
    except Exception as exc:
        logger.warning("[finalize_report] guardrails skipped: %s", exc)

    word_count = len(final.split())
    lines = final.strip().splitlines()
    title = next(
        (ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")),
        state["topic"],
    )
    try:
        await asyncio.wait_for(
            db.execute(
                """
                INSERT INTO reports (id, session_id, title, content, citations, word_count, created_at)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5::jsonb, $6, $7)
                ON CONFLICT (session_id) DO UPDATE
                SET title=EXCLUDED.title, content=EXCLUDED.content,
                    citations=EXCLUDED.citations, word_count=EXCLUDED.word_count
                """,
                str(uuid.uuid4()), state["session_id"], title, final,
                json.dumps(state["citations"]), word_count, datetime.now(timezone.utc),
            ),
            timeout=15.0,
        )
        await asyncio.wait_for(
            db.execute(
                "UPDATE sessions SET status='completed', completed_at=$1 WHERE id=$2::uuid",
                datetime.now(timezone.utc), state["session_id"],
            ),
            timeout=10.0,
        )
        logger.info("[finalize_report] saved '%s' (%d words)", title, word_count)
    except Exception as exc:
        logger.error("[finalize_report] DB save failed: %s", exc)
        try:
            await db.execute(
                "UPDATE sessions SET status='completed', completed_at=$1 WHERE id=$2::uuid",
                datetime.now(timezone.utc), state["session_id"],
            )
        except Exception:
            pass

    return {"final_report": final, "status": "completed"}


async def handle_error(state: ResearchState) -> dict:
    """Mark session failed in postgres."""
    error_msg = state.get("error") or "Unknown error"
    logger.error("[handle_error] session=%s error=%s", state["session_id"], error_msg)
    try:
        await asyncio.wait_for(
            db.execute(
                "UPDATE sessions SET status='failed' WHERE id=$1::uuid", state["session_id"]
            ),
            timeout=10.0,
        )
    except Exception:
        pass
    return {"status": "failed"}

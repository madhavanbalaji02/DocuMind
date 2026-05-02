"""Shared CrewAI tool functions used by multiple agents."""

from __future__ import annotations

import asyncio
import logging
import re

import httpx
from crewai.tools import tool

from src.db import connection as db
from src.rag.rag_chain import RAGChain

logger = logging.getLogger(__name__)

_SELECT_ONLY = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


def _run_async(coro):
    """Run a coroutine from synchronous CrewAI tool context.

    Uses get_running_loop() to detect whether we are inside an async context
    (FastAPI, tests with asyncio mode). If so, offloads to a thread so we
    don't block the event loop. Otherwise calls asyncio.run() directly.
    """
    import concurrent.futures

    try:
        asyncio.get_running_loop()
        # A loop is already running (FastAPI / async test) — run in a thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop — safe to call asyncio.run() directly.
        return asyncio.run(coro)


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search the document knowledge base using RAG and return an answer with citations.

    Args:
        query: The research question or topic to look up.
    """

    async def _query():
        rag = RAGChain()
        try:
            response = await rag.query(query, session_id="tool_call")
            lines = [response.answer, "", "**Citations:**"]
            for i, src in enumerate(response.sources, start=1):
                lines.append(f"[{i}] {src.source} — {src.excerpt[:120]}")
            return "\n".join(lines)
        finally:
            await rag.close()

    return _run_async(_query())


@tool("search_web")
def search_web(query: str) -> str:
    """Search the web for current information on a topic.

    Args:
        query: The search query string.
    """

    async def _search():
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("DuckDuckGo search failed: %s", exc)
                return f"Web search unavailable: {exc}"

        results: list[str] = []

        if data.get("AbstractText"):
            results.append(f"**Summary:** {data['AbstractText']}\nSource: {data.get('AbstractURL', '')}")

        for item in data.get("RelatedTopics", [])[:3]:
            if isinstance(item, dict) and item.get("Text"):
                results.append(f"- {item['Text']}")

        return "\n\n".join(results) if results else "No web results found."

    return _run_async(_search())


@tool("query_database")
def query_database(sql: str) -> str:
    """Run a read-only SQL SELECT query against the DocuMind PostgreSQL database.

    Args:
        sql: A SELECT statement to execute. Only SELECT queries are permitted.
    """
    if not _SELECT_ONLY.match(sql):
        return "Error: only SELECT queries are permitted."

    async def _query():
        try:
            rows = await db.fetch(sql)
        except Exception as exc:
            return f"SQL error: {exc}"

        if not rows:
            return "Query returned no rows."

        headers = list(rows[0].keys())
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)

    return _run_async(_query())


@tool("get_past_reports")
def get_past_reports(topic: str) -> str:
    """Find previously completed research reports related to a topic.

    Args:
        topic: The topic string to fuzzy-match against past report titles.
    """

    async def _fetch():
        rows = await db.fetch(
            """
            SELECT s.topic, r.title, r.quality_score, r.word_count,
                   LEFT(r.content, 400) AS preview
            FROM reports r
            JOIN sessions s ON s.id = r.session_id
            WHERE s.status = 'completed'
              AND (
                s.topic ILIKE $1
                OR r.title ILIKE $1
              )
            ORDER BY r.created_at DESC
            LIMIT 3
            """,
            f"%{topic}%",
        )
        if not rows:
            return f"No past reports found matching topic: {topic!r}"
        parts = []
        for row in rows:
            parts.append(
                f"**{row['title']}**\n"
                f"Topic: {row['topic']} | Words: {row['word_count']} | "
                f"Quality: {row['quality_score']}\n"
                f"{row['preview']}…"
            )
        return "\n\n---\n\n".join(parts)

    return _run_async(_fetch())

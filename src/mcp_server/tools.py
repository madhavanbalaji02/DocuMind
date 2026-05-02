"""MCP tool definitions for the DocuMind server."""

from __future__ import annotations

import asyncio
import logging

from mcp.server import Server

logger = logging.getLogger(__name__)


def register_tools(server: Server) -> None:
    """Register all DocuMind MCP tools on *server*."""

    @server.tool()
    async def search_knowledge_base(query: str, top_k: int = 5) -> str:
        """Search the DocuMind knowledge base using RAG.

        Args:
            query: The research question or topic to look up.
            top_k: Number of chunks to retrieve (default 5).
        """
        from src.rag.rag_chain import RAGChain

        rag = RAGChain(top_k_retrieve=top_k, top_n_rerank=min(top_k, 5))
        try:
            response = await rag.query(query, session_id="mcp_tool")
            lines = [response.answer, "", f"*Confidence: {response.confidence:.0%}*", "", "**Sources:**"]
            for i, src in enumerate(response.sources, start=1):
                lines.append(f"[{i}] {src.source} — {src.excerpt[:150]}")
            return "\n".join(lines)
        finally:
            await rag.close()

    @server.tool()
    async def ingest_document(source: str, doc_type: str = "text") -> str:
        """Ingest a document (file path or URL) into the knowledge base.

        Args:
            source: File path or URL of the document to ingest.
            doc_type: One of: pdf, web, csv, text, code.
        """
        from src.ingest.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        try:
            result = await pipeline.ingest(source, doc_type)
            return (
                f"Ingestion complete.\n"
                f"- document_id: {result.document_id}\n"
                f"- chunks created: {result.chunk_count}\n"
                f"- embed time: {result.embed_time_ms:.0f} ms\n"
                f"- total time: {result.total_time_ms:.0f} ms"
            )
        finally:
            await pipeline.close()

    @server.tool()
    async def run_research(topic: str, user_id: str = "mcp_user") -> str:
        """Start an async research workflow on a topic.

        Args:
            topic: The research topic or question.
            user_id: Optional identifier for the requesting user.
        """
        import uuid

        from src.db import connection as db

        await db.run_migrations()
        session_id = str(uuid.uuid4())
        await db.execute(
            """
            INSERT INTO sessions (id, user_id, topic, status)
            VALUES ($1::uuid, $2, $3, 'pending')
            """,
            session_id,
            user_id,
            topic,
        )

        # Fire-and-forget: start workflow in background
        async def _run_graph():
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
            await db.execute(
                "UPDATE sessions SET status = 'running' WHERE id = $1::uuid", session_id
            )
            try:
                await graph.ainvoke(initial_state)
            except Exception as exc:
                logger.error("Research workflow failed for session %s: %s", session_id, exc)
                await db.execute(
                    "UPDATE sessions SET status = 'failed' WHERE id = $1::uuid", session_id
                )

        asyncio.create_task(_run_graph())

        return (
            f"Research workflow started.\n"
            f"- session_id: {session_id}\n"
            f"- topic: {topic}\n"
            f"Use get_research_status to poll progress."
        )

    @server.tool()
    async def get_research_status(session_id: str) -> str:
        """Get the current status of a research workflow.

        Args:
            session_id: UUID of the research session.
        """
        from src.db import connection as db

        row = await db.fetchrow(
            """
            SELECT s.status, s.topic, s.created_at, s.completed_at,
                   COUNT(ar.id) AS agent_run_count
            FROM sessions s
            LEFT JOIN agent_runs ar ON ar.session_id = s.id
            WHERE s.id = $1::uuid
            GROUP BY s.id
            """,
            session_id,
        )
        if row is None:
            return f"No session found with id: {session_id}"
        return (
            f"Session: {session_id}\n"
            f"Topic: {row['topic']}\n"
            f"Status: {row['status']}\n"
            f"Started: {row['created_at']}\n"
            f"Completed: {row['completed_at'] or 'in progress'}\n"
            f"Agent runs: {row['agent_run_count']}"
        )

    @server.tool()
    async def get_report(session_id: str) -> str:
        """Retrieve the final research report for a completed session.

        Args:
            session_id: UUID of the research session.
        """
        from src.db import connection as db

        row = await db.fetchrow(
            """
            SELECT r.title, r.content, r.word_count, r.quality_score, r.citations
            FROM reports r
            WHERE r.session_id = $1::uuid
            """,
            session_id,
        )
        if row is None:
            return f"No report found for session {session_id}. The workflow may still be running."
        return (
            f"# {row['title']}\n\n"
            f"*{row['word_count']} words | quality score: {row['quality_score'] or 'N/A'}*\n\n"
            f"{row['content']}"
        )

    @server.tool()
    async def list_documents() -> str:
        """List all ingested documents in the knowledge base."""
        from src.db import connection as db

        rows = await db.fetch(
            """
            SELECT filename, source_url, doc_type, chunk_count, ingested_at
            FROM documents
            ORDER BY ingested_at DESC
            LIMIT 50
            """
        )
        if not rows:
            return "No documents have been ingested yet."

        lines = ["| Source | Type | Chunks | Ingested |", "|--------|------|--------|---------|"]
        for row in rows:
            source = row["filename"] or row["source_url"] or "unknown"
            lines.append(
                f"| {source[:60]} | {row['doc_type']} | {row['chunk_count']} | {row['ingested_at'].date()} |"
            )
        return "\n".join(lines)

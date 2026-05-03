"""MCP tool definitions — list_tools and call_tool handlers."""

from __future__ import annotations

from src.core.logging import get_logger

from mcp.types import TextContent, Tool

logger = get_logger(__name__)

# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="search_knowledge_base",
        description="Search the DocuMind knowledge base using RAG and return an answer with citations.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research question or topic to look up"},
                "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default 5)"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="ingest_document",
        description="Ingest a document (file path or URL) into the knowledge base.",
        inputSchema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "File path or URL of the document"},
                "doc_type": {
                    "type": "string",
                    "description": "One of: pdf, web, csv, text, code",
                    "default": "text",
                },
            },
            "required": ["source"],
        },
    ),
    Tool(
        name="run_research",
        description="Start an async research workflow on a topic. Returns a session_id immediately.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The research topic or question"},
                "user_id": {"type": "string", "description": "Optional identifier for the requesting user"},
            },
            "required": ["topic"],
        },
    ),
    Tool(
        name="get_research_status",
        description="Get the current status of a research workflow.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "UUID of the research session"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="get_report",
        description="Retrieve the final research report for a completed session.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "UUID of the research session"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="list_documents",
        description="List all ingested documents in the knowledge base.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


# ── Tool implementations ──────────────────────────────────────────────────────

async def _search_knowledge_base(query: str, top_k: int = 5) -> str:
    from src.rag.rag_chain import RAGChain

    rag = RAGChain(top_k_retrieve=top_k, top_n_rerank=min(top_k, 5))
    try:
        response = await rag.query(query, session_id="00000000-0000-0000-0000-000000000001")
        lines = [
            response.answer,
            "",
            f"*Confidence: {response.confidence:.0%}*",
            "",
            "**Sources:**",
        ]
        for i, src in enumerate(response.sources, start=1):
            lines.append(f"[{i}] {src.source} — {src.excerpt[:150]}")
        return "\n".join(lines)
    finally:
        await rag.close()


async def _ingest_document(source: str, doc_type: str = "text") -> str:
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


async def _run_research(topic: str, user_id: str = "mcp_user") -> str:
    import asyncio
    import uuid

    from src.db import connection as db

    session_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO sessions (id, user_id, topic, status) VALUES ($1::uuid, $2, $3, 'pending')",
        session_id,
        user_id,
        topic,
    )

    async def _run_graph():
        from src.workflow.graph import build_research_graph
        from src.workflow.state import ResearchState

        graph = build_research_graph(use_checkpointer=False)
        initial: ResearchState = {
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
            await graph.ainvoke(initial)
        except Exception as exc:
            logger.error("Research workflow failed session=%s: %s", session_id, exc)
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


async def _get_research_status(session_id: str) -> str:
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


async def _get_report(session_id: str) -> str:
    from src.db import connection as db

    row = await db.fetchrow(
        "SELECT title, content, word_count, quality_score FROM reports WHERE session_id = $1::uuid",
        session_id,
    )
    if row is None:
        return f"No report found for session {session_id}. The workflow may still be running."
    return (
        f"# {row['title']}\n\n"
        f"*{row['word_count']} words | quality score: {row['quality_score'] or 'N/A'}*\n\n"
        f"{row['content']}"
    )


async def _list_documents() -> str:
    from src.db import connection as db

    rows = await db.fetch(
        "SELECT filename, source_url, doc_type, chunk_count, ingested_at "
        "FROM documents ORDER BY ingested_at DESC LIMIT 50"
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


# ── Dispatcher ────────────────────────────────────────────────────────────────

async def dispatch(name: str, arguments: dict) -> list[TextContent]:
    """Route a tool call by name and return MCP TextContent results."""
    handlers = {
        "search_knowledge_base": lambda a: _search_knowledge_base(
            a["query"], int(a.get("top_k", 5))
        ),
        "ingest_document": lambda a: _ingest_document(
            a["source"], a.get("doc_type", "text")
        ),
        "run_research": lambda a: _run_research(a["topic"], a.get("user_id", "mcp_user")),
        "get_research_status": lambda a: _get_research_status(a["session_id"]),
        "get_report": lambda a: _get_report(a["session_id"]),
        "list_documents": lambda a: _list_documents(),
    }
    handler = handlers.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name!r}")
    result = await handler(arguments)
    return [TextContent(type="text", text=result)]

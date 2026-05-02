# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Infrastructure (required before running anything)
docker compose up -d

# Install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run API
uvicorn src.api.main:app --reload

# Run Streamlit dashboard
streamlit run src/ui/app.py

# Run MCP server (stdio — use with Claude Desktop, not directly)
documind-mcp

# Ingest a document via CLI
documind-ingest path/to/file.pdf --type pdf
documind-ingest https://example.com/article --type web

# Tests
pytest tests/                          # all tests
pytest tests/test_rag.py -v            # single file
pytest tests/test_workflow.py::TestCriticRouting -v  # single class

# Lint / type check
ruff check src/ tests/
mypy src/
```

## Architecture

The request flow for a research job:

```
POST /research → FastAPI → background task → LangGraph graph (ainvoke)
                                              ↓
                          plan_research → retrieve_context → analyze_data → run_crew
                                                                               ↓
                                                                         critic_review ←──┐
                                                                               ↓          │
                                                              pass / iter≥2   fail        │
                                                               ↓               ↓          │
                                                        finalize_report   revise_report ──┘
                                                               ↓
                                                          postgres reports table
```

**LangGraph (`src/workflow/`)** — `ResearchState` is a `TypedDict` that flows through all nodes. `graph.py:build_research_graph(use_checkpointer=False)` skips the Postgres checkpointer for tests. The graph is compiled with `interrupt_before=["critic_review"]` so human-in-the-loop approval is possible via `POST /research/{id}/approve`. `MAX_ITERATIONS = 2` caps the critic→revise loop.

**CrewAI (`src/agents/`)** — `ResearchCrew.run()` is async but `crew.kickoff()` is synchronous; it's wrapped in `loop.run_in_executor(None, crew.kickoff)` to avoid blocking. The crew runs 4 tasks sequentially: research → analysis → writing → critique. `tasks[2].output` (writer) is used as the draft report. Each agent file (`researcher.py`, `analyst.py`, `writer.py`, `critic.py`) exports a single `build_*()` factory function.

**RAG (`src/rag/`)** — `HybridRetriever` combines dense Qdrant search with BM25 sparse search, fused via Reciprocal Rank Fusion (`RRF_K=60`). The BM25 index is built lazily on the first query by scrolling the entire Qdrant collection. `CrossEncoderReranker` runs `cross-encoder/ms-marco-MiniLM-L-6-v2` in a thread executor. `RAGChain` orchestrates retrieve → rerank → Anthropic claude-haiku generation.

**Embeddings (`src/embeddings/`)** — `EMBEDDING_MODE=local` uses `sentence-transformers` (dim=384, model `BAAI/bge-small-en-v1.5`); `EMBEDDING_MODE=bigred200` POSTs to a vLLM OpenAI-compatible `/embeddings` endpoint (dim=1024, model `BAAI/bge-m3`). The collection vector size is set from `embedder.embedding_dim` at creation time — changing mode after a collection exists will cause a dimension mismatch.

**Ingestion (`src/ingest/`)** — `IngestionPipeline.ingest()` is idempotent: it checks `documents` table by `source_url` or `filename` and skips if already present. Chunk IDs are deterministic SHA-256 hashes, so re-ingesting the same content produces the same Qdrant point IDs.

**Database (`src/db/`)** — `get_pool()` returns a singleton `asyncpg` pool. All query helpers (`execute`, `fetch`, `fetchrow`, `fetchval`) acquire a connection from the pool and release it immediately. Migrations run from `src/db/migrations/*.sql` on startup via `run_migrations()`, called both in the FastAPI lifespan and the MCP server startup.

**MCP server (`src/mcp_server/`)** — stdio transport. All 6 tools are registered in `tools.py:register_tools(server)`. The server calls `db.run_migrations()` on startup so it works standalone without the FastAPI app running.

**Agents' sync/async bridge (`src/agents/tools.py`)** — CrewAI tool functions are synchronous. `_run_async(coro)` bridges to async code: it detects a running event loop (FastAPI context) and uses `ThreadPoolExecutor` + `asyncio.run` in a worker thread; otherwise falls back to `loop.run_until_complete`.

## Key env vars

| Variable | Effect |
|---|---|
| `EMBEDDING_MODE` | `local` or `bigred200` — controls which embedding backend is used |
| `QDRANT_COLLECTION` | collection name (default `documind_chunks`) |
| `DATABASE_URL_ASYNC` | asyncpg DSN — must use `postgresql://` scheme (the `+asyncpg` prefix is stripped in `connection.py`) |
| `ANTHROPIC_API_KEY` | used by all LLM calls (nodes, RAGChain, agent LLM) |

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Infrastructure (required before running anything)
docker compose up -d        # Qdrant :6333, Postgres :5432, Redis :6379
make up                     # same, with port summary printed

# Install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run API
make api                    # uvicorn on 127.0.0.1:8888
# or: uvicorn src.api.main:app --reload

# Run Streamlit dashboard
make ui                     # streamlit on :8501

# Run MCP server (stdio — use with Claude Desktop, not directly)
documind-mcp
# or: make mcp

# Ingest a document via CLI
documind-ingest path/to/file.pdf --type pdf
documind-ingest https://example.com/article --type web

# Tests
make test                             # all tests
pytest tests/test_rag.py -v           # single file
pytest tests/test_workflow.py::TestCriticRouting -v  # single class
make smoke                            # scripts/smoke_test.py
make bench                            # scripts/benchmark.py

# Demo data
make demo                             # loads scripts/load_demo_data.py

# AWS infrastructure (region us-east-2, account hardcoded in aws/config.py)
python aws/setup.py                   # provision all resources (idempotent)
python aws/teardown.py                # destroy all resources (irreversible, prompts)

# Lint / type check
ruff check src/ tests/
mypy src/

# Cleanup
make clean                            # remove __pycache__ / .pyc
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

**WebSocket (`src/api/websocket.py`)** — real-time job status streaming at `WS /ws/{job_id}`. Included in the FastAPI app via `ws_router`.

**Structured logging (`src/core/logging.py`)** — `get_logger(name)` returns a `logging.Logger` with `JSONFormatter` (outputs `{time, level, module, msg}` as JSON). Used throughout; replace `print()` with `logger.info/warning/error`.

**Metrics (`GET /metrics`)** — returns aggregate counts (documents, reports, sessions) directly from Postgres. No external metrics system.

**SQS worker (`src/workers/sqs_worker.py`)** — `SQSIngestionWorker` runs as a daemon thread inside the FastAPI lifespan. Polls SQS (long-poll, 20 s) and triggers Step Functions executions. Disabled (no-op) if `SQS_QUEUE_URL` or `STATE_MACHINE_ARN` are unset.

## AWS enterprise pipeline (`aws/`)

Enterprise-grade ingestion pipeline that runs entirely on AWS. Shares the same AWS access key across DocuMind and Sovereign SRE.

**Provisioning** — `aws/setup.py` creates all resources idempotently. `aws/teardown.py` destroys them (prompts before proceeding — irreversible).

**Resources created:**
- S3 bucket (`documind-documents-{account_id}`) with versioning + lifecycle
- SQS queue + DLQ (`documind-ingestion-queue`, `documind-ingestion-dlq`)
- DynamoDB jobs table (`documind-jobs`)
- IAM execution role (`documind-execution-role`) with least-privilege policies
- SSM parameters (`/documind/anthropic-api-key`, `/documind/rds-connection-string`, etc.)
- RDS `db.t3.micro` with pgvector (`documind-db`) — FREE TIER, stop when not demoing: `aws rds stop-db-instance --db-instance-identifier documind-db`
- Step Functions state machine (`documind-ingestion-pipeline`)
- AWS Glue database + crawler
- Lambda functions: `documind-s3-trigger`, `documind-stage1-parse`, `documind-stage2-embed`, `documind-stage3-store`
- CloudWatch billing alarm at $50

**`aws/bedrock/`** — `BedrockLLM` is a drop-in replacement for direct Anthropic SDK calls. Fallback chain: Bedrock → Anthropic SDK (on throttling). Default model: Claude Haiku (~$3/month for 500 research runs). Cost constants documented in `aws/config.py`.

**`aws/textract/`** — layout-aware PDF parsing. `DetectDocumentText` ($0.0015/page) for plain text, `AnalyzeDocument` ($0.015/page) for tables/forms. Hard cap at `TEXTRACT_MAX_PAGES=50` pages per document.

**`aws/pgvector/`** — `PgVectorHybridRetriever` is a drop-in replacement for `src/rag/retriever.HybridRetriever`. Uses RDS pgvector instead of Qdrant + BM25; injects parent-chunk context into results.

**`aws/step_functions/`** — `PipelineExecutor.start()` launches a Step Functions execution and returns the ARN. `wait_for_completion()` polls until terminal status (SUCCEEDED / FAILED / TIMED_OUT / ABORTED) with a 600 s default timeout.

**`aws/config.py`** — single source of truth for all resource names and cost constants. Region defaults to `us-east-2`. Account ID hardcoded as `574772738370` (also readable from `AWS_ACCOUNT_ID` env var).

## Key env vars

| Variable | Effect |
|---|---|
| `EMBEDDING_MODE` | `local` or `bigred200` — controls which embedding backend is used |
| `QDRANT_COLLECTION` | collection name (default `documind_chunks`) |
| `DATABASE_URL_ASYNC` | asyncpg DSN — must use `postgresql://` scheme (the `+asyncpg` prefix is stripped in `connection.py`) |
| `ANTHROPIC_API_KEY` | used by all LLM calls (nodes, RAGChain, agent LLM); also stored in SSM at `/documind/anthropic-api-key` |
| `AWS_DEFAULT_REGION` | defaults to `us-east-2` |
| `AWS_ACCOUNT_ID` | defaults to `574772738370` (hardcoded in `aws/config.py`) |
| `LLM_PROVIDER` | `bedrock` (default) or `anthropic` — controls `BedrockLLM` fallback behaviour |
| `SQS_QUEUE_URL` | required for SQS worker; worker is disabled if unset |
| `STATE_MACHINE_ARN` | required for SQS worker to trigger Step Functions |
| `RDS_DB` / `RDS_USER` / `RDS_PASSWORD` | pgvector RDS credentials (defaults in `aws/config.py`) |

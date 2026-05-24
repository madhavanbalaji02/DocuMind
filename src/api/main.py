"""FastAPI application entry point with lifespan management."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import ingest, research, sessions
from src.api.routes import documents as documents_router
from src.api.websocket import router as ws_router
from src.core.logging import get_logger

logger = get_logger(__name__)

_sqs_worker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sqs_worker
    import logging as _logging

    from dotenv import load_dotenv
    load_dotenv()

    os.makedirs("logs", exist_ok=True)
    file_handler = _logging.FileHandler("logs/api.log")
    file_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    _logging.getLogger().addHandler(file_handler)
    _logging.getLogger().setLevel(_logging.INFO)

    from src.db import connection as db
    logger.info("Running database migrations")
    await db.run_migrations()

    # Apply pgvector schema (idempotent — skips if tables exist)
    try:
        from aws.pgvector.client import PgVectorClient
        pg = PgVectorClient()
        pg.apply_schema()
        logger.info("pgvector schema applied")
    except Exception as e:
        logger.warning("pgvector schema skipped (local Qdrant mode): %s", e)

    # Ensure Qdrant collection for local dev
    try:
        from src.ingest.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        await pipeline._ensure_collection()
        await pipeline.close()
    except Exception as e:
        logger.warning("Qdrant collection setup skipped: %s", e)

    # Start SQS ingestion worker (no-op if SQS_QUEUE_URL not set)
    try:
        from src.workers.sqs_worker import start_worker_thread
        _sqs_worker = start_worker_thread()
        logger.info("SQS ingestion worker started")
    except Exception as e:
        logger.warning("SQS worker not started: %s", e)

    logger.info("Startup complete")

    yield

    if _sqs_worker:
        _sqs_worker.stop()
    await db.close_pool()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocuMind API",
        description="Multi-agent research and intelligence platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://localhost:3000", "http://127.0.0.1:8888"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(research.router, prefix="/research", tags=["research"])
    app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
    app.include_router(documents_router.router, prefix="/documents", tags=["documents"])
    app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
    app.include_router(ws_router, tags=["websocket"])

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal server error", "detail": str(exc)},
        )

    @app.get("/health", tags=["ops"])
    async def health_check():
        qdrant_ok = False
        postgres_ok = False
        pgvector_ok = False

        try:
            from src.core.qdrant import make_async_qdrant_client
            q = make_async_qdrant_client()
            await q.get_collections()
            await q.close()
            qdrant_ok = True
        except Exception as exc:
            logger.warning("Qdrant health: %s", exc)

        try:
            from src.db import connection as db
            await db.fetchval("SELECT 1")
            postgres_ok = True
        except Exception as exc:
            logger.warning("Postgres health: %s", exc)

        try:
            from aws.pgvector.client import PgVectorClient
            pg = PgVectorClient()
            pgvector_ok = pg.health_check()
        except Exception:
            pass  # pgvector is optional in local dev

        status = "ok" if postgres_ok else "degraded"
        return {
            "status": status,
            "qdrant_ok": qdrant_ok,
            "postgres_ok": postgres_ok,
            "pgvector_ok": pgvector_ok,
            "embedding_mode": os.environ.get("EMBEDDING_MODE", "local"),
            "llm_provider": os.environ.get("LLM_PROVIDER", "bedrock"),
        }

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        """Live system stats."""
        from src.db import connection as db

        qdrant_data: dict = {}
        try:
            from src.core.qdrant import make_async_qdrant_client
            collection_name = os.environ.get("QDRANT_COLLECTION", "documind_chunks")
            q = make_async_qdrant_client()
            info = await q.get_collection(collection_name)
            await q.close()
            qdrant_data = {
                "collection": collection_name,
                "vector_count": info.points_count or 0,
                "segment_count": info.segments_count,
                "status": str(info.status),
            }
        except Exception as exc:
            qdrant_data = {"error": str(exc)}

        pg_data: dict = {}
        try:
            row = await db.fetchrow(
                """
                SELECT
                    COUNT(*)                                                   AS sessions_total,
                    COUNT(*) FILTER (WHERE status = 'completed')              AS sessions_completed,
                    COUNT(*) FILTER (WHERE status = 'failed')                 AS sessions_failed,
                    (SELECT COUNT(*) FROM reports)                             AS reports_generated,
                    (SELECT COUNT(*) FROM agent_runs)                          AS agent_runs_total,
                    (SELECT COUNT(*) FROM documents)                           AS documents_ingested,
                    ROUND(
                        AVG(
                            EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000
                        ) FILTER (WHERE status = 'completed' AND completed_at IS NOT NULL)
                    )::bigint                                                   AS avg_workflow_ms
                FROM sessions
                """
            )
            pg_data = {k: v for k, v in dict(row).items()} if row else {}
        except Exception as exc:
            pg_data = {"error": str(exc)}

        embed_mode = os.environ.get("EMBEDDING_MODE", "local")
        embed_model = (
            os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
            if embed_mode == "local"
            else "amazon.titan-embed-text-v2:0"
        )
        embed_dim = 384 if embed_mode == "local" else 1536
        return {
            "qdrant": qdrant_data,
            "postgres": pg_data,
            "embeddings": {
                "mode": embed_mode,
                "model": embed_model,
                "dimension": embed_dim,
            },
        }

    return app


app = create_app()


def start() -> None:
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8888")),
        reload=True,
        log_level="info",
    )

"""FastAPI application entry point with lifespan management."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import ingest, research, sessions
from src.api.websocket import router as ws_router
from src.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    from src.ingest.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    await pipeline._ensure_collection()
    await pipeline.close()
    logger.info("Startup complete")

    yield

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
        try:
            from qdrant_client import AsyncQdrantClient
            q = AsyncQdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
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
        return {
            "status": "ok" if (qdrant_ok and postgres_ok) else "degraded",
            "qdrant_ok": qdrant_ok,
            "postgres_ok": postgres_ok,
            "embedding_mode": os.environ.get("EMBEDDING_MODE", "local"),
        }

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        """Live system stats."""
        from src.db import connection as db

        qdrant_data: dict = {}
        try:
            from qdrant_client import AsyncQdrantClient
            collection_name = os.environ.get("QDRANT_COLLECTION", "documind_chunks")
            q = AsyncQdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
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
            else os.environ.get("BIGRED200_EMBED_MODEL", "BAAI/bge-m3")
        )
        return {
            "qdrant": qdrant_data,
            "postgres": pg_data,
            "embeddings": {
                "mode": embed_mode,
                "model": embed_model,
                "dimension": 384 if embed_mode == "local" else 1024,
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

"""FastAPI application entry point with lifespan management."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import ingest, research, sessions
from src.api.websocket import router as ws_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    from dotenv import load_dotenv
    load_dotenv()

    from src.db import connection as db
    logger.info("Running database migrations…")
    await db.run_migrations()
    logger.info("Database ready")

    from src.ingest.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    await pipeline._ensure_collection()
    await pipeline.close()
    logger.info("Qdrant collection ensured")

    yield

    logger.info("Shutting down — closing DB pool")
    await db.close_pool()


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocuMind API",
        description="Multi-agent research and intelligence platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://localhost:3000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(research.router, prefix="/research", tags=["research"])
    app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
    app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
    app.include_router(ws_router, tags=["websocket"])

    @app.get("/health", tags=["health"])
    async def health_check():
        qdrant_ok = False
        postgres_ok = False

        try:
            from qdrant_client import AsyncQdrantClient
            qdrant = AsyncQdrantClient(
                url=os.environ.get("QDRANT_URL", "http://localhost:6333")
            )
            await qdrant.get_collections()
            await qdrant.close()
            qdrant_ok = True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)

        try:
            from src.db import connection as db
            await db.fetchval("SELECT 1")
            postgres_ok = True
        except Exception as exc:
            logger.warning("Postgres health check failed: %s", exc)

        return {
            "status": "ok" if (qdrant_ok and postgres_ok) else "degraded",
            "qdrant_ok": qdrant_ok,
            "postgres_ok": postgres_ok,
            "embedding_mode": os.environ.get("EMBEDDING_MODE", "local"),
        }

    return app


app = create_app()


def start() -> None:
    """CLI entry point: uvicorn src.api.main:app."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=True,
        log_level="info",
    )

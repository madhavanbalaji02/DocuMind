"""Orchestrates load → chunk → embed → store for any document source."""

from __future__ import annotations

from src.core.logging import get_logger
import os
import time
import uuid
from dataclasses import dataclass

import typer
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.db import connection as db
from src.embeddings.embedding_service import EmbeddingService
from src.ingest.chunker import SemanticChunker
from src.ingest.document_loader import DocumentLoader

logger = get_logger(__name__)
app = typer.Typer()


@dataclass
class IngestionResult:
    document_id: str
    chunk_count: int
    embed_time_ms: float
    total_time_ms: float


class IngestionPipeline:
    """End-to-end document ingestion: load → chunk → embed → Qdrant + Postgres."""

    def __init__(self) -> None:
        self.loader = DocumentLoader()
        self.chunker = SemanticChunker()
        self.embedder = EmbeddingService()
        self.qdrant = AsyncQdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
        self.collection = os.environ.get("QDRANT_COLLECTION", "documind_chunks")

    async def _ensure_collection(self) -> None:
        """Create Qdrant collection if it does not yet exist."""
        existing = await self.qdrant.get_collections()
        names = {c.name for c in existing.collections}
        if self.collection not in names:
            await self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embedder.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection %r (dim=%d)", self.collection, self.embedder.embedding_dim
            )

    async def _is_already_ingested(self, source: str) -> str | None:
        """Return existing document_id if source was already ingested, else None."""
        row = await db.fetchrow(
            "SELECT id::text FROM documents WHERE source_url = $1 OR filename = $1",
            source,
        )
        return str(row["id"]) if row else None

    async def ingest(self, source: str, doc_type: str) -> IngestionResult:
        """Full ingestion pipeline for a single document source.

        Idempotent: skips sources already present in postgres.
        """
        t_start = time.perf_counter()

        existing_id = await self._is_already_ingested(source)
        if existing_id:
            logger.info("Source already ingested (id=%s), skipping: %s", existing_id, source)
            row = await db.fetchrow(
                "SELECT chunk_count FROM documents WHERE id = $1::uuid", existing_id
            )
            chunk_count = row["chunk_count"] if row else 0
            return IngestionResult(
                document_id=existing_id,
                chunk_count=chunk_count,
                embed_time_ms=0.0,
                total_time_ms=0.0,
            )

        await self._ensure_collection()

        # Load
        text = await self.loader.load(source, doc_type)

        # Chunk
        chunks = self.chunker.chunk_document(text, source, doc_type)
        if not chunks:
            raise ValueError(f"No chunks produced from source: {source!r}")

        chunk_texts = [c.text for c in chunks]

        # Embed
        t_embed = time.perf_counter()
        vectors = await self.embedder.embed_texts(chunk_texts)
        embed_ms = (time.perf_counter() - t_embed) * 1000

        # Upsert to Qdrant
        document_id = str(uuid.uuid4())
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "text": chunks[i].text,
                    "source": chunks[i].source,
                    "doc_type": chunks[i].doc_type,
                    "chunk_index": chunks[i].chunk_index,
                    "document_id": document_id,
                    **{
                        k: v
                        for k, v in chunks[i].metadata.items()
                        if k not in ("source", "doc_type", "chunk_index")
                    },
                },
            )
            for i in range(len(chunks))
        ]
        await self.qdrant.upsert(collection_name=self.collection, points=points)
        logger.info("Upserted %d vectors to Qdrant collection %r", len(points), self.collection)

        # Record in Postgres
        filename = None if source.startswith("http") else source
        url = source if source.startswith("http") else None

        await db.execute(
            """
            INSERT INTO documents (id, filename, source_url, doc_type, chunk_count, metadata)
            VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb)
            """,
            document_id,
            filename,
            url,
            doc_type,
            len(chunks),
            "{}",
        )

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Ingested %s: %d chunks, embed=%.0fms, total=%.0fms",
            source,
            len(chunks),
            embed_ms,
            total_ms,
        )
        return IngestionResult(
            document_id=document_id,
            chunk_count=len(chunks),
            embed_time_ms=embed_ms,
            total_time_ms=total_ms,
        )

    async def close(self) -> None:
        await self.qdrant.close()
        await self.embedder.close()


@app.command()
def cli(
    source: str = typer.Argument(..., help="File path or URL to ingest"),
    doc_type: str = typer.Option("text", "--type", "-t", help="pdf | web | csv | text | code"),
) -> None:
    """CLI entry point: ingest a single document."""
    import asyncio

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async def _run() -> None:
        await db.run_migrations()
        pipeline = IngestionPipeline()
        try:
            result = await pipeline.ingest(source, doc_type)
            print(f"document_id : {result.document_id}")
            print(f"chunk_count : {result.chunk_count}")
            print(f"embed_ms    : {result.embed_time_ms:.0f}")
            print(f"total_ms    : {result.total_time_ms:.0f}")
        finally:
            await pipeline.close()
            await db.close_pool()

    asyncio.run(_run())

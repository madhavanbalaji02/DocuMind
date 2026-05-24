"""pgvector hybrid retriever — drop-in replacement for src/rag/retriever.py HybridRetriever.

Uses pgvector RDS instead of Qdrant + BM25.
Injects PARENT context into retrieved chunks for better LLM answers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from aws.pgvector.client import PgVectorClient, SearchResult
from src.rag.reranker import CrossEncoderReranker
from src.embeddings.embedding_service import EmbeddingService
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Compatible with src/rag/retriever.RetrievedChunk interface."""
    chunk_id: str
    text: str
    source: str
    score: float
    rank: int
    metadata: dict[str, Any] | None = None
    parent_context: str = ""


class PgVectorHybridRetriever:
    """Hybrid retrieval from pgvector RDS with parent context injection.

    Drop-in replacement for HybridRetriever in src/rag/retriever.py.
    """

    def __init__(
        self,
        top_k_retrieve: int = 20,
        top_n_rerank: int = 5,
    ) -> None:
        self.embedder = EmbeddingService()
        self.client = PgVectorClient()
        self.reranker = CrossEncoderReranker()
        self.top_k = top_k_retrieve
        self.top_n = top_n_rerank

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        session_id: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Hybrid search: dense + FTS → RRF → CrossEncoder rerank → parent inject."""
        k = top_k or self.top_k

        # Embed query (local bge-small, <100ms CPU)
        query_embedding = await self.embedder.embed_query(query)

        # Hybrid search in pgvector
        loop = asyncio.get_event_loop()
        raw_results: list[SearchResult] = await loop.run_in_executor(
            None,
            lambda: self.client.hybrid_search(query, query_embedding, top_k=k),
        )

        if not raw_results:
            logger.info("pgvector: no results for %r", query[:60])
            return []

        # Convert to RetrievedChunk for CrossEncoder
        chunks = [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                text=r.content,
                source=r.filename or r.s3_key,
                score=r.score,
                rank=r.rank,
                metadata={
                    "element_type": r.element_type,
                    "page_number": r.page_number,
                    "document_id": r.document_id,
                    "entities": r.entities,
                },
                parent_context=r.parent_content,
            )
            for r in raw_results
        ]

        # CrossEncoder rerank (same as existing reranker)
        # Create temporary RetrievedChunk-compatible objects for reranker
        from src.rag.retriever import RetrievedChunk as LegacyChunk
        legacy_chunks = [
            LegacyChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                score=c.score,
                rank=c.rank,
                metadata=c.metadata,
            )
            for c in chunks
        ]
        reranked_legacy = await self.reranker.rerank(query, legacy_chunks, top_n=self.top_n)

        # Map back and inject parent context
        chunk_by_id = {c.chunk_id: c for c in chunks}
        reranked: list[RetrievedChunk] = []
        for i, lc in enumerate(reranked_legacy):
            original = chunk_by_id.get(lc.chunk_id)
            rc = RetrievedChunk(
                chunk_id=lc.chunk_id,
                text=lc.text,
                source=lc.source,
                score=lc.score,
                rank=i + 1,
                metadata=lc.metadata,
                parent_context=original.parent_context if original else "",
            )
            reranked.append(rc)

        logger.info(
            "pgvector retrieve %r: raw=%d, reranked=%d",
            query[:60], len(chunks), len(reranked),
        )

        if session_id:
            await self._save_retrievals(session_id, query, reranked)

        return reranked

    async def _save_retrievals(
        self, session_id: str, query: str, chunks: list[RetrievedChunk]
    ) -> None:
        import re
        import uuid
        from src.db import connection as db
        _UUID_RE = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        if not _UUID_RE.match(session_id):
            return
        for chunk in chunks:
            try:
                await db.execute(
                    """
                    INSERT INTO retrievals
                        (id, session_id, query, chunk_id, document_source,
                         relevance_score, content_preview)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7)
                    """,
                    str(uuid.uuid4()), session_id, query,
                    chunk.chunk_id, chunk.source, chunk.score, chunk.text[:500],
                )
            except Exception:
                pass

    async def close(self) -> None:
        await self.embedder.close()

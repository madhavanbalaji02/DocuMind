"""Hybrid retriever: dense Qdrant search + BM25 sparse, fused via RRF."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient

from src.db import connection as db
from src.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

RRF_K = 60  # constant in Reciprocal Rank Fusion


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score: float
    rank: int
    metadata: dict[str, Any] | None = None


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def _reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Merge dense and sparse ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + _rrf_score(rank)
        chunks_by_id[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + _rrf_score(rank)
        if chunk.chunk_id not in chunks_by_id:
            chunks_by_id[chunk.chunk_id] = chunk

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    fused = []
    for final_rank, cid in enumerate(sorted_ids, start=1):
        chunk = chunks_by_id[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                score=scores[cid],
                rank=final_rank,
                metadata=chunk.metadata,
            )
        )
    return fused


class HybridRetriever:
    """Dense + sparse hybrid search with RRF fusion."""

    def __init__(self) -> None:
        self.embedder = EmbeddingService()
        self.qdrant = AsyncQdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
        self.collection = os.environ.get("QDRANT_COLLECTION", "documind_chunks")

        # BM25 index is built lazily on first query
        self._bm25_corpus: list[str] = []
        self._bm25_chunk_ids: list[str] = []
        self._bm25_index: Any = None

    async def _build_bm25_index(self) -> None:
        """Fetch all chunk texts from Qdrant and build a BM25 index."""
        from rank_bm25 import BM25Okapi

        logger.info("Building BM25 index from Qdrant collection %r …", self.collection)
        offset = None
        texts: list[str] = []
        ids: list[str] = []

        while True:
            result, next_offset = await self.qdrant.scroll(
                collection_name=self.collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not result:
                break
            for point in result:
                payload = point.payload or {}
                texts.append(payload.get("text", ""))
                ids.append(str(point.id))
            if next_offset is None:
                break
            offset = next_offset

        self._bm25_corpus = texts
        self._bm25_chunk_ids = ids
        tokenized = [t.lower().split() for t in texts]
        self._bm25_index = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d documents", len(texts))

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        session_id: str | None = None,
        score_threshold: float = 0.5,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval returning top_k fused chunks."""
        # Dense search
        query_vector = await self.embedder.embed_query(query)
        qdrant_results = await self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k * 2,
            score_threshold=score_threshold,
            with_payload=True,
        )
        dense_chunks = [
            RetrievedChunk(
                chunk_id=str(r.id),
                text=r.payload.get("text", ""),
                source=r.payload.get("source", ""),
                score=r.score,
                rank=idx + 1,
                metadata=r.payload,
            )
            for idx, r in enumerate(qdrant_results)
        ]

        # Sparse BM25 search
        if self._bm25_index is None:
            await self._build_bm25_index()

        tokenized_query = query.lower().split()
        bm25_scores = self._bm25_index.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: top_k * 2]

        sparse_chunks = [
            RetrievedChunk(
                chunk_id=self._bm25_chunk_ids[i],
                text=self._bm25_corpus[i],
                source="",
                score=float(bm25_scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_bm25_indices)
            if bm25_scores[i] > 0
        ]

        fused = _reciprocal_rank_fusion(dense_chunks, sparse_chunks, top_k)
        logger.info(
            "Retrieval for %r: dense=%d, sparse=%d, fused=%d",
            query[:60],
            len(dense_chunks),
            len(sparse_chunks),
            len(fused),
        )

        if session_id:
            await self._save_retrievals(session_id, query, fused)

        return fused

    async def _save_retrievals(
        self, session_id: str, query: str, chunks: list[RetrievedChunk]
    ) -> None:
        """Persist retrieval records to postgres for audit/analysis."""
        for chunk in chunks:
            await db.execute(
                """
                INSERT INTO retrievals
                    (id, session_id, query, chunk_id, document_source, relevance_score, content_preview)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7)
                """,
                str(uuid.uuid4()),
                session_id,
                query,
                chunk.chunk_id,
                chunk.source,
                chunk.score,
                chunk.text[:500],
            )

    async def close(self) -> None:
        await self.qdrant.close()
        await self.embedder.close()

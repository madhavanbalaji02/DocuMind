"""Cross-encoder reranking to refine retrieval results."""

from __future__ import annotations

import asyncio
from src.core.logging import get_logger
from typing import Any

from src.rag.retriever import RetrievedChunk

logger = get_logger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level CrossEncoder cache — thread-safe after loading.
_CROSS_ENCODER_CACHE: dict[str, Any] = {}


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = _CROSS_ENCODER_MODEL) -> None:
        self.model_name = model_name

    @property
    def _model(self):
        return _CROSS_ENCODER_CACHE.get(self.model_name)

    def _load_model(self) -> None:
        if self.model_name not in _CROSS_ENCODER_CACHE:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder model: %s (will cache)", self.model_name)
            _CROSS_ENCODER_CACHE[self.model_name] = CrossEncoder(self.model_name)
            logger.info("CrossEncoder model cached")

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int = 5,
    ) -> list[RetrievedChunk]:
        """Return top_n chunks sorted by cross-encoder relevance score."""
        if not chunks:
            return []

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        model = _CROSS_ENCODER_CACHE[self.model_name]
        pairs = [[query, chunk.text] for chunk in chunks]

        scores: list[float] = await loop.run_in_executor(
            None, model.predict, pairs
        )

        scored = sorted(
            zip(chunks, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        reranked = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                score=float(score),
                rank=rank + 1,
                metadata=chunk.metadata,
            )
            for rank, (chunk, score) in enumerate(scored)
        ]

        logger.info(
            "Reranked %d → %d chunks (top score=%.3f)",
            len(chunks),
            len(reranked),
            reranked[0].score if reranked else 0.0,
        )
        return reranked

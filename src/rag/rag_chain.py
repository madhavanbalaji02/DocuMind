"""Full RAG chain: retrieve → rerank → LLM answer with citation tracking."""

from __future__ import annotations

from src.core.logging import get_logger
import os
from dataclasses import dataclass, field

import anthropic

from src.rag.retriever import HybridRetriever, RetrievedChunk
from src.rag.reranker import CrossEncoderReranker

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using ONLY the provided context passages.

Rules:
1. Base every claim on a specific passage. If the context does not contain enough information to answer fully, say so explicitly.
2. At the end of your answer, include a "Sources:" section listing each passage you cited, formatted as:
   [N] <source_file_or_url> — "<brief excerpt>"
3. Do not fabricate facts, URLs, or citations.
4. Be concise but complete. Use markdown formatting where helpful.
"""


@dataclass
class Citation:
    source: str
    chunk_id: str
    relevance_score: float
    excerpt: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    confidence: float


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {chunk.source}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _estimate_confidence(chunks: list[RetrievedChunk]) -> float:
    """Heuristic confidence based on top reranked score and chunk count."""
    if not chunks:
        return 0.0
    top_score = chunks[0].score
    # Normalise cross-encoder logits (roughly -10 to 10) into [0, 1]
    import math
    sigmoid = 1.0 / (1.0 + math.exp(-top_score))
    coverage_factor = min(len(chunks) / 5, 1.0)
    return round(sigmoid * coverage_factor, 3)


class RAGChain:
    """Orchestrates retrieval, reranking, and LLM generation with citations."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        top_k_retrieve: int = 10,
        top_n_rerank: int = 5,
        llm_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank = top_n_rerank
        self.llm_model = llm_model
        self._anthropic = anthropic.AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    async def query(self, question: str, session_id: str) -> RAGResponse:
        """Full RAG pipeline: retrieve → rerank → generate answer."""
        logger.info("RAG query for session=%s: %r", session_id, question[:80])

        # Retrieve
        raw_chunks = await self.retriever.retrieve(
            query=question,
            top_k=self.top_k_retrieve,
            session_id=session_id,
        )
        if not raw_chunks:
            return RAGResponse(
                answer="No relevant documents found in the knowledge base for this query.",
                sources=[],
                retrieved_chunks=[],
                confidence=0.0,
            )

        # Rerank
        reranked = await self.reranker.rerank(
            query=question,
            chunks=raw_chunks,
            top_n=self.top_n_rerank,
        )

        context = _format_context(reranked)
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        # Generate
        response = await self._anthropic.messages.create(
            model=self.llm_model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text

        sources = [
            Citation(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                relevance_score=chunk.score,
                excerpt=chunk.text[:200],
            )
            for chunk in reranked
        ]
        confidence = _estimate_confidence(reranked)

        logger.info(
            "RAG answer generated: %d chars, confidence=%.3f, sources=%d",
            len(answer),
            confidence,
            len(sources),
        )
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=reranked,
            confidence=confidence,
        )

    async def close(self) -> None:
        await self.retriever.close()

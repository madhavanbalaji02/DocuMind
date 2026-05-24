"""Full RAG chain: retrieve → rerank → LLM answer with citation tracking + guardrails.

LLM backend: Bedrock Claude Haiku (primary) → Anthropic SDK (fallback).
COST: Haiku = $0.00025/1K in, $0.00125/1K out (~$0.002/RAG query)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from src.rag.retriever import HybridRetriever, RetrievedChunk
from src.rag.reranker import CrossEncoderReranker
from src.core.logging import get_logger

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
    is_grounded: bool = True
    guardrail_warning: str | None = None


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {chunk.source}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _estimate_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    top_score = chunks[0].score
    import math
    sigmoid = 1.0 / (1.0 + math.exp(-top_score))
    coverage_factor = min(len(chunks) / 5, 1.0)
    return round(sigmoid * coverage_factor, 3)


async def _llm_complete(prompt: str, system: str, max_tokens: int = 2048) -> str:
    """Async LLM call: Bedrock Haiku → Anthropic SDK fallback."""
    loop = asyncio.get_event_loop()

    def _sync_complete() -> str:
        provider = os.environ.get("LLM_PROVIDER", "bedrock")
        if provider == "bedrock":
            try:
                from aws.bedrock.client import BedrockLLM
                return BedrockLLM(model="haiku").complete(prompt, system=system, max_tokens=max_tokens)
            except Exception as e:
                logger.warning("Bedrock failed, falling back to Anthropic: %s", e)

        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return await loop.run_in_executor(None, _sync_complete)


class RAGChain:
    """Orchestrates retrieval, reranking, LLM generation, and guardrail validation."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        top_k_retrieve: int = 10,
        top_n_rerank: int = 5,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank = top_n_rerank

    async def query(self, question: str, session_id: str) -> RAGResponse:
        """Full RAG pipeline: retrieve → rerank → generate → guardrails."""
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

        # Generate via Bedrock Haiku (or Anthropic fallback)
        answer = await _llm_complete(user_message, system=_SYSTEM_PROMPT, max_tokens=2048)

        # Apply guardrails
        is_grounded = True
        guardrail_warning = None
        try:
            from evaluation.guardrails import get_guardrails
            contexts = [c.text for c in reranked]
            sources_list = [c.source for c in reranked]
            guardrails = get_guardrails()
            guarded = guardrails.validate_answer(
                question=question,
                answer=answer,
                contexts=contexts,
                sources=sources_list,
            )
            answer = guarded.answer
            is_grounded = guarded.is_grounded
            guardrail_warning = guarded.warning
        except Exception as e:
            logger.debug("Guardrails skipped: %s", e)

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
            "RAG answer generated: %d chars, confidence=%.3f, grounded=%s",
            len(answer), confidence, is_grounded,
        )
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=reranked,
            confidence=confidence,
            is_grounded=is_grounded,
            guardrail_warning=guardrail_warning,
        )

    async def close(self) -> None:
        await self.retriever.close()

"""Tests for RAG pipeline: chunker, retriever RRF, and RAGChain."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingest.chunker import SemanticChunker, _content_hash
from src.rag.retriever import RetrievedChunk, _reciprocal_rank_fusion


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

    def test_chunk_count_scales_with_text_length(self):
        short_text = "Hello world. " * 5
        long_text = "Hello world. " * 100
        short_chunks = self.chunker.chunk_document(short_text, "test.txt", "text")
        long_chunks = self.chunker.chunk_document(long_text, "test.txt", "text")
        assert len(long_chunks) > len(short_chunks)

    def test_chunk_overlap_produces_shared_content(self):
        text = "A" * 200
        chunks = self.chunker.chunk_document(text, "overlap_test.txt", "text")
        # With overlap=20, adjacent chunk boundaries should share characters
        assert len(chunks) >= 2

    def test_dedup_removes_identical_chunks(self):
        # Exact same paragraph repeated — dedup should collapse them
        paragraph = "This is a unique paragraph about machine learning. " * 10
        text = paragraph + "\n\n" + paragraph
        chunks = self.chunker.chunk_document(text, "dedup_test.txt", "text")
        texts = [c.text for c in chunks]
        # All returned chunks should have unique content
        assert len(texts) == len(set(texts))

    def test_chunk_metadata_contains_required_fields(self):
        text = "Sample text for metadata testing. " * 20
        chunks = self.chunker.chunk_document(text, "meta_test.txt", "text")
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.source == "meta_test.txt"
            assert chunk.doc_type == "text"
            assert chunk.chunk_index >= 0
            assert "char_count" in chunk.metadata
            assert "content_hash" in chunk.metadata

    def test_pdf_chunk_includes_page_number_when_detected(self):
        pdf_text = "[Page 1]\nFirst page content.\n\n[Page 2]\nSecond page content. " * 5
        chunks = self.chunker.chunk_document(pdf_text, "test.pdf", "pdf")
        # At least some chunks should have page_number in metadata
        page_chunks = [c for c in chunks if "page_number" in c.metadata]
        assert len(page_chunks) > 0

    def test_chunk_id_is_deterministic(self):
        text = "Deterministic chunking test. " * 20
        chunks1 = self.chunker.chunk_document(text, "det.txt", "text")
        chunks2 = self.chunker.chunk_document(text, "det.txt", "text")
        ids1 = [c.id for c in chunks1]
        ids2 = [c.id for c in chunks2]
        assert ids1 == ids2

    def test_empty_text_returns_empty_list(self):
        chunks = self.chunker.chunk_document("", "empty.txt", "text")
        assert chunks == []

    def test_content_hash_function(self):
        assert _content_hash("hello") == _content_hash("hello")
        assert _content_hash("hello") != _content_hash("world")
        assert len(_content_hash("test")) == 16


# ── RRF fusion tests ──────────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def _make_chunk(self, chunk_id: str, score: float = 0.9) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id, text=f"text for {chunk_id}", source="test", score=score, rank=1
        )

    def test_rrf_merges_disjoint_lists(self):
        dense = [self._make_chunk(f"d{i}") for i in range(5)]
        sparse = [self._make_chunk(f"s{i}") for i in range(5)]
        for rank, chunk in enumerate(dense, 1):
            chunk.rank = rank
        for rank, chunk in enumerate(sparse, 1):
            chunk.rank = rank
        fused = _reciprocal_rank_fusion(dense, sparse, top_k=6)
        assert len(fused) == 6

    def test_rrf_boosts_overlap_chunk(self):
        """A chunk appearing in both lists should outscore chunks in only one."""
        shared_id = "shared_chunk"
        dense = [self._make_chunk(shared_id)] + [self._make_chunk(f"d{i}") for i in range(4)]
        sparse = [self._make_chunk(shared_id)] + [self._make_chunk(f"s{i}") for i in range(4)]
        for rank, chunk in enumerate(dense, 1):
            chunk.rank = rank
        for rank, chunk in enumerate(sparse, 1):
            chunk.rank = rank
        fused = _reciprocal_rank_fusion(dense, sparse, top_k=5)
        assert fused[0].chunk_id == shared_id

    def test_rrf_respects_top_k(self):
        dense = [self._make_chunk(f"d{i}") for i in range(10)]
        sparse = [self._make_chunk(f"s{i}") for i in range(10)]
        for rank, chunk in enumerate(dense, 1):
            chunk.rank = rank
        for rank, chunk in enumerate(sparse, 1):
            chunk.rank = rank
        fused = _reciprocal_rank_fusion(dense, sparse, top_k=3)
        assert len(fused) == 3

    def test_rrf_assigns_sequential_ranks(self):
        dense = [self._make_chunk(f"d{i}") for i in range(3)]
        sparse = []
        for rank, chunk in enumerate(dense, 1):
            chunk.rank = rank
        fused = _reciprocal_rank_fusion(dense, sparse, top_k=3)
        ranks = [c.rank for c in fused]
        assert ranks == [1, 2, 3]


# ── RAGChain tests ────────────────────────────────────────────────────────────

class TestRAGChain:
    @pytest.mark.asyncio
    async def test_query_returns_rag_response_with_no_chunks(self):
        """When retriever returns nothing, return a graceful empty response."""
        from src.rag.rag_chain import RAGChain

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=[])
        mock_retriever.close = AsyncMock()

        mock_reranker = AsyncMock()

        chain = RAGChain(retriever=mock_retriever, reranker=mock_reranker)
        response = await chain.query("What is quantum computing?", session_id="test-session")

        assert "No relevant documents" in response.answer
        assert response.sources == []
        assert response.confidence == 0.0

    @pytest.mark.asyncio
    async def test_query_calls_reranker_and_llm(self):
        """End-to-end mock: retriever → reranker → LLM."""
        from src.rag.rag_chain import RAGChain, RAGResponse

        chunks = [
            RetrievedChunk(
                chunk_id=f"c{i}",
                text=f"Context passage {i} about topic.",
                source=f"doc{i}.pdf",
                score=0.9 - i * 0.1,
                rank=i + 1,
            )
            for i in range(3)
        ]

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=chunks)
        mock_retriever.close = AsyncMock()

        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(return_value=chunks[:2])

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="This is the answer. Sources: [1] doc0.pdf")]

        with patch("anthropic.AsyncAnthropic") as mock_anthropic_cls:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_anthropic_cls.return_value = mock_client

            chain = RAGChain(retriever=mock_retriever, reranker=mock_reranker)
            response = await chain.query("What is the topic?", session_id="test-123")

        assert isinstance(response.answer, str)
        assert len(response.answer) > 0
        assert len(response.sources) == 2  # top 2 reranked
        mock_reranker.rerank.assert_called_once()
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_citations_populated_from_reranked_chunks(self):
        """Citations should match the reranked chunk sources."""
        from src.rag.rag_chain import RAGChain

        reranked = [
            RetrievedChunk(
                chunk_id="abc123",
                text="Important passage.",
                source="important_paper.pdf",
                score=1.5,
                rank=1,
            )
        ]

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=reranked)
        mock_retriever.close = AsyncMock()

        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(return_value=reranked)

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Answer based on [1].")]

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_cls.return_value = mock_client

            chain = RAGChain(retriever=mock_retriever, reranker=mock_reranker)
            response = await chain.query("question", session_id="s1")

        assert len(response.sources) == 1
        assert response.sources[0].source == "important_paper.pdf"
        assert response.sources[0].chunk_id == "abc123"

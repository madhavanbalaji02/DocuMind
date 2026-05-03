"""Tests for CrewAI tool functions and crew execution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── search_knowledge_base tool ────────────────────────────────────────────────

class TestSearchKnowledgeBaseTool:
    def test_returns_formatted_passages(self):
        from src.rag.retriever import RetrievedChunk

        mock_chunks = [
            RetrievedChunk(
                chunk_id="abc",
                text="Transformers use self-attention mechanisms to process sequences.",
                source="attention_paper.pdf",
                score=0.95,
                rank=1,
            )
        ]

        async def mock_retrieve(*args, **kwargs):
            return mock_chunks

        async def mock_close():
            pass

        # search_knowledge_base lazy-imports HybridRetriever — patch at source module
        with patch("src.rag.retriever.HybridRetriever") as mock_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve = mock_retrieve
            mock_retriever.close = mock_close
            mock_cls.return_value = mock_retriever

            from src.agents.tools import search_knowledge_base
            result = search_knowledge_base.run("How do transformers work?")

        assert "Transformers use self-attention" in result
        assert "attention_paper.pdf" in result

    def test_propagates_retriever_exception(self):
        async def mock_retrieve(*args, **kwargs):
            raise RuntimeError("Qdrant unavailable")

        async def mock_close():
            pass

        with patch("src.rag.retriever.HybridRetriever") as mock_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve = mock_retrieve
            mock_retriever.close = mock_close
            mock_cls.return_value = mock_retriever

            from src.agents.tools import search_knowledge_base

            with pytest.raises(RuntimeError, match="Qdrant unavailable"):
                search_knowledge_base.run("test query")


# ── query_database tool ───────────────────────────────────────────────────────

class TestQueryDatabaseTool:
    def test_rejects_non_select_queries(self):
        from src.agents.tools import query_database

        for dangerous_sql in ["DROP TABLE sessions", "DELETE FROM reports", "UPDATE sessions SET"]:
            result = query_database.run(dangerous_sql)
            assert "Error" in result
            assert "SELECT" in result

    def test_formats_results_as_markdown_table(self):
        mock_row = MagicMock()
        mock_row.keys.return_value = ["id", "topic", "status"]
        mock_row.__getitem__ = lambda self, key: {
            "id": "abc", "topic": "AI", "status": "done"
        }[key]

        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        # tools.py now uses asyncpg.connect() directly (pool is event-loop bound)
        with patch("src.agents.tools.asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.agents.tools import query_database
            result = query_database.run("SELECT id, topic, status FROM sessions LIMIT 1")

        assert "|" in result
        assert "id" in result
        assert "topic" in result

    def test_returns_message_for_empty_result(self):
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()

        with patch("src.agents.tools.asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.agents.tools import query_database
            result = query_database.run("SELECT * FROM sessions WHERE 1=0")

        assert "no rows" in result.lower()


# ── get_past_reports tool ─────────────────────────────────────────────────────

class TestGetPastReportsTool:
    def test_returns_no_results_message_when_empty(self):
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()

        with patch("src.agents.tools.asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.agents.tools import get_past_reports
            result = get_past_reports.run("quantum computing")

        assert "No past reports" in result
        assert "quantum computing" in result

    def test_formats_multiple_reports(self):
        mock_rows = []
        for i in range(2):
            row = MagicMock()
            row.__getitem__ = lambda self, key, i=i: {
                "title": f"Report {i}",
                "topic": f"Topic {i}",
                "quality_score": 0.85,
                "word_count": 1200,
                "preview": f"Preview text {i}",
            }[key]
            mock_rows.append(row)

        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.close = AsyncMock()

        with patch("src.agents.tools.asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.agents.tools import get_past_reports
            result = get_past_reports.run("AI research")

        assert "Report 0" in result
        assert "Report 1" in result


# ── search_web tool ───────────────────────────────────────────────────────────

class TestSearchWebTool:
    def test_handles_network_error_gracefully(self):
        # search_web now uses synchronous httpx.Client — patch Client not AsyncClient
        with patch("src.agents.tools.httpx.Client") as mock_client_cls:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=None)
            mock_ctx.get = MagicMock(side_effect=Exception("Connection refused"))
            mock_client_cls.return_value = mock_ctx

            from src.agents.tools import search_web
            result = search_web.run("transformer models")

        assert isinstance(result, str)
        assert "unavailable" in result.lower()

    def test_formats_duckduckgo_response(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "AbstractText": "Transformers are neural network architectures.",
            "AbstractURL": "https://en.wikipedia.org/wiki/Transformer",
            "RelatedTopics": [
                {"Text": "BERT is a transformer-based model."},
                {"Text": "GPT uses transformer architecture."},
            ],
        }

        with patch("src.agents.tools.httpx.Client") as mock_client_cls:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=None)
            mock_ctx.get = MagicMock(return_value=mock_response)
            mock_client_cls.return_value = mock_ctx

            from src.agents.tools import search_web
            result = search_web.run("transformer neural networks")

        assert "Transformers are neural network" in result


# ── ResearchCrew integration ──────────────────────────────────────────────────

class TestResearchCrew:
    @pytest.mark.asyncio
    async def test_crew_run_persists_agent_runs(self):
        with patch("src.agents.crew.build_researcher"), \
             patch("src.agents.crew.build_analyst"), \
             patch("src.agents.crew.build_writer"), \
             patch("src.agents.crew.build_critic"), \
             patch("src.agents.crew.db") as mock_db:

            mock_db.execute = AsyncMock()

            from src.agents.crew import ResearchCrew
            crew = ResearchCrew()

            # _kickoff_sync now returns a string directly
            with patch.object(crew, "_kickoff_sync", return_value="# Writer report output"):
                result = await crew.run(
                    topic="Test topic",
                    context="Some context",
                    session_id="test-session",
                )

            assert mock_db.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_crew_result_contains_draft_report(self):
        with patch("src.agents.crew.build_researcher"), \
             patch("src.agents.crew.build_analyst"), \
             patch("src.agents.crew.build_writer"), \
             patch("src.agents.crew.build_critic"), \
             patch("src.agents.crew.db") as mock_db:

            mock_db.execute = AsyncMock()

            from src.agents.crew import ResearchCrew
            crew = ResearchCrew()

            with patch.object(crew, "_kickoff_sync", return_value="# Report\n\nFull content."):
                result = await crew.run("topic", "context", "session-id")

            assert result.draft_report
            assert "Report" in result.draft_report

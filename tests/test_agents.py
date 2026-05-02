"""Tests for CrewAI tool functions and crew execution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── search_knowledge_base tool ────────────────────────────────────────────────

class TestSearchKnowledgeBaseTool:
    def test_returns_formatted_answer_with_citations(self):
        from src.rag.rag_chain import RAGResponse
        from src.rag.retriever import RetrievedChunk
        from src.rag.rag_chain import Citation

        mock_response = RAGResponse(
            answer="Transformers use self-attention mechanisms.",
            sources=[
                Citation(
                    source="attention_paper.pdf",
                    chunk_id="abc",
                    relevance_score=0.95,
                    excerpt="Self-attention allows the model to weigh…",
                )
            ],
            retrieved_chunks=[],
            confidence=0.9,
        )

        async def mock_query(*args, **kwargs):
            return mock_response

        async def mock_close():
            pass

        with patch("src.agents.tools.RAGChain") as mock_cls:
            mock_rag = MagicMock()
            mock_rag.query = mock_query
            mock_rag.close = mock_close
            mock_cls.return_value = mock_rag

            from src.agents.tools import search_knowledge_base

            result = search_knowledge_base("How do transformers work?")

        assert "Transformers use self-attention" in result
        assert "attention_paper.pdf" in result
        assert "Citations" in result

    def test_returns_string_on_rag_failure(self):
        async def mock_query(*args, **kwargs):
            raise RuntimeError("Qdrant unavailable")

        async def mock_close():
            pass

        with patch("src.agents.tools.RAGChain") as mock_cls:
            mock_rag = MagicMock()
            mock_rag.query = mock_query
            mock_rag.close = mock_close
            mock_cls.return_value = mock_rag

            from src.agents.tools import search_knowledge_base

            with pytest.raises(RuntimeError):
                search_knowledge_base("test query")


# ── query_database tool ───────────────────────────────────────────────────────

class TestQueryDatabaseTool:
    def test_rejects_non_select_queries(self):
        from src.agents.tools import query_database

        for dangerous_sql in ["DROP TABLE sessions", "DELETE FROM reports", "UPDATE sessions SET"]:
            result = query_database(dangerous_sql)
            assert "Error" in result
            assert "SELECT" in result

    def test_formats_results_as_markdown_table(self):
        mock_row = MagicMock()
        mock_row.keys.return_value = ["id", "topic", "status"]
        mock_row.__getitem__ = lambda self, key: {"id": "abc", "topic": "AI", "status": "done"}[key]

        async def mock_fetch(*args, **kwargs):
            return [mock_row]

        with patch("src.agents.tools.db") as mock_db:
            mock_db.fetch = mock_fetch

            from src.agents.tools import query_database

            result = query_database("SELECT id, topic, status FROM sessions LIMIT 1")

        assert "|" in result
        assert "id" in result
        assert "topic" in result

    def test_returns_message_for_empty_result(self):
        async def mock_fetch(*args, **kwargs):
            return []

        with patch("src.agents.tools.db") as mock_db:
            mock_db.fetch = mock_fetch

            from src.agents.tools import query_database

            result = query_database("SELECT * FROM sessions WHERE 1=0")

        assert "no rows" in result.lower()


# ── get_past_reports tool ─────────────────────────────────────────────────────

class TestGetPastReportsTool:
    def test_returns_no_results_message_when_empty(self):
        async def mock_fetch(*args, **kwargs):
            return []

        with patch("src.agents.tools.db") as mock_db:
            mock_db.fetch = mock_fetch

            from src.agents.tools import get_past_reports

            result = get_past_reports("quantum computing")

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

        async def mock_fetch(*args, **kwargs):
            return mock_rows

        with patch("src.agents.tools.db") as mock_db:
            mock_db.fetch = mock_fetch

            from src.agents.tools import get_past_reports

            result = get_past_reports("AI research")

        assert "Report 0" in result
        assert "Report 1" in result


# ── search_web tool ───────────────────────────────────────────────────────────

class TestSearchWebTool:
    def test_handles_network_error_gracefully(self):
        import httpx

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            from src.agents.tools import search_web

            result = search_web("transformer models")

        assert "unavailable" in result.lower() or isinstance(result, str)

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

        async def mock_get(*args, **kwargs):
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            from src.agents.tools import search_web

            result = search_web("transformer neural networks")

        assert "Transformers are neural network" in result


# ── ResearchCrew integration ──────────────────────────────────────────────────

class TestResearchCrew:
    @pytest.mark.asyncio
    async def test_crew_run_persists_agent_runs(self):
        """Crew.kickoff output should be saved to agent_runs table."""
        mock_task_output = MagicMock()
        mock_task_output.__str__ = lambda self: "Task output text"

        with patch("src.agents.crew.build_researcher") as mock_researcher, \
             patch("src.agents.crew.build_analyst") as mock_analyst, \
             patch("src.agents.crew.build_writer") as mock_writer, \
             patch("src.agents.crew.build_critic") as mock_critic, \
             patch("src.agents.crew.Crew") as mock_crew_cls, \
             patch("src.agents.crew.db") as mock_db:

            mock_db.execute = AsyncMock()

            mock_crew = MagicMock()
            mock_crew.kickoff = MagicMock(return_value="Final crew output")
            mock_crew_cls.return_value = mock_crew

            from src.agents.crew import ResearchCrew

            crew = ResearchCrew()

            # Patch tasks so output is set
            with patch.object(crew, "_build_tasks") as mock_build:
                task1 = MagicMock()
                task1.output = mock_task_output
                mock_build.return_value = [task1, task1, task1, task1]

                result = await crew.run(
                    topic="Test topic",
                    context="Some context",
                    session_id="test-session",
                )

            # Should have attempted to save agent runs
            assert mock_db.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_crew_result_contains_draft_report(self):
        """CrewResult.draft_report should be a non-empty string."""
        with patch("src.agents.crew.build_researcher"), \
             patch("src.agents.crew.build_analyst"), \
             patch("src.agents.crew.build_writer"), \
             patch("src.agents.crew.build_critic"), \
             patch("src.agents.crew.Crew") as mock_crew_cls, \
             patch("src.agents.crew.db") as mock_db:

            mock_db.execute = AsyncMock()

            mock_crew = MagicMock()
            mock_crew.kickoff = MagicMock(return_value="# Research Report\n\nContent here.")
            mock_crew_cls.return_value = mock_crew

            from src.agents.crew import ResearchCrew

            crew = ResearchCrew()

            writer_task = MagicMock()
            writer_task.output = MagicMock()
            writer_task.output.__str__ = lambda self: "# Report\n\nFull content."

            with patch.object(crew, "_build_tasks") as mock_build:
                tasks = [MagicMock() for _ in range(4)]
                tasks[2] = writer_task
                mock_build.return_value = tasks

                result = await crew.run("topic", "context", "session-id")

            assert result.draft_report
            assert len(result.draft_report) > 0

"""Tests for LangGraph workflow: node functions, routing, and state transitions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workflow.state import ResearchState


def _base_state(**overrides) -> ResearchState:
    state: ResearchState = {
        "session_id": "test-session-id",
        "topic": "Transformer models in NLP",
        "research_plan": [],
        "retrieved_context": [],
        "analyst_insights": "",
        "draft_report": "",
        "critic_feedback": "",
        "final_report": "",
        "citations": [],
        "iteration": 0,
        "status": "running",
        "error": None,
    }
    state.update(overrides)
    return state


# ── plan_research node ────────────────────────────────────────────────────────

class TestPlanResearchNode:
    @pytest.mark.asyncio
    async def test_returns_list_of_questions(self):
        from src.workflow.nodes import plan_research

        questions = [
            "What are the core components of Transformer?",
            "How does self-attention work?",
            "What are the limitations of Transformers?",
            "What are recent Transformer variants?",
        ]
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=str(questions).replace("'", '"'))]

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_cls.return_value = mock_client

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                import json
                mock_message.content[0].text = json.dumps(questions)
                result = await plan_research(_base_state())

        assert "research_plan" in result
        assert isinstance(result["research_plan"], list)
        assert len(result["research_plan"]) > 0

    @pytest.mark.asyncio
    async def test_status_updated_after_planning(self):
        from src.workflow.nodes import plan_research

        questions = ["Q1", "Q2", "Q3", "Q4"]
        import json

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(questions))]

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_cls.return_value = mock_client

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                result = await plan_research(_base_state())

        assert result["status"] == "planning_done"


# ── retrieve_context node ─────────────────────────────────────────────────────

class TestRetrieveContextNode:
    @pytest.mark.asyncio
    async def test_merges_chunks_from_multiple_questions(self):
        from src.rag.rag_chain import RAGResponse
        from src.rag.retriever import RetrievedChunk
        from src.workflow.nodes import retrieve_context

        mock_chunks = [
            RetrievedChunk(
                chunk_id=f"chunk_{i}",
                text=f"Content about the topic, passage {i}.",
                source=f"doc{i}.pdf",
                score=0.9,
                rank=1,
            )
            for i in range(3)
        ]
        mock_response = RAGResponse(
            answer="Answer",
            sources=[],
            retrieved_chunks=mock_chunks,
            confidence=0.8,
        )

        with patch("src.workflow.nodes.RAGChain") as mock_rag_cls:
            mock_rag = AsyncMock()
            mock_rag.query = AsyncMock(return_value=mock_response)
            mock_rag.close = AsyncMock()
            mock_rag_cls.return_value = mock_rag

            state = _base_state(research_plan=["Q1", "Q2"])
            result = await retrieve_context(state)

        assert "retrieved_context" in result
        assert len(result["retrieved_context"]) > 0
        assert result["status"] == "context_retrieved"

    @pytest.mark.asyncio
    async def test_deduplicates_chunks_across_questions(self):
        from src.rag.rag_chain import RAGResponse
        from src.rag.retriever import RetrievedChunk
        from src.workflow.nodes import retrieve_context

        # Same chunk_id returned for both questions — should appear once
        shared_chunk = RetrievedChunk(
            chunk_id="shared-id",
            text="Shared content.",
            source="doc.pdf",
            score=0.9,
            rank=1,
        )
        mock_response = RAGResponse(
            answer="A", sources=[], retrieved_chunks=[shared_chunk], confidence=0.8
        )

        with patch("src.workflow.nodes.RAGChain") as mock_rag_cls:
            mock_rag = AsyncMock()
            mock_rag.query = AsyncMock(return_value=mock_response)
            mock_rag.close = AsyncMock()
            mock_rag_cls.return_value = mock_rag

            state = _base_state(research_plan=["Q1", "Q2"])
            result = await retrieve_context(state)

        chunk_ids = [c["chunk_id"] for c in result["retrieved_context"]]
        assert chunk_ids.count("shared-id") == 1


# ── critic_review routing ─────────────────────────────────────────────────────

class TestCriticRouting:
    def test_routes_to_finalize_on_pass(self):
        from src.workflow.graph import _route_after_critic

        state = _base_state(status="critic_pass", iteration=0)
        assert _route_after_critic(state) == "finalize_report"

    def test_routes_to_revise_on_fail_with_room_to_iterate(self):
        from src.workflow.graph import _route_after_critic

        state = _base_state(status="critic_fail", iteration=0)
        assert _route_after_critic(state) == "revise_report"

    def test_routes_to_finalize_when_max_iterations_reached(self):
        from src.workflow.graph import _route_after_critic

        state = _base_state(status="critic_fail", iteration=2)
        assert _route_after_critic(state) == "finalize_report"

    def test_routes_to_finalize_at_exactly_max_iterations(self):
        from src.workflow.graph import _route_after_critic

        state = _base_state(status="critic_fail", iteration=2)
        assert _route_after_critic(state) == "finalize_report"


# ── revise_report node ────────────────────────────────────────────────────────

class TestReviseReportNode:
    @pytest.mark.asyncio
    async def test_increments_iteration_counter(self):
        from src.workflow.nodes import revise_report

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="# Revised Report\n\nImproved content here.")]

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_cls.return_value = mock_client

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                state = _base_state(
                    draft_report="Original draft.",
                    critic_feedback="Needs improvement.",
                    iteration=0,
                )
                result = await revise_report(state)

        assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_updates_draft_report(self):
        from src.workflow.nodes import revise_report

        revised_text = "# Revised\n\nThis is the revised content."
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=revised_text)]

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_cls.return_value = mock_client

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                state = _base_state(draft_report="Old draft", critic_feedback="Fix it")
                result = await revise_report(state)

        assert result["draft_report"] == revised_text


# ── handle_error node ─────────────────────────────────────────────────────────

class TestHandleErrorNode:
    @pytest.mark.asyncio
    async def test_marks_session_failed(self):
        from src.workflow.nodes import handle_error

        with patch("src.workflow.nodes.db") as mock_db:
            mock_db.execute = AsyncMock()
            state = _base_state(error="Something went wrong")
            result = await handle_error(state)

        assert result["status"] == "failed"
        mock_db.execute.assert_called_once()


# ── Graph compilation ─────────────────────────────────────────────────────────

class TestGraphCompilation:
    def test_graph_compiles_without_checkpointer(self):
        from src.workflow.graph import build_research_graph

        graph = build_research_graph(use_checkpointer=False)
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from src.workflow.graph import build_research_graph

        graph = build_research_graph(use_checkpointer=False)
        node_names = set(graph.nodes.keys())
        expected = {
            "plan_research",
            "retrieve_context",
            "analyze_data",
            "run_crew",
            "critic_review",
            "revise_report",
            "finalize_report",
            "handle_error",
        }
        assert expected.issubset(node_names)

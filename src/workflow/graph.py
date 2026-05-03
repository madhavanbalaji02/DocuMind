"""Assembles the LangGraph StateGraph for the research workflow."""

from __future__ import annotations

from src.core.logging import get_logger

from langgraph.graph import END, START, StateGraph

from src.workflow.nodes import (
    analyze_data,
    critic_review,
    finalize_report,
    handle_error,
    plan_research,
    retrieve_context,
    revise_report,
    run_crew,
)
from src.workflow.state import ResearchState

logger = get_logger(__name__)

MAX_ITERATIONS = 2


def _route_after_critic(state: ResearchState) -> str:
    """Route based on critic verdict and revision count."""
    if state["status"] == "critic_pass" or state["iteration"] >= MAX_ITERATIONS:
        return "finalize_report"
    return "revise_report"


def build_research_graph(use_checkpointer: bool = True):
    """Build and compile the research StateGraph.

    Args:
        use_checkpointer: When False the graph is compiled without a checkpointer
            (useful for unit tests that do not need postgres).
    """
    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("plan_research", plan_research)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("analyze_data", analyze_data)
    builder.add_node("run_crew", run_crew)
    builder.add_node("critic_review", critic_review)
    builder.add_node("revise_report", revise_report)
    builder.add_node("finalize_report", finalize_report)
    builder.add_node("handle_error", handle_error)

    # Linear edges
    builder.add_edge(START, "plan_research")
    builder.add_edge("plan_research", "retrieve_context")
    builder.add_edge("retrieve_context", "analyze_data")
    builder.add_edge("analyze_data", "run_crew")
    builder.add_edge("run_crew", "critic_review")

    # Conditional: critic pass/fail with revision loop
    builder.add_conditional_edges(
        "critic_review",
        _route_after_critic,
        {
            "finalize_report": "finalize_report",
            "revise_report": "revise_report",
        },
    )
    builder.add_edge("revise_report", "critic_review")
    builder.add_edge("finalize_report", END)
    builder.add_edge("handle_error", END)

    if use_checkpointer:
        from src.workflow.checkpointer import get_postgres_checkpointer

        checkpointer = get_postgres_checkpointer()
        compiled = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["critic_review"],
        )
        logger.info("Research graph compiled with PostgresCheckpointer")
    else:
        compiled = builder.compile()
        logger.info("Research graph compiled (no checkpointer)")

    return compiled

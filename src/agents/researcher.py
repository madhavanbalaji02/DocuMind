"""CrewAI Researcher agent definition."""

from __future__ import annotations

import os

from crewai import Agent, LLM

from src.agents.tools import search_knowledge_base, search_web


def build_researcher() -> Agent:
    llm = LLM(
        model="anthropic/claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return Agent(
        role="Senior Research Specialist",
        goal=(
            "Thoroughly investigate the research topic by querying the knowledge base "
            "and supplementing with web searches. Uncover key facts, data points, and "
            "perspectives that will form the foundation of the final report."
        ),
        backstory=(
            "You are a meticulous researcher with 15 years of experience in academic and "
            "industry research. You never accept surface-level answers — you dig deep, "
            "cross-reference sources, and surface non-obvious insights. You always cite "
            "the exact source of every claim you make."
        ),
        tools=[search_knowledge_base, search_web],
        llm=llm,
        verbose=True,
        max_iter=5,
    )

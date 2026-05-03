"""CrewAI Analyst agent definition."""

from __future__ import annotations

import os

from crewai import Agent, LLM

from src.agents.tools import query_database, search_knowledge_base


def build_analyst() -> Agent:
    llm = LLM(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return Agent(
        role="Data Analyst",
        goal="Analyse the research findings and surface key insights.",
        backstory=(
            "You are a concise analyst. You make at most 2 tool calls, "
            "then provide a focused analytical summary with confidence ratings."
        ),
        tools=[query_database, search_knowledge_base],
        llm=llm,
        verbose=False,
        max_iter=3,
        max_rpm=10,
    )

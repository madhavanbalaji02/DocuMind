"""CrewAI Researcher agent definition."""

from __future__ import annotations

import os

from crewai import Agent, LLM

from src.agents.tools import search_knowledge_base, search_web


def build_researcher() -> Agent:
    llm = LLM(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return Agent(
        role="Research Specialist",
        goal="Find accurate information about the topic using the knowledge base.",
        backstory=(
            "You are a focused researcher. You make at most 3 tool calls, "
            "then synthesise the results into a clear summary with citations."
        ),
        tools=[search_knowledge_base, search_web],
        llm=llm,
        verbose=False,
        max_iter=3,
        max_rpm=10,
    )

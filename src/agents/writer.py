"""CrewAI Writer agent definition."""

from __future__ import annotations

import os

from crewai import Agent, LLM


def build_writer() -> Agent:
    llm = LLM(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return Agent(
        role="Technical Writer",
        goal="Synthesise findings into a clear, well-cited markdown report.",
        backstory=(
            "You write directly from the provided context. "
            "No tool calls needed — your job is synthesis and clear writing."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        max_iter=2,
        max_rpm=10,
    )

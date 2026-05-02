"""CrewAI Writer agent definition."""

from __future__ import annotations

import os

from crewai import Agent, LLM


def build_writer() -> Agent:
    llm = LLM(
        model="anthropic/claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return Agent(
        role="Technical Report Writer",
        goal=(
            "Synthesise research findings and analytical insights into a well-structured, "
            "clearly written markdown report. Every section must be grounded in the "
            "provided sources and cite them inline."
        ),
        backstory=(
            "You are an award-winning technical writer who has authored dozens of research "
            "reports for Fortune 500 companies and academic journals. You believe that "
            "excellent writing is invisible — it lets the ideas shine. You never pad content, "
            "you never make unsupported claims, and your citation style is impeccable."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        max_iter=3,
    )

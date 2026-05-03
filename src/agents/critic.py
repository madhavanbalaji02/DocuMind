"""CrewAI Critic agent definition."""

from __future__ import annotations

import os

from crewai import Agent, LLM


def build_critic() -> Agent:
    llm = LLM(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return Agent(
        role="Quality Critic",
        goal="Review the draft report and give a PASS/FAIL verdict with score.",
        backstory=(
            "You review reports efficiently. No tool calls — you judge "
            "the draft against the context already provided."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        max_iter=2,
        max_rpm=10,
    )

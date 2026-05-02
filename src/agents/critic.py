"""CrewAI Critic agent definition."""

from __future__ import annotations

import os

from crewai import Agent, LLM

from src.agents.tools import search_knowledge_base


def build_critic() -> Agent:
    llm = LLM(
        model="anthropic/claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return Agent(
        role="Research Quality Critic",
        goal=(
            "Rigorously evaluate the draft report for factual accuracy, logical consistency, "
            "and completeness. Identify every unsupported claim and flag it. Provide "
            "actionable, specific feedback that enables the writer to produce a publication-ready report."
        ),
        backstory=(
            "You are the toughest peer reviewer in the business. You have rejected more "
            "papers than you've approved, and every rejection made the final work better. "
            "You verify every claim against source material, flag logical gaps, and "
            "demand that every assertion be traceable to evidence."
        ),
        tools=[search_knowledge_base],
        llm=llm,
        verbose=True,
        max_iter=3,
    )

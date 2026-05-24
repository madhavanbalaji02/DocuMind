"""CrewAI Analyst agent — uses Bedrock Claude Haiku (fallback: direct Anthropic)."""

from __future__ import annotations

import os

from crewai import Agent, LLM

from src.agents.tools import query_database, search_knowledge_base


def _build_llm() -> LLM:
    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    if provider == "bedrock":
        return LLM(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            aws_region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
        )
    return LLM(
        model="anthropic/claude-haiku-4-5",
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )


def build_analyst() -> Agent:
    return Agent(
        role="Data Analyst",
        goal="Analyse the research findings and surface key insights.",
        backstory=(
            "You are a concise analyst. You make at most 2 tool calls, "
            "then provide a focused analytical summary with confidence ratings."
        ),
        tools=[query_database, search_knowledge_base],
        llm=_build_llm(),
        verbose=False,
        max_iter=3,
        max_rpm=10,
    )

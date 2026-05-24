"""CrewAI Researcher agent — uses Bedrock Claude Haiku (fallback: direct Anthropic)."""

from __future__ import annotations

import os

from crewai import Agent, LLM

from src.agents.tools import search_knowledge_base, search_web


def _build_llm() -> LLM:
    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    if provider == "bedrock":
        # CrewAI supports Bedrock via boto3 under the hood through LiteLLM
        return LLM(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            aws_region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
        )
    return LLM(
        model="anthropic/claude-haiku-4-5",
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )


def build_researcher() -> Agent:
    return Agent(
        role="Research Specialist",
        goal="Find accurate information about the topic using the knowledge base.",
        backstory=(
            "You are a focused researcher. You make at most 3 tool calls, "
            "then synthesise the results into a clear summary with citations."
        ),
        tools=[search_knowledge_base, search_web],
        llm=_build_llm(),
        verbose=False,
        max_iter=3,
        max_rpm=10,
    )

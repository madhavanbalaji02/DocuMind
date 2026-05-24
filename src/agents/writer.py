"""CrewAI Writer agent — uses Bedrock Claude Haiku (fallback: direct Anthropic)."""

from __future__ import annotations

import os

from crewai import Agent, LLM


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


def build_writer() -> Agent:
    return Agent(
        role="Technical Writer",
        goal="Synthesise findings into a clear, well-cited markdown report.",
        backstory=(
            "You write directly from the provided context. "
            "No tool calls needed — your job is synthesis and clear writing."
        ),
        tools=[],
        llm=_build_llm(),
        verbose=False,
        max_iter=2,
        max_rpm=10,
    )
